"""Train paper-v1 APLER; select checkpoints using free-running development Arg-C."""
import argparse
import json
import math
import random
from pathlib import Path
from src.dataset_spec import load_config, load_schema, load_partitions, write_json, sha256, code_identity
from src.latent_budget import training_statistics
from src.span_grounding import audit_gold
from src.paper_data import tokenize_example
from src.paper_runtime import evaluate, learning_rate_factor, DevelopmentSelection
from src.checkpoints import save_training_checkpoint, load_training_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/apler_emergency_data.json")
    parser.add_argument("--manifest", default="data/paper_split/manifest.json")
    parser.add_argument("--output", default="outputs/apler_emergency_data")
    parser.add_argument("--resume", action="store_true", help="Resume this output's last completed epoch")
    parser.add_argument("--smoke", action="store_true", help="Offline tiny RANDOM model on real data; no meaningful extraction results")
    parser.add_argument("--smoke-epochs", type=int, default=1, help="Epochs for the offline smoke model only")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop after committing this epoch; preserve the full schedule for --resume")
    args = parser.parse_args()
    if args.smoke_epochs < 1 or (args.smoke_epochs != 1 and not args.smoke):
        parser.error("--smoke-epochs requires --smoke and a positive epoch count")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    config = load_config(args.config); schema = load_schema(config["schema"])
    partitions, manifest = load_partitions(args.manifest, schema)
    statistics = training_statistics(partitions["train"])
    if args.smoke:
        config.update(smoke_only=True, model_name="local-random-smoke-only", distill_layers=[1,2,3,4],
                      bf16=False, epochs=args.smoke_epochs, gradient_accumulation_steps=2, input_window=8192, max_output_tokens=512)
        ordered = sorted(partitions["train"], key=lambda r: len(r["events"]))
        selected = [ordered[0], next(r for r in ordered if len(r["events"]) == 1), ordered[-1]]
        partitions = {"train": selected, "dev": partitions["dev"][:2], "test": partitions["test"][:2]}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if (output/"run.json").exists() and not args.resume:
        raise FileExistsError("Run already exists: use a new output or --resume")
    audits = {name: audit_gold(rows) for name, rows in partitions.items()}
    write_json(output/"gold_span_audit.json", audits)
    if config.get("evaluation_mode", "paper_span") == "paper_span" and any(a["issues"] for a in audits.values()):
        raise ValueError("Gold span ambiguity/unmapped values must be resolved; see gold_span_audit.json")
    import torch
    from src.model import load_pretrained
    random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        from src.testing_backbone import load_smoke
        model, tokenizer = load_smoke(config, statistics)
    else:
        model, tokenizer = load_pretrained(config, statistics, device)
    items, issues = [], []
    for split, rows in partitions.items():
        for row in rows:
            try:
                item = tokenize_example(row, tokenizer, schema, config, statistics, training=split == "train")
                model.validate_context(item, split == "train")
                if split == "train": items.append(item)
            except ValueError as exc: issues.append({"split": split, "doc_id": row["doc_id"], "error": str(exc)})
    write_json(output/"length_audit.json", issues)
    if issues: raise ValueError("Length preflight failed; no rows were silently filtered. See length_audit.json")
    metadata = {"manifest": manifest, "manifest_sha256": sha256(args.manifest), "schema": schema,
                "code_identity": code_identity(),
                "run_kind": "random_model_pipeline_smoke" if args.smoke else "pretrained_model_training",
                "used_documents": {k:[r["doc_id"] for r in v] for k,v in partitions.items()},
                "config": config, "statistics": statistics}
    if args.resume:
        from src.dataset_spec import read_json
        if read_json(output/"run.json") != metadata: raise ValueError("Resume data/config identity mismatch")
    else: write_json(output/"run.json", metadata)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    accumulation = config["gradient_accumulation_steps"]
    total = math.ceil(len(items)/accumulation)*config["epochs"]
    warmup = math.ceil(total*config["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, total, warmup, config["learning_rate"], config["min_learning_rate"]))
    selection = DevelopmentSelection(config["patience"])
    start_epoch, step = 0, 0
    if args.resume:
        checkpoint_path, _, state = load_training_checkpoint(output/"last")
        model.load_checkpoint(checkpoint_path)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        selection.best, selection.bad_epochs = state["best"], state["bad_epochs"]
        start_epoch, step = state["epoch"], state["step"]
        random.setstate(state["random_state"]); torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda_rng"])
    if selection.bad_epochs >= selection.patience: return
    if args.stop_after_epoch is not None and start_epoch >= args.stop_after_epoch: return
    for epoch in range(start_epoch, config["epochs"]):
        model.train(); order = list(range(len(items))); random.shuffle(order)
        for begin in range(0, len(order), accumulation):
            batch = order[begin:begin+accumulation]; optimizer.zero_grad(set_to_none=True)
            logs = {}
            for index in batch:
                result = model(items[index], progress=step/max(1, total))
                if not torch.isfinite(result["loss"]): raise FloatingPointError("Non-finite training loss")
                (result["loss"]/len(batch)).backward()
                for key, value in result.items(): logs[key] = logs.get(key, 0.0) + float(value.detach() if hasattr(value, "detach") else value)/len(batch)
            torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            record = {"epoch": epoch+1, "step": step, "learning_rate": optimizer.param_groups[0]["lr"], **logs}
            with (output/"training.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
        report = evaluate(model, tokenizer, partitions["dev"], schema, config, statistics)
        write_json(output/f"dev_epoch_{epoch+1}.json", report)
        improved, stop = selection.update(report["summary"]["arg_c"]["f1"])
        checkpoint_meta = {**metadata, "epoch": epoch+1, "step": step, "development": report["summary"]}
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best": selection.best,
                 "bad_epochs": selection.bad_epochs, "epoch": epoch+1, "step": step, "random_state": random.getstate(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        save_training_checkpoint(output, model, checkpoint_meta, state, is_best=improved)
        if stop or (args.stop_after_epoch is not None and epoch+1 >= args.stop_after_epoch): break
    print(f"Best checkpoint: {output / 'best'}; development Arg-C={selection.best:.6f}")


if __name__ == "__main__": main()
