"""Evaluate a paper-v1 checkpoint on an explicitly identified held-out split."""
import argparse
from src.dataset_spec import load_partitions, write_json, sha256
from src.checkpoints import checkpoint_metadata
from src.span_grounding import audit_gold
from src.paper_runtime import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/paper_split/manifest.json")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint_path, metadata = checkpoint_metadata(args.checkpoint)
    config, statistics = metadata["config"], metadata["statistics"]
    schema = metadata["metadata"]["schema"]
    if sha256(args.manifest) != metadata["metadata"]["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from checkpoint training identity")
    partitions, _ = load_partitions(args.manifest, schema)
    audit = audit_gold(partitions[args.split])
    if config.get("evaluation_mode", "paper_span") == "paper_span" and audit["issues"]:
        raise ValueError(f"Gold spans require resolution: {audit['issues'][:3]}")
    import torch
    from src.model import load_pretrained
    if config.get("smoke_only"):
        from src.testing_backbone import load_smoke
        model, tokenizer = load_smoke(config, statistics)
        used_ids = set(metadata["metadata"]["used_documents"][args.split])
        partitions[args.split] = [r for r in partitions[args.split] if r["doc_id"] in used_ids]
    else:
        model, tokenizer = load_pretrained(config, statistics, "cuda" if torch.cuda.is_available() else "cpu")
    model.load_checkpoint(checkpoint_path)
    report = evaluate(model, tokenizer, partitions[args.split], schema, config, statistics)
    report.update(split=args.split, manifest_sha256=sha256(args.manifest), checkpoint=str(checkpoint_path),
                  run_kind=metadata["metadata"].get("run_kind", "pretrained_model_training"))
    write_json(args.output, report)
    print(report["summary"])


if __name__ == "__main__": main()
