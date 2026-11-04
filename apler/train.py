# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import logging
import os
import hashlib
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from math import ceil
from peft import LoraConfig, TaskType

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    load_tokenizer_with_fallback,
)
from src.event_utils import (
    EventDecoderTokenization,
    FINAL_JSON_MARKER,
    JSON_EVENT_ARRAY_PREFIX,
    PROCESS_DONE_MARKERS,
    PROCESS_MARKERS,
    build_event_prompt,
    build_document_token_mask_from_source,
    build_sentence_rep_positions_from_source,
    build_final_answer,
    build_teacher_process,
    canonical_events,
    event_decoder_context_length,
    encode_json_event_array_prefix,
    event_object_start_candidate_token_ids,
    example_complexity,
    filter_positive_event_examples,
    format_prompt_for_model,
    limit_dataset_for_debug,
    load_emergency_examples,
    require_data_name,
    resolve_event_identity_anchor,
    resolve_event_task,
    token_mask_from_char_spans,
    tokenize_event_decoder_sequence,
)
from src.event_decode_state import (
    at_completed_event_boundary,
    classify_event_token_transition,
    direct_event_boundary_token_groups,
    event_boundary_token_groups,
    inspect_event_decode_state,
    normalize_event_boundary_control,
)

IGNORE_INDEX = -100
JSON_TRAIN_PROXY_MASK_NAMES = ("trigger", "arg_i", "arg_c", "span", "head")
TRAIN_PROXY_METRIC_BASES = (
    "trigger",
    "arg_i",
    "arg_c",
    "span",
    "head",
    "soft_trigger",
    "relaxed_span",
    "normalized_arg_i",
    "normalized_arg_c",
    "partial_argument",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def validate_resume_checkpoint(checkpoint_dir: str) -> Tuple[str, int]:
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    required_files = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [
        filename
        for filename in required_files
        if not os.path.isfile(os.path.join(checkpoint_dir, filename))
    ]
    has_weights = any(
        os.path.isfile(os.path.join(checkpoint_dir, filename))
        for filename in ("pytorch_model.bin", "model.safetensors")
    )
    if not has_weights:
        missing.append("pytorch_model.bin/model.safetensors")
    if missing:
        raise FileNotFoundError(
            f"Cannot fully resume from {checkpoint_dir!r}; missing: {', '.join(missing)}"
        )

    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(state_path, encoding="utf-8") as f:
        trainer_state = json.load(f)
    global_step = int(trainer_state.get("global_step", 0) or 0)
    if global_step <= 0:
        raise ValueError(
            f"Checkpoint {checkpoint_dir!r} has invalid global_step={global_step}."
        )
    return checkpoint_dir, global_step


class ResumeVerificationCallback(TrainerCallback):
    def __init__(self, expected_step: int):
        self.expected_step = int(expected_step)

    def on_train_begin(self, args, state, control, **kwargs):
        if int(state.global_step) < self.expected_step:
            raise RuntimeError(
                "Trainer did not restore checkpoint progress: "
                f"expected step >= {self.expected_step}, got {state.global_step}."
            )
        print(
            f"Confirmed full checkpoint resume at global_step={state.global_step}, "
            f"epoch={state.epoch}.",
            flush=True,
        )
        return control


def resolve_lora_target_modules(model_name_or_path: str, override: str) -> List[str]:
    override = (override or "auto").strip()
    if override and override.lower() not in {"auto", "default"}:
        modules = [item.strip() for item in override.split(",") if item.strip()]
        if not modules:
            raise ValueError("--lora_target_modules was provided but no valid module names were found.")
        return modules

    model_name = model_name_or_path.lower()
    if any(name in model_name for name in ["llama", "mistral", "falcon", "qwen"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    if any(name in model_name for name in ["phi"]):
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    if any(name in model_name for name in ["gpt2"]):
        return ["c_attn", "c_proj", "c_fc"]
    raise ValueError(f"Unsupported LoRA target modules for {model_name_or_path}.")

class CustomTrainer(Trainer):
    COMPACT_SAVE_PREFIXES = (
        "prj.",
        "latent_budget_head.",
        "latent_budget_embedding.",
        "event_handle_query.",
        "event_handle_embedding.",
        "event_handle_active_embedding.",
        "event_handle_anchor_proj.",
        "event_handle_query_norm.",
        "event_handle_condition_proj.",
        "event_handle_exist_head.",
        "handle_latent_query_proj.",
        "handle_latent_key_proj.",
        "handle_latent_value_proj.",
        "handle_latent_gate.",
        "decoder_bridge_proj.",
        "decoder_bridge_gate.",
        "decoder_bridge_contribution_gate.",
        "document_summary_gate.",
        "latent_boundary_embedding.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest_aux_logs = {}
        self._aux_log_sums = defaultdict(float)
        self._aux_log_count = 0

    def _compact_state_dict(self, model):
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        state_dict = model.state_dict()
        compact = {}
        for name, tensor in state_dict.items():
            is_trainable = name in trainable_names
            is_lora = "lora_" in name or "modules_to_save" in name
            is_custom = any(name.startswith(prefix) for prefix in self.COMPACT_SAVE_PREFIXES)
            if is_trainable or is_lora or is_custom:
                compact[name] = tensor.detach().cpu()
        return compact

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Ignore Trainer's full state_dict argument; for large backbones it would
        # write the frozen base model and quickly exhaust disk space.
        state_dict = self._compact_state_dict(self.model)
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        temporary_save_path = save_path + ".tmp"
        torch.save(state_dict, temporary_save_path)
        os.replace(temporary_save_path, save_path)
        args_path = os.path.join(output_dir, "training_args.bin")
        temporary_args_path = args_path + ".tmp"
        torch.save(self.args, temporary_args_path)
        os.replace(temporary_args_path, args_path)
        processor = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(output_dir)
        metadata_path = os.path.join(output_dir, "compact_checkpoint.json")
        temporary_metadata_path = metadata_path + ".tmp"
        with open(temporary_metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": "compact_trainable_and_apler_modules",
                    "num_tensors": len(state_dict),
                    "note": "Base model weights are loaded from model_name_or_path; this checkpoint stores LoRA and APLER task heads.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(temporary_metadata_path, metadata_path)

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary_path, path)

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _selection_thresholds(self) -> List[float]:
        if not bool(getattr(self.args, "selection_checkpoint_enable", False)):
            return []
        start = float(self.args.selection_checkpoint_start_epoch)
        interval = float(self.args.selection_checkpoint_interval_epochs)
        final_epoch = float(self.args.num_train_epochs)
        thresholds = []
        target = start
        while target < final_epoch - 1e-8:
            thresholds.append(round(target, 8))
            target += interval
        return thresholds

    def _selection_root(self) -> str:
        return os.path.join(self.args.output_dir, "selection_checkpoints")

    def _selection_manifest_path(self) -> str:
        return os.path.join(self._selection_root(), "candidates.json")

    def _selection_config_summary(self) -> Dict:
        model_args = getattr(self.model, "model_args", None)
        return {
            "experiment_name": str(getattr(self.args, "expt_name", "")),
            "model_name_or_path": str(
                getattr(model_args, "model_name_or_path", "")
            ),
            "num_train_epochs": float(self.args.num_train_epochs),
            "max_steps": int(getattr(self.args, "max_steps", -1)),
            "learning_rate": float(self.args.learning_rate),
            "seed": int(self.args.seed),
            "per_device_train_batch_size": int(
                self.args.per_device_train_batch_size
            ),
            "gradient_accumulation_steps": int(
                self.args.gradient_accumulation_steps
            ),
            "selection_checkpoint_start_epoch": float(
                self.args.selection_checkpoint_start_epoch
            ),
            "selection_checkpoint_interval_epochs": float(
                self.args.selection_checkpoint_interval_epochs
            ),
        }

    def _load_selection_manifest(self) -> Dict:
        path = self._selection_manifest_path()
        if not os.path.isfile(path):
            return {
                "format_version": 1,
                "selection_checkpoint_start_epoch": float(
                    self.args.selection_checkpoint_start_epoch
                ),
                "selection_checkpoint_interval_epochs": float(
                    self.args.selection_checkpoint_interval_epochs
                ),
                "final_epoch": float(self.args.num_train_epochs),
                "candidates": [],
            }
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if int(manifest.get("format_version", 0)) != 1:
            raise RuntimeError(f"Unsupported selection manifest: {path}")
        return manifest

    def _write_selection_manifest_entry(self, entry: Dict):
        manifest = self._load_selection_manifest()
        candidates = [
            candidate
            for candidate in manifest.get("candidates", [])
            if abs(
                float(candidate.get("target_epoch", -1.0))
                - float(entry["target_epoch"])
            )
            > 1e-8
        ]
        candidates.append(entry)
        candidates.sort(key=lambda candidate: float(candidate["target_epoch"]))
        manifest["candidates"] = candidates
        self._atomic_write_json(self._selection_manifest_path(), manifest)

    def _validate_selection_snapshot(
        self, snapshot_dir: str, target_epoch: float
    ) -> Dict:
        metadata_path = os.path.join(snapshot_dir, "selection_snapshot.json")
        weight_path = os.path.join(snapshot_dir, "pytorch_model.bin")
        if not os.path.isfile(metadata_path) or not os.path.isfile(weight_path):
            raise RuntimeError(
                f"Incomplete selection snapshot at {snapshot_dir!r}."
            )
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if abs(float(metadata.get("target_epoch", -1.0)) - target_epoch) > 1e-8:
            raise RuntimeError(
                f"Selection snapshot epoch mismatch at {snapshot_dir!r}."
            )
        if metadata.get("training_config") != self._selection_config_summary():
            raise RuntimeError(
                f"Selection snapshot training configuration mismatch at {snapshot_dir!r}."
            )
        actual_sha256 = self._file_sha256(weight_path)
        if metadata.get("weights_sha256") != actual_sha256:
            raise RuntimeError(
                f"Selection snapshot checksum mismatch at {snapshot_dir!r}."
            )
        return metadata

    def _save_selection_snapshot(self, target_epoch: float):
        label = f"{target_epoch:.8f}".rstrip("0").rstrip(".")
        if "." not in label:
            label += ".0"
        snapshot_dir = os.path.join(self._selection_root(), f"epoch_{label}")
        if os.path.exists(snapshot_dir):
            metadata = self._validate_selection_snapshot(snapshot_dir, target_epoch)
            self._write_selection_manifest_entry(
                {
                    "target_epoch": target_epoch,
                    "observed_epoch": float(metadata["observed_epoch"]),
                    "global_step": int(metadata["global_step"]),
                    "checkpoint_dir": os.path.relpath(
                        snapshot_dir, self.args.output_dir
                    ).replace(os.sep, "/"),
                    "weights_sha256": metadata["weights_sha256"],
                    "kind": "model_only_snapshot",
                }
            )
            return

        os.makedirs(self._selection_root(), exist_ok=True)
        temporary_dir = (
            snapshot_dir
            + f".tmp-{os.getpid()}-{int(self.state.global_step)}"
        )
        if os.path.exists(temporary_dir):
            shutil.rmtree(temporary_dir)
        os.makedirs(temporary_dir)
        try:
            state_dict = self._compact_state_dict(self.model)
            weight_path = os.path.join(temporary_dir, "pytorch_model.bin")
            torch.save(state_dict, weight_path)
            weights_sha256 = self._file_sha256(weight_path)
            self._atomic_write_json(
                os.path.join(temporary_dir, "compact_checkpoint.json"),
                {
                    "format": "compact_trainable_and_apler_modules",
                    "num_tensors": len(state_dict),
                    "note": "Model-only selection snapshot; optimizer and Trainer state are intentionally omitted.",
                },
            )
            metadata = {
                "format_version": 1,
                "kind": "model_only_selection_snapshot",
                "target_epoch": target_epoch,
                "observed_epoch": float(self.state.epoch),
                "global_step": int(self.state.global_step),
                "weights_sha256": weights_sha256,
                "training_config": self._selection_config_summary(),
            }
            self._atomic_write_json(
                os.path.join(temporary_dir, "selection_snapshot.json"),
                metadata,
            )
            os.replace(temporary_dir, snapshot_dir)
        except Exception:
            if os.path.isdir(temporary_dir):
                shutil.rmtree(temporary_dir)
            raise

        self._write_selection_manifest_entry(
            {
                "target_epoch": target_epoch,
                "observed_epoch": float(self.state.epoch),
                "global_step": int(self.state.global_step),
                "checkpoint_dir": os.path.relpath(
                    snapshot_dir, self.args.output_dir
                ).replace(os.sep, "/"),
                "weights_sha256": weights_sha256,
                "kind": "model_only_snapshot",
            }
        )
        print(
            "Saved selection snapshot | "
            f"target_epoch={target_epoch:g} observed_epoch={float(self.state.epoch):.6f} "
            f"step={int(self.state.global_step)} path={snapshot_dir}",
            flush=True,
        )

    def initialize_selection_snapshot_tracking(self, current_epoch: float):
        self._selection_last_epoch = float(current_epoch or 0.0)
        if not bool(getattr(self.args, "selection_checkpoint_enable", False)):
            return
        thresholds = self._selection_thresholds()
        existing_root = self._selection_root()
        if self._selection_last_epoch <= 1e-8 and os.path.isdir(existing_root):
            entries = os.listdir(existing_root)
            if entries:
                raise RuntimeError(
                    "Selection snapshots already exist for a fresh training run. "
                    "Use a new EXPT_NAME or remove the stale experiment directory."
                )
        for target_epoch in thresholds:
            if target_epoch > self._selection_last_epoch + 1e-8:
                continue
            label = f"{target_epoch:.8f}".rstrip("0").rstrip(".")
            if "." not in label:
                label += ".0"
            snapshot_dir = os.path.join(existing_root, f"epoch_{label}")
            self._validate_selection_snapshot(snapshot_dir, target_epoch)

    def maybe_save_selection_snapshots(self, current_epoch: float):
        if not bool(getattr(self.args, "selection_checkpoint_enable", False)):
            return
        if not self.is_world_process_zero():
            return
        current_epoch = float(current_epoch or 0.0)
        previous_epoch = float(
            getattr(self, "_selection_last_epoch", current_epoch)
        )
        for target_epoch in self._selection_thresholds():
            if previous_epoch < target_epoch <= current_epoch + 1e-8:
                self._save_selection_snapshot(target_epoch)
        self._selection_last_epoch = current_epoch

    def register_final_selection_candidate(self):
        if not bool(getattr(self.args, "selection_checkpoint_enable", False)):
            return
        if not self.is_world_process_zero():
            return
        final_epoch = float(self.args.num_train_epochs)
        observed_epoch = float(self.state.epoch or 0.0)
        if observed_epoch < final_epoch - 1e-6:
            print(
                "Skipping final selection candidate because training ended before "
                f"epoch {final_epoch:g} (observed_epoch={observed_epoch:.6f}).",
                flush=True,
            )
            return
        weight_path = os.path.join(self.args.output_dir, "pytorch_model.bin")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Final compact model is missing: {weight_path!r}"
            )
        weights_sha256 = self._file_sha256(weight_path)
        self._write_selection_manifest_entry(
            {
                "target_epoch": final_epoch,
                "observed_epoch": observed_epoch,
                "global_step": int(self.state.global_step),
                "checkpoint_dir": ".",
                "weights_sha256": weights_sha256,
                "kind": "final_model",
            }
        )
        self._atomic_write_json(
            os.path.join(self.args.output_dir, "selection_final.json"),
            {
                "format_version": 1,
                "target_epoch": final_epoch,
                "observed_epoch": observed_epoch,
                "global_step": int(self.state.global_step),
                "weights_sha256": weights_sha256,
                "training_config": self._selection_config_summary(),
            },
        )
        print(
            "Registered final selection candidate | "
            f"epoch={final_epoch:g} step={int(self.state.global_step)} "
            f"path={self.args.output_dir}",
            flush=True,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps.  HuggingFace uses max_steps as the actual
        # schedule length when it is set, so the model-side teacher-forcing
        # annealing should follow the same clock.
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        if self.args.max_steps and self.args.max_steps > 0:
            total_steps = self.args.max_steps
        else:
            total_steps = ceil(dataset_size / effective_batch_size) * num_epochs
        total_steps = max(float(total_steps), 1.0)

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        self._latest_aux_logs = {
            "ce_loss": outputs["ce_loss"],
            "native_boundary_loss": outputs.get("native_boundary_loss", 0.0),
            "native_boundary_bce": outputs.get("native_boundary_bce", 0.0),
            "process_distill_loss": outputs.get("process_distill_loss", 0.0),
            "reason_loss": outputs.get("reason_loss", 0.0),
            "instance_loss": outputs.get("instance_loss", 0.0),
            "auxiliary_loss_gate": outputs.get("auxiliary_loss_gate", 1.0),
            "complexity_loss": outputs.get("complexity_loss", 0.0),
            "event_handle_anchor_loss": outputs.get("event_handle_anchor_loss", 0.0),
            "event_handle_exist_loss": outputs.get("event_handle_exist_loss", 0.0),
            "event_handle_exist_bce_loss": outputs.get(
                "event_handle_exist_bce_loss", 0.0
            ),
            "event_handle_count_nll_loss": outputs.get(
                "event_handle_count_nll_loss", 0.0
            ),
            "event_handle_gold_count_probability": outputs.get(
                "event_handle_gold_count_probability", 0.0
            ),
            "event_handle_count_entropy": outputs.get(
                "event_handle_count_entropy", 0.0
            ),
            "event_handle_count_reliability": outputs.get(
                "event_handle_count_reliability", 0.0
            ),
            "event_handle_expected_count_mae": outputs.get(
                "event_handle_expected_count_mae", 0.0
            ),
            "ce_token_weight_alpha": outputs.get("ce_token_weight_alpha", 1.0),
            "predicted_complexity": outputs.get("predicted_complexity", 0.0),
            "latent_budget_acc": outputs.get("latent_budget_acc", 0.0),
            "latent_budget_top1_mean": outputs.get(
                "latent_budget_top1_mean", 0.0
            ),
            "latent_budget_expected_mean": outputs.get(
                "latent_budget_expected_mean", 0.0
            ),
            "latent_budget_expected_mae": outputs.get(
                "latent_budget_expected_mae", 0.0
            ),
            "latent_budget_proposal_acc": outputs.get(
                "latent_budget_proposal_acc", 0.0
            ),
            "latent_budget_proposal_mae": outputs.get(
                "latent_budget_proposal_mae", 0.0
            ),
            "latent_budget_entropy": outputs.get("latent_budget_entropy", 0.0),
            "latent_budget_top1_margin": outputs.get(
                "latent_budget_top1_margin", 0.0
            ),
            "latent_budget_ce_loss": outputs.get("latent_budget_ce_loss", 0.0),
            "latent_budget_ordinal_loss": outputs.get(
                "latent_budget_ordinal_loss", 0.0
            ),
            "decoder_exit_mix_rate": outputs.get("decoder_exit_mix_rate", 0.0),
            "decoder_exit_gap": outputs.get("decoder_exit_gap", 0.0),
            "decoder_exit_used": outputs.get("decoder_exit_used", 0.0),
            "event_handle_anchor_acc": outputs.get("event_handle_anchor_acc", 0.0),
            "event_handle_exist_acc": outputs.get("event_handle_exist_acc", 0.0),
            "event_handle_exist_pos_probability": outputs.get(
                "event_handle_exist_pos_probability", 0.0
            ),
            "event_handle_exist_neg_probability": outputs.get(
                "event_handle_exist_neg_probability", 0.0
            ),
            "event_handle_conf_ge30": outputs.get("event_handle_conf_ge30", 0.0),
            "event_handle_conf_ge50": outputs.get("event_handle_conf_ge50", 0.0),
            "event_handle_conf_ge75": outputs.get(
                "event_handle_conf_ge75",
                outputs.get("event_handle_control_recall", 0.0),
            ),
            "event_handle_count_acc": outputs.get("event_handle_count_acc", 0.0),
            "event_handle_query_seed_guard_rate": outputs.get(
                "event_handle_query_seed_guard_rate", 0.0
            ),
            "decoder_bridge_out_gate": outputs.get("decoder_bridge_out_gate", 1.0),
            "decoder_bridge_delta": outputs.get("decoder_bridge_delta", 0.0),
            "json_prefix_robust_rate": outputs.get("json_prefix_robust_rate", 0.0),
            "json_prefix_value_robust_rate": outputs.get("json_prefix_value_robust_rate", 0.0),
            "json_prefix_key_robust_rate": outputs.get("json_prefix_key_robust_rate", 0.0),
            "json_prefix_event_start_robust_rate": outputs.get("json_prefix_event_start_robust_rate", 0.0),
            "json_start_continuation_rate": outputs.get("json_start_continuation_rate", 0.0),
            "boundary_active_rate": outputs.get("boundary_active_rate", 0.0),
            "boundary_prior_confidence": outputs.get("boundary_prior_confidence", 0.0),
            "boundary_gold_continue_rate": outputs.get("boundary_gold_continue_rate", 0.0),
            "boundary_gold_stop_rate": outputs.get("boundary_gold_stop_rate", 0.0),
            "boundary_prior_agreement": outputs.get("boundary_prior_agreement", 0.0),
            "boundary_qwen_continue": outputs.get("boundary_qwen_continue", 0.0),
            "boundary_blended_continue": outputs.get("boundary_blended_continue", 0.0),
            "native_boundary_coverage": outputs.get(
                "native_boundary_coverage", 0.0
            ),
            "native_boundary_commit_rate": outputs.get(
                "native_boundary_commit_rate", 0.0
            ),
            "native_boundary_choice_acc": outputs.get(
                "native_boundary_choice_acc", 0.0
            ),
            "native_continue_gold_continue": outputs.get(
                "native_continue_gold_continue", 0.0
            ),
            "native_continue_gold_stop": outputs.get(
                "native_continue_gold_stop", 0.0
            ),
            "native_boundary_proposal_failure_rate": outputs.get(
                "native_boundary_proposal_failure_rate", 0.0
            ),
            "native_boundary_low_count_acc": outputs.get(
                "native_boundary_low_count_acc", 0.0
            ),
            "native_boundary_high_count_acc": outputs.get(
                "native_boundary_high_count_acc", 0.0
            ),
            "native_boundary_first_acc": outputs.get(
                "native_boundary_first_acc", 0.0
            ),
            "native_boundary_later_acc": outputs.get(
                "native_boundary_later_acc", 0.0
            ),
            "sample_loss_weight": outputs.get("sample_loss_weight", 1.0),
        }
        metric_model = getattr(model, "module", model)
        for budget_step in range(
            3,
            int(getattr(metric_model, "num_latent", 3)) + 1,
        ):
            for histogram_name in ("target", "proposal"):
                key = f"latent_budget_{histogram_name}_t{budget_step}_rate"
                self._latest_aux_logs[key] = outputs.get(key, 0.0)
        for metric_name in TRAIN_PROXY_METRIC_BASES:
            for suffix in ("p", "r", "f1"):
                key = f"train_proxy_{metric_name}_{suffix}"
                self._latest_aux_logs[key] = outputs.get(key, 0.0)
        for key, value in self._latest_aux_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().item()
            self._aux_log_sums[key] += float(value)
        self._aux_log_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.state.global_step is None:
            return

        merged_logs = dict(logs)
        if any(key in logs for key in ("loss", "learning_rate", "grad_norm")):
            if self._aux_log_count:
                merged_logs.update(
                    {
                        key: value / self._aux_log_count
                        for key, value in self._aux_log_sums.items()
                    }
                )
                self._aux_log_sums = defaultdict(float)
                self._aux_log_count = 0
            else:
                merged_logs.update(self._latest_aux_logs)
        if self.state.epoch is not None:
            merged_logs.setdefault("epoch", round(self.state.epoch, 4))

        display_logs = dict(merged_logs)
        display_logs["step"] = self.state.global_step
        self.state.log_history.append(display_logs)
        print(self._format_log_line(display_logs), flush=True)

    @staticmethod
    def _format_number(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().item()
        if isinstance(value, float):
            if value != value:
                return "nan"
            if abs(value) >= 100:
                return f"{value:.2f}"
            if abs(value) >= 1:
                return f"{value:.4f}"
            return f"{value:.6f}"
        return str(value)

    def _format_log_line(self, logs):
        step = logs.get("step", self.state.global_step)
        epoch = logs.get("epoch", "")
        loss_component_keys = [
            "loss",
            "ce_loss",
            "native_boundary_loss",
            "native_boundary_bce",
            "process_distill_loss",
            "event_handle_anchor_loss",
            "event_handle_exist_loss",
            "event_handle_exist_bce_loss",
            "event_handle_count_nll_loss",
            "complexity_loss",
            "latent_budget_ce_loss",
            "latent_budget_ordinal_loss",
            "reason_loss",
            "instance_loss",
            "auxiliary_loss_gate",
            "ce_token_weight_alpha",
        ]
        proxy_metric_keys = {
            f"train_proxy_{metric_name}_{suffix}"
            for metric_name in TRAIN_PROXY_METRIC_BASES
            for suffix in ("p", "r", "f1")
        }
        latent_keys = [
            ("complex", "complexity_loss"),
            ("pred_budget", "predicted_complexity"),
            ("top1", "latent_budget_top1_mean"),
            ("expected", "latent_budget_expected_mean"),
            ("top1_acc", "latent_budget_acc"),
            ("proposal_acc", "latent_budget_proposal_acc"),
            ("expected_mae", "latent_budget_expected_mae"),
            ("proposal_mae", "latent_budget_proposal_mae"),
            ("entropy", "latent_budget_entropy"),
            ("margin", "latent_budget_top1_margin"),
            ("exit_mix", "decoder_exit_mix_rate"),
            ("exit_gap", "decoder_exit_gap"),
            ("exit_used", "decoder_exit_used"),
        ]
        opt_keys = [
            ("lr", "learning_rate"),
            ("grad", "grad_norm"),
        ]
        structure_keys = [
            ("handle_anchor_acc", "event_handle_anchor_acc"),
            ("handle_exist_acc", "event_handle_exist_acc"),
            ("handle_exist_pos", "event_handle_exist_pos_probability"),
            ("handle_exist_neg", "event_handle_exist_neg_probability"),
            ("handle_pos_ge30", "event_handle_conf_ge30"),
            ("handle_pos_ge50", "event_handle_conf_ge50"),
            ("handle_pos_ge75", "event_handle_conf_ge75"),
            ("handle_count_acc", "event_handle_count_acc"),
            ("handle_gold_prob", "event_handle_gold_count_probability"),
            ("handle_count_entropy", "event_handle_count_entropy"),
            ("handle_count_reliability", "event_handle_count_reliability"),
            ("handle_expected_mae", "event_handle_expected_count_mae"),
            ("handle_seed_guard", "event_handle_query_seed_guard_rate"),
            ("bridge_out_gate", "decoder_bridge_out_gate"),
            ("bridge_delta", "decoder_bridge_delta"),
            ("prefix_robust", "json_prefix_robust_rate"),
            ("prefix_value", "json_prefix_value_robust_rate"),
            ("prefix_key", "json_prefix_key_robust_rate"),
            ("prefix_event", "json_prefix_event_start_robust_rate"),
            ("start_cont", "json_start_continuation_rate"),
            ("sample_w", "sample_loss_weight"),
        ]
        boundary_keys = [
            ("active", "boundary_active_rate"),
            ("confidence", "boundary_prior_confidence"),
            ("gold_continue", "boundary_gold_continue_rate"),
            ("gold_stop", "boundary_gold_stop_rate"),
            ("agreement", "boundary_prior_agreement"),
            ("qwen_continue", "boundary_qwen_continue"),
            ("blended_continue", "boundary_blended_continue"),
        ]
        native_boundary_keys = [
            ("coverage", "native_boundary_coverage"),
            ("commit", "native_boundary_commit_rate"),
            ("choice_acc", "native_boundary_choice_acc"),
            ("p_on_continue", "native_continue_gold_continue"),
            ("p_on_stop", "native_continue_gold_stop"),
            ("proposal_fail", "native_boundary_proposal_failure_rate"),
            ("low_count_acc", "native_boundary_low_count_acc"),
            ("high_count_acc", "native_boundary_high_count_acc"),
            ("first_acc", "native_boundary_first_acc"),
            ("later_acc", "native_boundary_later_acc"),
        ]

        def value(key):
            raw_value = logs.get(key, 0.0)
            if isinstance(raw_value, torch.Tensor):
                raw_value = raw_value.detach().float().item()
            try:
                return float(raw_value)
            except Exception:
                return 0.0

        def group(items):
            return " ".join(
                f"{name}={self._format_number(logs[key])}"
                for name, key in items
                if key in logs
            )

        def loss_detail_group(items):
            detail = group(items)
            return detail if detail else ""

        def grouped_loss_line():
            if "loss" not in logs:
                return ""
            return (
                f"total={self._format_number(value('loss'))} "
                f"ce={self._format_number(value('ce_loss'))} "
                f"reason={self._format_number(value('reason_loss'))} "
                f"instance={self._format_number(value('instance_loss'))}"
            )

        def grouped_balance_line():
            if "auxiliary_loss_gate" not in logs:
                return ""
            return (
                f"aux_gate={self._format_number(value('auxiliary_loss_gate'))} "
                f"ce_warm={self._format_number(value('ce_token_weight_alpha'))}"
            )

        def grouped_loss_details():
            if "loss" not in logs:
                return []
            detail_specs = [
                (
                    "loss.process",
                    [
                        ("latent", "process_distill_loss"),
                    ],
                ),
                (
                    "loss.instance",
                    [
                        ("handle_anchor", "event_handle_anchor_loss"),
                        ("handle_exist", "event_handle_exist_loss"),
                        ("handle_bce", "event_handle_exist_bce_loss"),
                        ("handle_count_nll", "event_handle_count_nll_loss"),
                    ],
                ),
                (
                    "loss.boundary",
                    [
                        ("native", "native_boundary_loss"),
                        ("raw_bce", "native_boundary_bce"),
                    ],
                ),
                (
                    "loss.adaptive",
                    [
                        ("complexity", "complexity_loss"),
                        ("budget_ce", "latent_budget_ce_loss"),
                        ("budget_ordinal", "latent_budget_ordinal_loss"),
                    ],
                ),
            ]
            parts = []
            for label, items in detail_specs:
                detail = loss_detail_group(items)
                if detail:
                    parts.append(f"{label} | {detail}")
            return parts

        def proxy_triplet(metric_name):
            p_key = f"train_proxy_{metric_name}_p"
            r_key = f"train_proxy_{metric_name}_r"
            f1_key = f"train_proxy_{metric_name}_f1"
            if p_key not in logs and r_key not in logs and f1_key not in logs:
                return ""
            return (
                f"p={self._format_number(value(p_key))} "
                f"r={self._format_number(value(r_key))} "
                f"f1={self._format_number(value(f1_key))}"
            )

        def proxy_section(label, specs):
            entries = []
            for display_name, metric_name in specs:
                triplet = proxy_triplet(metric_name)
                if triplet:
                    entries.append(f"{display_name} {triplet}")
            return f"{label} | " + " | ".join(entries) if entries else ""

        parts = [f"*** step={step} epoch={epoch}"]
        loss_part = grouped_loss_line()
        loss_detail_parts = grouped_loss_details()
        balance_part = grouped_balance_line()
        latent_part = group(latent_keys)
        latent_histogram_items = []
        for budget_step in range(3, 33):
            target_key = f"latent_budget_target_t{budget_step}_rate"
            proposal_key = f"latent_budget_proposal_t{budget_step}_rate"
            if target_key in logs or proposal_key in logs:
                latent_histogram_items.append(
                    f"T{budget_step}:target={self._format_number(value(target_key))}"
                    f"/proposal={self._format_number(value(proposal_key))}"
                )
        latent_histogram_part = " ".join(latent_histogram_items)
        opt_part = group(opt_keys)
        structure_part = group(structure_keys)
        boundary_part = group(boundary_keys)
        native_boundary_part = group(native_boundary_keys)
        paper_main_part = proxy_section(
            "paper.main(train-proxy)",
            (
                ("Trigger", "trigger"),
                ("Arg-I", "arg_i"),
                ("Arg-C", "arg_c"),
            ),
        )
        paper_appendix_part = proxy_section(
            "paper.appendix(train-proxy)",
            (
                ("Span", "span"),
                ("Head", "head"),
            ),
        )
        paper_relaxed_part = proxy_section(
            "paper.relaxed(train-proxy)",
            (
                ("Soft Trigger", "soft_trigger"),
                ("Relaxed Span", "relaxed_span"),
                ("Normalized Arg-I", "normalized_arg_i"),
                ("Normalized Arg-C", "normalized_arg_c"),
                ("Partial Arg-C", "partial_argument"),
            ),
        )
        if loss_part:
            parts.append(f"loss | {loss_part}")
        if balance_part:
            parts.append(f"loss.balance | {balance_part}")
        parts.extend(loss_detail_parts)
        if paper_main_part:
            parts.append(paper_main_part)
        if paper_appendix_part:
            parts.append(paper_appendix_part)
        if paper_relaxed_part:
            parts.append(paper_relaxed_part)
        if latent_part:
            parts.append(f"latent | {latent_part}")
        if latent_histogram_part:
            parts.append(f"latent.hist | {latent_histogram_part}")
        if structure_part:
            parts.append(f"structure | {structure_part}")
        if boundary_part:
            parts.append(f"boundary | {boundary_part}")
        if native_boundary_part:
            parts.append(f"boundary.native | {native_boundary_part}")
        if opt_part:
            parts.append(f"opt | {opt_part}")

        leftovers = {
            key: value
            for key, value in logs.items()
            if key not in {"step", "epoch"}
            and key not in set(loss_component_keys)
            and key not in proxy_metric_keys
            and key not in {
                item[1]
                for item in (
                    latent_keys
                    + opt_keys
                    + structure_keys
                    + boundary_keys
                    + native_boundary_keys
                )
            }
            and not key.startswith("latent_budget_target_t")
            and not key.startswith("latent_budget_proposal_t")
        }
        if leftovers:
            parts.append(
                "extra | "
                + " ".join(f"{key}={self._format_number(value)}" for key, value in sorted(leftovers.items()))
            )
        if len(parts) == 1:
            return parts[0] + " ***"
        header, *body = parts
        return header + "\n" + "\n".join(f"--- {part}" for part in body) + "\n***"


class SelectionCheckpointCallback(TrainerCallback):
    """Save late model-only candidates without changing Trainer checkpoints."""

    def __init__(self, trainer: CustomTrainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.initialize_selection_snapshot_tracking(state.epoch or 0.0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer.maybe_save_selection_snapshots(state.epoch or 0.0)
        return control


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int = 256) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def token_ids_no_warning(
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    add_special_tokens: bool = True,
) -> List[int]:
    """Tokenize for length checks without emitting model_max_length warnings."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        ids = encoded["input_ids"]
    except TypeError:
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def token_count_no_warning(
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    add_special_tokens: bool = True,
) -> int:
    return len(token_ids_no_warning(text, tokenizer, add_special_tokens=add_special_tokens))


def training_path_token_lengths(
    source: str,
    process: str,
    answer: str,
    tokenizer: transformers.PreTrainedTokenizer,
    bot_id: int,
    eot_id: int,
    num_latent: int,
    max_event_count: int,
    remove_eos: bool,
) -> Tuple[int, int, int]:
    """Measure the two actual training paths before any tokenizer truncation."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The tokenizer must define eos_token_id for CODI training.")
    bos_id = tokenizer.bos_token_id

    source_ids = token_ids_no_warning(source, tokenizer)
    if not remove_eos:
        source_ids.append(int(eos_id))
    source_ids.append(int(bot_id))

    process_ids = token_ids_no_warning(process, tokenizer)
    if not remove_eos:
        process_ids.append(int(eos_id))
    if process_ids and bos_id is not None and process_ids[0] == bos_id:
        process_ids = process_ids[1:]

    decoder_text = strip_final_json_marker_text(answer)
    if not decoder_text.startswith(JSON_EVENT_ARRAY_PREFIX):
        raise ValueError(
            "Training decoder JSON does not use the canonical root prefix: "
            f"{decoder_text[:80]!r}."
        )
    decoder_ids = (
        encode_json_event_array_prefix(tokenizer)
        + encode_without_special_tokens(
            decoder_text[len(JSON_EVENT_ARRAY_PREFIX) :],
            tokenizer,
        )
        + [int(eos_id)]
    )
    decoder_prefix = [int(eot_id)] if remove_eos else [int(eot_id), int(eos_id)]
    decoder_ids = decoder_prefix + decoder_ids

    reference_length = len(source_ids) + len(process_ids)
    decoder_length = event_decoder_context_length(
        source_length=len(source_ids),
        answer_length=len(decoder_ids),
        num_latent=num_latent,
        max_event_count=max_event_count,
    )
    return len(source_ids), reference_length, decoder_length


def strip_final_json_marker_text(text: str) -> str:
    if FINAL_JSON_MARKER in text:
        return text.split(FINAL_JSON_MARKER, 1)[1].lstrip()
    return text


def encode_without_special_tokens(text: str, tokenizer: transformers.PreTrainedTokenizer) -> List[int]:
    return token_ids_no_warning(text, tokenizer, add_special_tokens=False)


def find_all_subsequence_starts(tokens: Sequence[int], pattern: Sequence[int]) -> List[int]:
    if not pattern:
        return []
    token_list = list(tokens)
    pattern_list = list(pattern)
    starts = []
    limit = len(token_list) - len(pattern_list) + 1
    for start in range(max(0, limit)):
        if token_list[start : start + len(pattern_list)] == pattern_list:
            starts.append(start)
    return starts


def marker_token_variants(marker: str, tokenizer: transformers.PreTrainedTokenizer) -> List[List[int]]:
    """Return tokenizer-stable marker encodings without relying on special tokens."""
    variants = []
    for text in (marker, f"{marker}\n", f"\n{marker}", f" {marker}"):
        ids = encode_without_special_tokens(text, tokenizer)
        if ids and ids not in variants:
            variants.append(ids)
    return variants


def find_marker_span(tokens: Sequence[int], marker_variants: Sequence[Sequence[int]]) -> Tuple[int, int]:
    spans = []
    for variant in marker_variants:
        for start in find_all_subsequence_starts(tokens, variant):
            spans.append((start, start + len(variant)))
    return min(spans, key=lambda item: item[0]) if spans else (-1, -1)


def _torch_mask_from_char_spans(
    layout: EventDecoderTokenization,
    spans: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    return torch.tensor(
        token_mask_from_char_spans(layout.offset_mapping, spans),
        dtype=torch.bool,
    )


def _exclude_json_root_prefix(
    mask: torch.Tensor,
    layout: EventDecoderTokenization,
) -> torch.Tensor:
    """Keep the force-fed JSON root prefix immutable under input perturbations."""
    protected = mask.clone()
    protected[int(layout.json_token_start) : int(layout.prefix_end)] = False
    return protected


def _event_value_string_spans(layout: EventDecoderTokenization):
    records = []
    for record in layout.spans.string_spans:
        path = record.path
        if (
            len(path) >= 3
            and path[0] == "events"
            and isinstance(path[1], int)
            and (path[2] in {"event_type", "trigger", "evidence"} or path[2] == "arguments")
            and record.end > record.start
        ):
            records.append(record)
    return records


def _event_internal_key_spans(layout: EventDecoderTokenization):
    return [
        record
        for record in layout.spans.key_spans
        if len(record.path) >= 2
        and record.path[0] == "events"
        and isinstance(record.path[1], int)
    ]


def build_json_structure_token_weights(
    layout: EventDecoderTokenization,
    structure_weight: float,
    event_boundary_weight: float,
    value_weight: float = 1.0,
    boundary_label_positions: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Upweight JSON skeleton and event value spans.

    Early APLER checkpoints learned event counts and braces before learning to
    copy enough casualty/location/time/facility values.  Value-token weighting
    makes the decoder pay more attention to role values and evidence ids rather
    than only the JSON frame.
    """
    weights = torch.ones(len(layout.input_ids), dtype=torch.float)
    structure_weight = max(1.0, float(structure_weight))
    value_weight = max(1.0, float(value_weight))
    punctuation_weight = 1.0 + 0.5 * (structure_weight - 1.0)
    event_boundary_weight = max(structure_weight, float(event_boundary_weight))
    structure_spans = [layout.spans.root_prefix_span] + [
        (record.start, record.end)
        for record in _event_internal_key_spans(layout)
    ]
    structure_mask = _torch_mask_from_char_spans(layout, structure_spans)
    punctuation_mask = _torch_mask_from_char_spans(layout, layout.spans.punctuation_spans)
    value_mask = _torch_mask_from_char_spans(
        layout,
        [(record.start, record.end) for record in _event_value_string_spans(layout)],
    )
    weights[structure_mask] = torch.maximum(
        weights[structure_mask],
        torch.full_like(weights[structure_mask], structure_weight),
    )
    weights[punctuation_mask] = torch.maximum(
        weights[punctuation_mask],
        torch.full_like(weights[punctuation_mask], punctuation_weight),
    )
    if boundary_label_positions is None:
        legacy_boundary_mask = _torch_mask_from_char_spans(
            layout,
            layout.spans.event_header_spans,
        )
        weights[legacy_boundary_mask] = torch.maximum(
            weights[legacy_boundary_mask],
            torch.full_like(
                weights[legacy_boundary_mask],
                event_boundary_weight,
            ),
        )
    else:
        for label_position in boundary_label_positions:
            label_position = int(label_position)
            if not 0 <= label_position < len(weights):
                raise ValueError(
                    f"Boundary label position {label_position} is outside "
                    f"decoder length {len(weights)}."
                )
            weights[label_position] = max(
                float(weights[label_position].item()),
                event_boundary_weight,
            )
    weights[value_mask] = torch.maximum(
        weights[value_mask],
        torch.full_like(weights[value_mask], value_weight),
    )
    return weights


def build_event_boundary_supervision(
    layout: EventDecoderTokenization,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size: int,
) -> Dict[str, List]:
    """Locate every gold continue/stop decision on the real decoder token path."""

    event_count = len(layout.spans.event_object_spans)
    if event_count <= 0:
        raise ValueError("Boundary supervision requires at least one event object.")

    global_groups = event_boundary_token_groups(tokenizer, vocab_size)
    positions: List[int] = []
    targets: List[int] = []
    completed_counts: List[int] = []
    continue_ids: List[List[int]] = []
    stop_ids: List[List[int]] = []

    token_index = int(layout.json_token_start)
    while token_index < int(layout.json_token_end) and len(positions) < event_count:
        start, end = layout.offset_mapping[token_index]
        start, end = int(start), int(end)
        if start < 0 or end <= start:
            token_index += 1
            continue
        prefix = layout.text[:start]
        piece = layout.text[start:end]
        before = inspect_event_decode_state(prefix)
        transition = classify_event_token_transition(prefix, piece)
        if (
            transition is None
            or int(before.completed_event_count) != len(positions)
        ):
            token_index += 1
            continue

        completed_count = len(positions) + 1
        expected_target = int(completed_count < event_count)
        boundary_position = token_index
        if transition in {"continue", "stop"}:
            actual_target = int(transition == "continue")
            groups = direct_event_boundary_token_groups(
                tokenizer,
                vocab_size,
                prefix,
                state=before,
            )
        elif transition == "complete":
            next_index = token_index + 1
            while next_index < int(layout.json_token_end):
                next_start, next_end = layout.offset_mapping[next_index]
                next_start, next_end = int(next_start), int(next_end)
                if next_start >= 0 and next_end > next_start:
                    next_piece = layout.text[next_start:next_end]
                    if next_piece.strip():
                        break
                next_index += 1
            if next_index >= int(layout.json_token_end):
                raise ValueError(
                    f"Event {completed_count} completes without a following "
                    "continue/stop token."
                )
            boundary_position = next_index
            token_id = int(layout.input_ids[boundary_position])
            if token_id in global_groups["continue"]:
                actual_target = 1
            elif token_id in global_groups["stop"]:
                actual_target = 0
            else:
                raise ValueError(
                    f"Event {completed_count} completed at token {token_index}, "
                    f"but next token id={token_id} is outside continue/stop groups."
                )
            completed_prefix = layout.text[: int(layout.offset_mapping[next_index][0])]
            if not at_completed_event_boundary(completed_prefix):
                raise ValueError(
                    f"Event {completed_count} next-token decision is not at a "
                    "completed event boundary."
                )
            groups = global_groups
        else:
            token_index += 1
            continue

        if actual_target != expected_target:
            raise ValueError(
                f"Event {completed_count}/{event_count} has boundary target "
                f"{actual_target}, expected {expected_target}."
            )
        if not groups["continue"] or not groups["stop"]:
            raise ValueError(
                f"Event {completed_count} boundary lacks a complete counterfactual "
                f"branch set: continue={len(groups['continue'])} "
                f"stop={len(groups['stop'])}."
            )
        gold_token_id = int(layout.input_ids[boundary_position])
        gold_group = groups["continue"] if actual_target else groups["stop"]
        if gold_token_id not in gold_group:
            raise ValueError(
                f"Boundary token id={gold_token_id} at position={boundary_position} "
                f"is absent from its gold {'continue' if actual_target else 'stop'} group."
            )
        positions.append(int(boundary_position))
        targets.append(int(actual_target))
        completed_counts.append(int(completed_count))
        continue_ids.append([int(value) for value in groups["continue"]])
        stop_ids.append([int(value) for value in groups["stop"]])
        token_index = max(token_index + 1, boundary_position + 1)

    if len(positions) != event_count:
        raise ValueError(
            f"Located {len(positions)} event boundaries for {event_count} events."
        )
    if sum(targets) != event_count - 1 or targets[-1] != 0:
        raise ValueError(
            f"Invalid event boundary targets for {event_count} events: {targets}."
        )
    return {
        "label_positions": positions,
        "targets": targets,
        "completed_counts": completed_counts,
        "continue_ids": continue_ids,
        "stop_ids": stop_ids,
    }


def build_json_prefix_robust_mask(
    layout: EventDecoderTokenization,
) -> torch.Tensor:
    """Mark JSON value tokens whose teacher-forced prefix may be perturbed."""
    return _exclude_json_root_prefix(
        _torch_mask_from_char_spans(
            layout,
            [(record.start, record.end) for record in _event_value_string_spans(layout)],
        ),
        layout,
    )


def build_json_prefix_key_robust_mask(
    layout: EventDecoderTokenization,
) -> torch.Tensor:
    """Mark event-internal JSON field-key context tokens for denoising.

    The root ``events`` key is force-fed during inference, so keep it stable in
    training and concentrate key perturbation on event objects and role fields.
    """
    return _exclude_json_root_prefix(
        _torch_mask_from_char_spans(
            layout,
            [(record.start, record.end) for record in _event_internal_key_spans(layout)],
        ),
        layout,
    )


def build_json_prefix_event_start_robust_mask(
    layout: EventDecoderTokenization,
) -> torch.Tensor:
    """Mark event headers and later inter-event contexts, never the root prefix."""
    spans = list(layout.spans.event_header_spans)
    spans.extend(layout.spans.event_context_spans[1:])
    return _exclude_json_root_prefix(
        _torch_mask_from_char_spans(layout, spans),
        layout,
    )


def build_json_start_continuation_mask(
    layout: EventDecoderTokenization,
    max_tokens: int,
) -> torch.Tensor:
    """Mark the event-array start target and its continuation region.

    The mask is used for bridge-warmed continuation training. The force-fed
    legal root prefix remains visible so the first body token is predicted
    under the same context as inference; only the following body inputs are
    replaceable while their original next-token labels remain unchanged.
    """
    mask = torch.zeros(len(layout.input_ids), dtype=torch.bool)
    max_tokens = int(max(0, max_tokens))
    if max_tokens <= 0:
        return mask
    prefix_end = int(layout.prefix_end)
    if prefix_end <= int(layout.json_token_start) or prefix_end > int(layout.json_token_end):
        raise ValueError(
            "Invalid stable JSON prefix boundary: "
            f"start={layout.json_token_start} end={prefix_end} json_end={layout.json_token_end}."
        )
    start = int(prefix_end)
    end = min(int(layout.json_token_end), start + max_tokens)
    if end > start:
        mask[start:end] = True
    return _exclude_json_root_prefix(mask, layout)


def check_decoder_invariants(
    decoder_layouts: Sequence[EventDecoderTokenization],
    tokenizer: transformers.PreTrainedTokenizer,
    value_masks: Sequence[torch.Tensor],
    key_masks: Sequence[torch.Tensor],
    event_start_masks: Sequence[torch.Tensor],
    start_continuation_masks: Sequence[torch.Tensor],
) -> Dict[str, int]:
    """Check the three decoder invariants that can invalidate training or generation."""
    sample_count = len(decoder_layouts)
    named_sequences = {
        "value_masks": value_masks,
        "key_masks": key_masks,
        "event_start_masks": event_start_masks,
        "start_continuation_masks": start_continuation_masks,
    }
    for name, values in named_sequences.items():
        if len(values) != sample_count:
            raise ValueError(
                f"Decoder invariant length mismatch for {name}: "
                f"values={len(values)} samples={sample_count}."
            )

    expected_prefix_ids = encode_json_event_array_prefix(tokenizer)
    allowed_start_ids = event_object_start_candidate_token_ids(
        tokenizer,
        len(tokenizer),
    )
    if not allowed_start_ids:
        raise ValueError(
            "The active tokenizer produced no event-object start whitelist tokens."
        )

    def decoded_token(token_id: int) -> str:
        try:
            return tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode([int(token_id)])

    first_event_allowed = 0
    later_event_allowed = 0
    later_event_independent = 0
    later_event_fused = 0
    for sample_index, (layout, value_mask, key_mask, event_start_mask, start_continuation_mask) in enumerate(
        zip(
            decoder_layouts,
            value_masks,
            key_masks,
            event_start_masks,
            start_continuation_masks,
        )
    ):
        root_start = int(layout.json_token_start)
        root_end = int(layout.prefix_end)
        actual_prefix_ids = list(layout.input_ids[root_start:root_end])
        if actual_prefix_ids != expected_prefix_ids:
            raise ValueError(
                f"sample={sample_index} root prefix token mismatch: "
                f"actual={actual_prefix_ids} expected={expected_prefix_ids}"
            )

        masks = {
            "value": value_mask.bool(),
            "key": key_mask.bool(),
            "event_start": event_start_mask.bool(),
            "start_continuation": start_continuation_mask.bool(),
        }
        for mask_name, mask in masks.items():
            if len(mask) != len(layout.input_ids):
                raise ValueError(
                    f"sample={sample_index} mask={mask_name} length={len(mask)} "
                    f"tokens={len(layout.input_ids)}"
                )
            if bool(mask[root_start:root_end].any().item()):
                raise ValueError(
                    f"sample={sample_index} mask={mask_name} perturbs the immutable "
                    f"root prefix token range [{root_start},{root_end})."
                )

        expanded_value_mask = value_mask.bool().clone()
        if len(expanded_value_mask) > 1:
            expanded_value_mask[:-1] |= value_mask.bool()[1:]
        if bool(expanded_value_mask[root_start:root_end].any().item()):
            raise ValueError(
                f"sample={sample_index} left-expanded value mask reaches the "
                "immutable root prefix."
            )
        if not layout.spans.event_object_spans:
            raise ValueError(
                f"sample={sample_index} has no event object to validate."
            )
        for event_index, object_span in enumerate(layout.spans.event_object_spans):
            object_start = int(object_span[0])
            start_token_indices = [
                token_index
                for token_index, (token_start, token_end) in enumerate(layout.offset_mapping)
                if int(token_start) <= object_start < int(token_end)
            ]
            if len(start_token_indices) != 1:
                raise ValueError(
                    f"sample={sample_index} event={event_index} object_start={object_start} "
                    f"maps to token indices {start_token_indices}."
                )
            token_index = start_token_indices[0]
            token_start, token_end = layout.offset_mapping[token_index]
            token_id = int(layout.input_ids[token_index])
            if event_index == 0:
                if token_index != root_end or int(token_start) != object_start:
                    raise ValueError(
                        f"sample={sample_index} first event does not start at stable body "
                        f"boundary: token_index={token_index} prefix_end={root_end} "
                        f"offset=({token_start},{token_end}) object_start={object_start}."
                    )
                if token_id not in allowed_start_ids:
                    raise ValueError(
                        f"sample={sample_index} first event token id={token_id} "
                        f"text={decoded_token(token_id)!r} is outside the event-start whitelist."
                    )
                first_event_allowed += 1
            elif int(token_start) == object_start:
                later_event_independent += 1
                if token_id not in allowed_start_ids:
                    raise ValueError(
                        f"sample={sample_index} event={event_index} independent start "
                        f"token id={token_id} text={decoded_token(token_id)!r} "
                        "is outside the event-start whitelist."
                    )
                later_event_allowed += 1
            elif int(token_start) < object_start:
                later_event_fused += 1
            else:
                raise ValueError(
                    f"sample={sample_index} event={event_index} invalid start offset "
                    f"({token_start},{token_end}) for object_start={object_start}."
                )

    print(
        "Decoder invariant check | "
        f"tokenizer={type(tokenizer).__name__} "
        f"samples={sample_count} "
        f"prefix_match={sample_count} "
        f"root_prefix_clean={sample_count} "
        f"first_event_allowed={first_event_allowed} "
        f"later_event_allowed={later_event_allowed}/{later_event_independent} "
        f"later_event_fused={later_event_fused}",
        flush=True,
    )
    return {
        "samples": sample_count,
        "prefix_match": sample_count,
        "root_prefix_clean": sample_count,
        "first_event_allowed": first_event_allowed,
        "later_event_allowed": later_event_allowed,
        "later_event_independent": later_event_independent,
        "later_event_fused": later_event_fused,
    }


def build_json_train_proxy_masks(
    layout: EventDecoderTokenization,
) -> Dict[str, torch.Tensor]:
    """Mark gold JSON token regions used for lightweight train-proxy metrics.

    These masks do not supervise new heads and do not affect loss.  They only
    let the training logger approximate the paper metrics from teacher-forced
    decoder logits: trigger tokens, argument value tokens, role-conditioned
    argument regions, all value spans, and lightweight head tokens.
    """
    masks = {
        name: torch.zeros(len(layout.input_ids), dtype=torch.bool)
        for name in JSON_TRAIN_PROXY_MASK_NAMES
    }

    def mark_span(char_span: Tuple[int, int], *mask_names: str, head: bool = False):
        token_mask = _torch_mask_from_char_spans(layout, [char_span])
        for mask_name in mask_names:
            masks[mask_name] |= token_mask
        if head:
            indices = token_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() > 0:
                masks["head"][int(indices[-1].item())] = True

    for record in layout.spans.string_spans:
        path = record.path
        if len(path) >= 3 and path[0] == "events" and isinstance(path[1], int):
            if path[2] == "trigger" and record.end > record.start:
                mark_span((record.start, record.end), "trigger", "span", head=True)
            elif path[2] == "arguments" and len(path) >= 4 and record.end > record.start:
                mark_span((record.start, record.end), "arg_i", "arg_c", "span", head=True)
    for record in _event_internal_key_spans(layout):
        if len(record.path) >= 3 and record.path[2] == "arguments":
            mark_span((record.start, record.name_end), "arg_c")
    return masks


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.selection_checkpoint_enable:
        if training_args.selection_checkpoint_start_epoch < 0:
            raise ValueError("selection_checkpoint_start_epoch must be non-negative.")
        if training_args.selection_checkpoint_interval_epochs <= 0:
            raise ValueError(
                "selection_checkpoint_interval_epochs must be greater than zero."
            )
        if training_args.save_total_limit != 1:
            raise ValueError(
                "Selection checkpointing uses one rolling recovery checkpoint; "
                "set SAVE_TOTAL_LIMIT=1."
            )
        save_strategy = str(training_args.save_strategy).lower().split(".")[-1]
        if save_strategy == "no":
            raise ValueError(
                "Selection checkpointing requires a non-'no' SAVE_STRATEGY so "
                "one resumable checkpoint remains available."
            )
    training_args.event_boundary_control = normalize_event_boundary_control(
        training_args.event_boundary_control
    )
    if training_args.native_boundary_loss_factor < 0.0:
        raise ValueError("native_boundary_loss_factor must be non-negative.")
    data_args.data_name = require_data_name(data_args.data_name)
    resume_step = 0
    if str(training_args.restore_from or "").strip():
        training_args.restore_from, resume_step = validate_resume_checkpoint(
            training_args.restore_from
        )
        print(
            f"Validated full checkpoint resume: {training_args.restore_from} "
            f"(global_step={resume_step})",
            flush=True,
        )
    is_event_task = resolve_event_task(data_args.data_name, data_args.task_type)
    if not is_event_task:
        raise ValueError(
            "This slimmed training entrypoint only supports emergency event extraction. "
            "Use --task_type emergency and an emergency DATA_NAME."
        )
    model_args.train = True
    if is_event_task and not training_args.fix_attn_mask:
        logging.warning("Enabling fix_attn_mask for emergency event-handle training.")
        training_args.fix_attn_mask = True

    ##########################
    #       Peft Model       #
    ##########################
    lora_config = None
    if training_args.use_lora and model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = resolve_lora_target_modules(
            model_args.model_name_or_path,
            model_args.lora_target_modules,
        )
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            init_lora_weights=True,
        )


    model = CODI(model_args, training_args, lora_config)
    tokenizer = load_tokenizer_with_fallback(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    if bool(getattr(tokenizer, "is_fast", False)):
        json_alignment_tokenizer = tokenizer
    else:
        json_alignment_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    if not bool(getattr(json_alignment_tokenizer, "is_fast", False)):
        raise TypeError(
            "Event decoder JSON alignment requires the model's fast tokenizer; "
            f"got {type(json_alignment_tokenizer).__name__}."
        )
    print(
        "Decoder JSON token alignment | "
        f"primary={type(tokenizer).__name__} "
        f"offsets={type(json_alignment_tokenizer).__name__}",
        flush=True,
    )

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
        latent_target_steps: Optional[Sequence[int]] = None,
        event_count_targets: Optional[Sequence[int]] = None,
        event_handle_anchor_targets: Optional[Sequence[Sequence[int]]] = None,
        sample_loss_weights: Optional[Sequence[float]] = None,
        is_event_task: bool = False,
    ) -> Dict:
        if not sources:
            raise ValueError(
                "No training examples remain after filtering. Check the dataset path, "
                "max_token_num, model_max_length, and exp_mode settings."
            )
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id for CODI training.")
        if not is_event_task:
            raise ValueError("preprocess only supports emergency event extraction examples.")
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer, training_args.model_max_length)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer, training_args.model_max_length)["input_ids"]
        decoder_answers = [strip_final_json_marker_text(answer) for answer in answers]
        decoder_leading_ids = [eot_id] if training_args.remove_eos else [eot_id, tokenizer.eos_token_id]
        decoder_layouts = [
            tokenize_event_decoder_sequence(
                answer,
                tokenizer,
                json_alignment_tokenizer,
                leading_token_ids=decoder_leading_ids,
                trailing_token_ids=[tokenizer.eos_token_id],
                max_json_tokens=training_args.model_max_length,
            )
            for answer in decoder_answers
        ]
        boundary_enabled = (
            str(training_args.event_boundary_control).lower() != "off"
            or bool(training_args.native_boundary_loss_enable)
        )
        if boundary_enabled:
            boundary_supervisions = [
                build_event_boundary_supervision(
                    layout,
                    tokenizer,
                    len(tokenizer),
                )
                for layout in decoder_layouts
            ]
        else:
            boundary_supervisions = [
                {
                    "label_positions": [],
                    "targets": [],
                    "completed_counts": [],
                    "continue_ids": [],
                    "stop_ids": [],
                }
                for _layout in decoder_layouts
            ]
        max_boundaries = int(training_args.max_event_count)
        max_boundary_candidates = max(
            1,
            max(
                (
                    len(candidate_ids)
                    for supervision in boundary_supervisions
                    for key in ("continue_ids", "stop_ids")
                    for candidate_ids in supervision[key]
                ),
                default=0,
            ),
        )
        total_boundary_decisions = sum(
            len(supervision["targets"])
            for supervision in boundary_supervisions
        )
        total_continue_decisions = sum(
            sum(supervision["targets"])
            for supervision in boundary_supervisions
        )
        print(
            "Decoder boundary supervision | "
            f"mode={training_args.event_boundary_control} "
            f"samples={len(boundary_supervisions)} "
            f"decisions={total_boundary_decisions} "
            f"continue={total_continue_decisions} "
            f"stop={total_boundary_decisions - total_continue_decisions} "
            f"candidate_width={max_boundary_candidates}",
            flush=True,
        )

        def padded_boundary_vector(values, fill_value):
            values = list(values)
            if len(values) > max_boundaries:
                raise ValueError(
                    f"Boundary count {len(values)} exceeds max_event_count={max_boundaries}."
                )
            return torch.tensor(
                values + [fill_value] * (max_boundaries - len(values)),
                dtype=torch.long,
            )

        def padded_boundary_candidates(values):
            rows = []
            for candidate_ids in values:
                candidate_ids = list(candidate_ids)
                if len(candidate_ids) > max_boundary_candidates:
                    raise ValueError("Boundary candidate width changed during packing.")
                rows.append(
                    candidate_ids
                    + [-1] * (max_boundary_candidates - len(candidate_ids))
                )
            rows.extend(
                [[-1] * max_boundary_candidates]
                * (max_boundaries - len(rows))
            )
            return torch.tensor(rows, dtype=torch.long)

        answers_id = [
            torch.tensor(layout.input_ids, dtype=torch.long)
            for layout in decoder_layouts
        ]

        # Preserve the pretrained EOS convention on source/process paths.
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]

        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None:
            cot_id = [x[1:] if len(x) and x[0].item() == bos_token_id else x for x in cot_id]

        # Add the same BOT boundary to the student encoder path and the
        # teacher-process reference path. This keeps explicit process hidden
        # states and implicit latent states in the same answer-side context.
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        ref_input_ids = [
            torch.cat([source_ids, process_ids]).to(torch.long)
            for source_ids, process_ids in zip(sources_id, cot_id)
        ]

        process_start_marker_ids = [
            marker_token_variants(marker, tokenizer)
            for marker in PROCESS_MARKERS[: len(PROCESS_DONE_MARKERS)]
        ]
        process_done_marker_ids = [
            marker_token_variants(marker, tokenizer)
            for marker in PROCESS_DONE_MARKERS
        ]
        process_ref_ranges = []
        for token_ids in ref_input_ids:
            token_list = token_ids.tolist()
            ranges = []
            for start_variants, done_variants in zip(process_start_marker_ids, process_done_marker_ids):
                start_span = find_marker_span(token_list, start_variants)
                done_span = find_marker_span(token_list, done_variants)
                if start_span[0] < 0 or done_span[0] < 0:
                    raise ValueError("Cannot find one or more process markers in event extraction teacher process.")
                content_start = start_span[1]
                content_end = done_span[0]
                if content_end <= content_start:
                    content_start, content_end = start_span[0], done_span[1]
                ranges.append([content_start, content_end])
            process_ref_ranges.append(ranges)

        output = dict(
            encoder_input_ids=sources_id,
            decoder_input_ids=answers_id,
            ref_input_ids=ref_input_ids,
            labels=answers_id,
        )
        output["document_token_mask"] = [
            torch.tensor(
                build_document_token_mask_from_source(
                    token_ids.tolist(),
                    tokenizer,
                    source_text,
                    max_sentence_markers=training_args.max_sentence_markers,
                ),
                dtype=torch.bool,
            )
            for token_ids, source_text in zip(sources_id, sources)
        ]
        decoder_token_weights = [
            build_json_structure_token_weights(
                layout,
                training_args.json_structure_token_weight,
                training_args.event_boundary_token_weight,
                training_args.json_value_token_weight,
                boundary_label_positions=(
                    supervision["label_positions"]
                    if boundary_enabled
                    else None
                ),
            )
            for layout, supervision in zip(decoder_layouts, boundary_supervisions)
        ]
        decoder_prefix_robust_masks = [
            build_json_prefix_robust_mask(layout)
            for layout in decoder_layouts
        ]
        decoder_prefix_key_masks = [
            build_json_prefix_key_robust_mask(layout)
            for layout in decoder_layouts
        ]
        decoder_prefix_event_start_masks = [
            build_json_prefix_event_start_robust_mask(layout)
            for layout in decoder_layouts
        ]
        decoder_start_continuation_masks = [
            build_json_start_continuation_mask(
                layout,
                getattr(training_args, "json_start_continuation_max_tokens", 0),
            )
            for layout in decoder_layouts
        ]
        output["decoder_token_weights"] = decoder_token_weights
        output["decoder_prefix_robust_mask"] = decoder_prefix_robust_masks
        output["decoder_prefix_key_robust_mask"] = decoder_prefix_key_masks
        output["decoder_prefix_event_start_robust_mask"] = decoder_prefix_event_start_masks
        output["decoder_start_continuation_mask"] = decoder_start_continuation_masks
        output["decoder_boundary_label_positions"] = [
            padded_boundary_vector(supervision["label_positions"], -1)
            for supervision in boundary_supervisions
        ]
        output["decoder_boundary_targets"] = [
            padded_boundary_vector(supervision["targets"], -1)
            for supervision in boundary_supervisions
        ]
        output["decoder_boundary_completed_counts"] = [
            padded_boundary_vector(supervision["completed_counts"], 0)
            for supervision in boundary_supervisions
        ]
        output["decoder_boundary_continue_ids"] = [
            padded_boundary_candidates(supervision["continue_ids"])
            for supervision in boundary_supervisions
        ]
        output["decoder_boundary_stop_ids"] = [
            padded_boundary_candidates(supervision["stop_ids"])
            for supervision in boundary_supervisions
        ]

        check_decoder_invariants(
            decoder_layouts,
            tokenizer,
            decoder_prefix_robust_masks,
            decoder_prefix_key_masks,
            decoder_prefix_event_start_masks,
            decoder_start_continuation_masks,
        )
        decoder_proxy_masks = {
            name: []
            for name in JSON_TRAIN_PROXY_MASK_NAMES
        }
        for layout in decoder_layouts:
            proxy_masks = build_json_train_proxy_masks(layout)
            for name in JSON_TRAIN_PROXY_MASK_NAMES:
                decoder_proxy_masks[name].append(proxy_masks[name])
        for name, values in decoder_proxy_masks.items():
            output[f"decoder_proxy_{name}_mask"] = values
        output["process_ref_ranges"] = process_ref_ranges
        if latent_target_steps is not None:
            output["latent_target_steps"] = list(latent_target_steps)
        if event_count_targets is not None:
            output["event_count_targets"] = list(event_count_targets)
        output["sentence_rep_positions"] = [
            build_sentence_rep_positions_from_source(
                token_ids.tolist(),
                tokenizer,
                source_text,
                max_sentence_markers=training_args.max_sentence_markers,
            )
            for token_ids, source_text in zip(sources_id, sources)
        ]
        if event_handle_anchor_targets is not None:
            output["event_handle_anchor_targets"] = [list(values) for values in event_handle_anchor_targets]
        if sample_loss_weights is not None:
            output["sample_loss_weights"] = list(sample_loss_weights)
        return output


    class SupervisedDataset(Dataset):
        def __init__(self, data_name, raw_data, tokenizer, bot, eot, is_event_task=False):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")
            if not is_event_task:
                raise ValueError("SupervisedDataset only supports emergency event extraction.")

            self.data_name = data_name
            self.is_event_task = True
            questions, cots, answers = [], [], []
            latent_target_steps = []
            event_count_targets = []
            event_handle_anchor_targets = []
            sample_loss_weights = []
            skipped_too_long = 0
            skipped_too_many_events = 0
            effective_path_limit = max(
                1,
                min(int(training_args.max_token_num), int(training_args.model_max_length)),
            )

            exp_limit = max(0, int(training_args.exp_data_num))
            for example in raw_data:
                if training_args.exp_mode and len(questions) >= exp_limit:
                    break
                question = f"{example.get('question', '')}"
                if self.is_event_task:
                    question = build_event_prompt(example)
                    question = format_prompt_for_model(
                        question,
                        tokenizer,
                        use_chat_template=training_args.use_chat_template,
                    )
                    cot = build_teacher_process(
                        example,
                        max_sentence_markers=training_args.max_sentence_markers,
                    )
                    answer = build_final_answer(example)
                    _, reference_length, decoder_length = training_path_token_lengths(
                        question,
                        cot,
                        answer,
                        tokenizer,
                        bot_id=bot,
                        eot_id=eot,
                        num_latent=training_args.num_latent,
                        max_event_count=training_args.max_event_count,
                        remove_eos=training_args.remove_eos,
                    )
                    if max(reference_length, decoder_length) > effective_path_limit:
                        skipped_too_long += 1
                        continue
                    ordered_events = canonical_events(example.get("events", []))
                    if len(ordered_events) > int(training_args.max_event_count):
                        skipped_too_many_events += 1
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                    latent_target_steps.append(example_complexity(example, training_args.num_latent))
                    features = example.get("complexity_features", {})
                    event_count_targets.append(len(ordered_events))
                    event_count = len(ordered_events)
                    extra_complexity = max(0, event_count - 1)
                    extra_complexity += int(bool(features.get("same_type_events", False)))
                    extra_complexity += int(bool(features.get("type_confusion", False)))
                    extra_complexity += int(bool(features.get("tertiary_event", False)))
                    sample_weight_min = float(getattr(training_args, "sample_loss_weight_min", 0.5))
                    sample_weight_max = float(getattr(training_args, "sample_loss_weight_max", 1.6))
                    if sample_weight_max < sample_weight_min:
                        sample_weight_max = sample_weight_min
                    complex_weight = min(
                        sample_weight_max,
                        1.0 + float(training_args.complex_sample_loss_factor) * extra_complexity,
                    )
                    try:
                        quality_weight = float(
                            example.get(
                                "sample_loss_weight",
                                example.get("quality_weight", 1.0),
                            )
                        )
                    except (TypeError, ValueError):
                        quality_weight = 1.0
                    quality_weight = max(sample_weight_min, min(1.15, quality_weight))
                    sample_loss_weights.append(
                        max(sample_weight_min, min(sample_weight_max, complex_weight * quality_weight))
                    )
                    sentences = [str(sentence) for sentence in example.get("sentences", [])]
                    handle_anchor_values = [
                        resolve_event_identity_anchor(
                            event,
                            sentences,
                            max_sentence_markers=training_args.max_sentence_markers,
                        )
                        for event in ordered_events
                    ]
                    handle_anchor_values = handle_anchor_values[: training_args.max_event_count]
                    while len(handle_anchor_values) < training_args.max_event_count:
                        handle_anchor_values.append(-100)
                    event_handle_anchor_targets.append(handle_anchor_values)

            if sample_loss_weights:
                raw_weight_min = min(sample_loss_weights)
                raw_weight_max = max(sample_loss_weights)
                raw_weight_mean = sum(sample_loss_weights) / len(sample_loss_weights)
                sample_loss_weights = [
                    weight / raw_weight_mean for weight in sample_loss_weights
                ]
                print(
                    "Sample loss weights | "
                    f"raw_min={raw_weight_min:.4f} raw_mean={raw_weight_mean:.4f} raw_max={raw_weight_max:.4f} "
                    f"normalized_min={min(sample_loss_weights):.4f} "
                    f"normalized_mean={sum(sample_loss_weights) / len(sample_loss_weights):.4f} "
                    f"normalized_max={max(sample_loss_weights):.4f}"
                )
            
            print(f"{len(cots)} data in total...")
            print(
                "Training eligibility | "
                f"positive_input={len(raw_data)} skipped_overlength={skipped_too_long} "
                f"skipped_event_count={skipped_too_many_events} selected={len(cots)}"
            )
            if latent_target_steps:
                budget_counts = Counter(latent_target_steps)
                print(
                    "Latent budget targets | "
                    + " ".join(
                        f"T{step}={budget_counts[step]}"
                        for step in sorted(budget_counts)
                    )
                )
            if skipped_too_long:
                print(
                    f"Skipped {skipped_too_long} overlength examples "
                    f"(exact_reference_or_decoder_limit={effective_path_limit}, "
                    f"model_max_length={training_args.model_max_length})."
                )
            if skipped_too_many_events:
                print(
                    f"Skipped {skipped_too_many_events} examples with event_count > "
                    f"max_event_count ({training_args.max_event_count}) after length filtering."
                )
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(
                questions,
                cots,
                answers,
                tokenizer,
                bot,
                eot,
                latent_target_steps=latent_target_steps,
                event_count_targets=event_count_targets if event_count_targets else None,
                event_handle_anchor_targets=event_handle_anchor_targets if event_handle_anchor_targets else None,
                sample_loss_weights=sample_loss_weights if sample_loss_weights else None,
                is_event_task=self.is_event_task,
            )
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels = tuple(
                [instance[key] for instance in instances]
                for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels")
            )
            process_ref_ranges = [instance["process_ref_ranges"] for instance in instances] if "process_ref_ranges" in instances[0] else None
            latent_target_steps = [instance["latent_target_steps"] for instance in instances] if "latent_target_steps" in instances[0] else None
            event_count_targets = [instance["event_count_targets"] for instance in instances] if "event_count_targets" in instances[0] else None
            sentence_rep_positions = [instance["sentence_rep_positions"] for instance in instances] if "sentence_rep_positions" in instances[0] else None
            event_handle_anchor_targets = [instance["event_handle_anchor_targets"] for instance in instances] if "event_handle_anchor_targets" in instances[0] else None
            document_token_mask = [instance["document_token_mask"] for instance in instances] if "document_token_mask" in instances[0] else None
            decoder_token_weights = [instance["decoder_token_weights"] for instance in instances] if "decoder_token_weights" in instances[0] else None
            decoder_prefix_robust_mask = [instance["decoder_prefix_robust_mask"] for instance in instances] if "decoder_prefix_robust_mask" in instances[0] else None
            decoder_prefix_key_robust_mask = [instance["decoder_prefix_key_robust_mask"] for instance in instances] if "decoder_prefix_key_robust_mask" in instances[0] else None
            decoder_prefix_event_start_robust_mask = [instance["decoder_prefix_event_start_robust_mask"] for instance in instances] if "decoder_prefix_event_start_robust_mask" in instances[0] else None
            decoder_start_continuation_mask = [instance["decoder_start_continuation_mask"] for instance in instances] if "decoder_start_continuation_mask" in instances[0] else None
            decoder_boundary_fields = {
                name: [instance[name] for instance in instances]
                for name in (
                    "decoder_boundary_label_positions",
                    "decoder_boundary_targets",
                    "decoder_boundary_completed_counts",
                    "decoder_boundary_continue_ids",
                    "decoder_boundary_stop_ids",
                )
                if name in instances[0]
            }
            decoder_proxy_masks = {
                name: [instance[f"decoder_proxy_{name}_mask"] for instance in instances]
                for name in JSON_TRAIN_PROXY_MASK_NAMES
                if f"decoder_proxy_{name}_mask" in instances[0]
            }
            sample_loss_weights = [instance["sample_loss_weights"] for instance in instances] if "sample_loss_weights" in instances[0] else None
        
            # Pad encoder-side tensors on the right.  Qwen's SDPA + KV-cache
            # path is sensitive to left-padded batch prefixes during latent
            # rollout; right padding keeps the 2D attention mask aligned with
            # the cached key/value length when micro-batch > 1.
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
                encoder_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            if document_token_mask is not None:
                document_token_mask = torch.nn.utils.rnn.pad_sequence(
                    document_token_mask,
                    batch_first=True,
                    padding_value=False,
                )
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if decoder_token_weights is not None:
                decoder_token_weights = torch.nn.utils.rnn.pad_sequence(
                    decoder_token_weights,
                    batch_first=True,
                    padding_value=1.0,
                )
            if decoder_prefix_robust_mask is not None:
                decoder_prefix_robust_mask = torch.nn.utils.rnn.pad_sequence(
                    decoder_prefix_robust_mask,
                    batch_first=True,
                    padding_value=False,
                )
            if decoder_prefix_key_robust_mask is not None:
                decoder_prefix_key_robust_mask = torch.nn.utils.rnn.pad_sequence(
                    decoder_prefix_key_robust_mask,
                    batch_first=True,
                    padding_value=False,
                )
            if decoder_prefix_event_start_robust_mask is not None:
                decoder_prefix_event_start_robust_mask = torch.nn.utils.rnn.pad_sequence(
                    decoder_prefix_event_start_robust_mask,
                    batch_first=True,
                    padding_value=False,
                )
            if decoder_start_continuation_mask is not None:
                decoder_start_continuation_mask = torch.nn.utils.rnn.pad_sequence(
                    decoder_start_continuation_mask,
                    batch_first=True,
                    padding_value=False,
                )
            padded_decoder_proxy_masks = {
                name: torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=False,
                )
                for name, values in decoder_proxy_masks.items()
            }
          
            batch = dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            )
            if process_ref_ranges is not None:
                batch["process_ref_ranges"] = torch.tensor(process_ref_ranges, dtype=torch.long)
            if latent_target_steps is not None:
                batch["latent_target_steps"] = torch.tensor(latent_target_steps, dtype=torch.long)
            if event_count_targets is not None:
                batch["event_count_targets"] = torch.tensor(event_count_targets, dtype=torch.long)
            if sentence_rep_positions is not None:
                batch["sentence_rep_positions"] = torch.tensor(sentence_rep_positions, dtype=torch.long)
            if event_handle_anchor_targets is not None:
                batch["event_handle_anchor_targets"] = torch.tensor(event_handle_anchor_targets, dtype=torch.long)
            if document_token_mask is not None:
                batch["document_token_mask"] = document_token_mask.bool()
            if decoder_token_weights is not None:
                batch["decoder_token_weights"] = decoder_token_weights.float()
            if decoder_prefix_robust_mask is not None:
                batch["decoder_prefix_robust_mask"] = decoder_prefix_robust_mask.bool()
            if decoder_prefix_key_robust_mask is not None:
                batch["decoder_prefix_key_robust_mask"] = decoder_prefix_key_robust_mask.bool()
            if decoder_prefix_event_start_robust_mask is not None:
                batch["decoder_prefix_event_start_robust_mask"] = decoder_prefix_event_start_robust_mask.bool()
            if decoder_start_continuation_mask is not None:
                batch["decoder_start_continuation_mask"] = decoder_start_continuation_mask.bool()
            for name, values in decoder_boundary_fields.items():
                batch[name] = torch.stack(values, dim=0).long()
            for name, value in padded_decoder_proxy_masks.items():
                batch[f"decoder_proxy_{name}_mask"] = value.bool()
            if sample_loss_weights is not None:
                batch["sample_loss_weights"] = torch.tensor(sample_loss_weights, dtype=torch.float)
            return batch

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Loading emergency event extraction data")
        dataset = load_emergency_examples(data_args.data_name, split="train")
        raw_count = len(dataset)
        dataset, skipped_zero_event = filter_positive_event_examples(dataset)
        print(
            "Training event filter | "
            f"raw={raw_count} skipped_zero_event={skipped_zero_event} kept={len(dataset)}"
        )
        dataset = limit_dataset_for_debug(
            dataset, data_args.debug_data, data_args.debug_data_size
        )
        train_dataset = SupervisedDataset(
            data_name=data_args.data_name,
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id,
            is_event_task=True,
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if resume_step > 0:
        trainer.add_callback(ResumeVerificationCallback(resume_step))
    if training_args.selection_checkpoint_enable:
        trainer.add_callback(SelectionCheckpointCallback(trainer))
    resume_from_checkpoint = training_args.restore_from or None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Safetensors cannot serialize some tied GPT-style weights from the outer CODI
    # wrapper, so detach them before the final save when present.
    if hasattr(trainer.model.codi, "transformer") and hasattr(trainer.model.codi, "lm_head"):
        trainer.model.codi.lm_head.weight = torch.nn.Parameter(
            trainer.model.codi.lm_head.weight.detach().clone()
        )

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    trainer.register_final_selection_candidate()


if __name__ == "__main__":
    train()
