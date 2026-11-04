import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from peft import (
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import math
from safetensors.torch import load_file
from src.event_boundary import (
    blend_event_boundary_logits,
    conditional_event_continue_probability_tensor,
    event_boundary_proposal_mask,
    event_count_distribution_tensor,
    event_count_reliability_tensor,
    event_count_target_nll,
    mean_event_boundary_loss_per_sample,
    native_event_boundary_statistics,
)
from src.event_utils import MIN_LATENT_BUDGET, validate_event_configuration
from src.event_decode_state import normalize_event_boundary_control
from src.set_utils import minimum_cost_bipartite_assignment
try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HANDLE_QUERY_SEED_NORM_FLOOR = 0.25


def stabilize_handle_query_seed(
    query_seed,
    norm_floor=HANDLE_QUERY_SEED_NORM_FLOOR,
):
    """Normalize handle queries without amplifying near-cancelled directions."""

    floor = float(norm_floor)
    if floor <= 0.0:
        raise ValueError("Handle query seed norm floor must be greater than zero.")
    seed_norm = query_seed.float().norm(dim=-1, keepdim=True)
    stabilized = query_seed.float() / seed_norm.clamp_min(floor)
    return stabilized, seed_norm


def auxiliary_gate_cross_entropy(
    token_loss,
    loss_weights,
    boundary_token_mask,
    sample_mask,
):
    """Average non-boundary CE, falling back per sample when none is available."""

    if (
        token_loss.shape != loss_weights.shape
        or token_loss.shape != boundary_token_mask.shape
    ):
        raise ValueError(
            "Auxiliary-gate CE tensors must have matching [B,L] shapes: "
            f"loss={tuple(token_loss.shape)}, "
            f"weights={tuple(loss_weights.shape)}, "
            f"boundary={tuple(boundary_token_mask.shape)}."
        )
    full_den = loss_weights.sum(dim=1)
    full_per_sample = (
        token_loss * loss_weights
    ).sum(dim=1) / full_den.clamp_min(1.0)
    gate_weights = loss_weights.masked_fill(boundary_token_mask.bool(), 0.0)
    gate_den = gate_weights.sum(dim=1)
    gate_per_sample = (
        token_loss * gate_weights
    ).sum(dim=1) / gate_den.clamp_min(1.0)
    gate_per_sample = torch.where(
        gate_den.gt(0.0),
        gate_per_sample,
        full_per_sample,
    )
    active_samples = sample_mask.to(
        device=token_loss.device,
        dtype=torch.bool,
    )
    if active_samples.any():
        return gate_per_sample[active_samples].mean()
    return token_loss.sum() * 0.0


def _looks_like_tokenizer(obj):
    return (not isinstance(obj, bool)) and hasattr(obj, "pad_token_id") and hasattr(obj, "encode")


def load_tokenizer_with_fallback(model_name_or_path, **kwargs):
    errors = []
    for use_fast in (False, True):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast, **kwargs)
            if _looks_like_tokenizer(tokenizer):
                return tokenizer
            errors.append(f"AutoTokenizer(use_fast={use_fast}) returned {type(tokenizer).__name__}")
        except Exception as exc:
            errors.append(f"AutoTokenizer(use_fast={use_fast}) failed: {exc}")

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, **kwargs)
        if _looks_like_tokenizer(tokenizer):
            return tokenizer
        errors.append(f"PreTrainedTokenizerFast returned {type(tokenizer).__name__}")
    except Exception as exc:
        errors.append(f"PreTrainedTokenizerFast failed: {exc}")

    raise TypeError(f"Cannot load a valid tokenizer from {model_name_or_path!r}. Attempts: {' | '.join(errors)}")


def resolve_checkpoint_weight_path(checkpoint_path):
    """Resolve a checkpoint directory or weight file to a supported weight file."""
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required.")

    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        for filename in ("model.safetensors", "pytorch_model.bin"):
            candidate = path / filename
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Cannot find model.safetensors or pytorch_model.bin in {str(path)!r}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {str(path)!r}")
    if path.suffix not in {".safetensors", ".bin"}:
        raise ValueError(
            f"Unsupported checkpoint file {str(path)!r}; expected .safetensors or .bin."
        )
    return path


def load_checkpoint_state_dict(checkpoint_path):
    """Load a compact CODI/APLER state dict from a directory or weight file."""
    weight_path = resolve_checkpoint_weight_path(checkpoint_path)
    if weight_path.suffix == ".safetensors":
        state_dict = load_file(str(weight_path))
    else:
        try:
            state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(str(weight_path), map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint {str(weight_path)!r} did not contain a state dictionary."
        )
    return state_dict


LEGACY_CHECKPOINT_MARKERS = (
    "event_slot_",
    "slot_latent_",
)


def current_checkpoint_requirements(model):
    """Return current-version task modules that must be present in compact checkpoints."""
    requirements = [
        (("latent_budget_head.",), "adaptive latent reasoning", getattr(model, "adaptive_latent", False)),
        (("latent_boundary_embedding.",), "latent boundary embeddings", hasattr(model, "latent_boundary_embedding")),
        (("prj.",), "latent projection", getattr(model, "use_prj", False)),
        (
            ("latent_budget_head.", "latent_budget_embedding."),
            "latent budget conditioning",
            getattr(model, "adaptive_latent", False)
            and hasattr(model, "latent_budget_head")
            and float(getattr(model, "latent_budget_condition_factor", 0.0)) > 0,
        ),
        (
            (
                "event_handle_embedding.",
                "event_handle_query.",
                "event_handle_active_embedding.",
                "event_handle_anchor_proj.",
                "event_handle_query_norm.",
                "event_handle_condition_proj.",
                "event_handle_exist_head.",
            ),
            "event-handle planner",
            hasattr(model, "event_handle_embedding"),
        ),
        (
            (
                "event_handle_budget_handle_proj.",
                "event_handle_budget_budget_proj.",
                "event_handle_budget_out.",
            ),
            "budget-conditioned event-handle count",
            hasattr(model, "event_handle_embedding")
            and bool(getattr(model, "budget_conditioned_handle_count", False))
            and float(getattr(model, "budget_handle_count_factor", 0.0)) > 0,
        ),
        (
            (
                "handle_latent_query_proj.",
                "handle_latent_key_proj.",
                "handle_latent_value_proj.",
                "handle_latent_gate.",
            ),
            "event-handle latent attention",
            float(getattr(model, "event_handle_latent_attention_factor", 0.0)) > 0,
        ),
        (
            ("decoder_bridge_proj.", "decoder_bridge_gate."),
            "latent-to-JSON decoder bridge",
            float(getattr(model, "decoder_bridge_factor", 0.0)) > 0,
        ),
        (
            ("decoder_bridge_contribution_gate.",),
            "sample-level decoder bridge contribution gate",
            float(getattr(model, "decoder_bridge_factor", 0.0)) > 0
            and bool(getattr(model, "adaptive_bridge_gate", False)),
        ),
        (
            ("document_summary_gate.",),
            "document summary fusion",
            float(getattr(model, "document_summary_fusion_factor", 0.0)) > 0,
        ),
    ]
    return requirements


def validate_current_checkpoint_load(load_result, model, checkpoint_path):
    """Reject checkpoints that do not contain all enabled current-version APLER modules."""
    missing_keys = list(getattr(load_result, "missing_keys", []))
    missing_groups = []
    for prefixes, label, enabled in current_checkpoint_requirements(model):
        if not enabled:
            continue
        missing_prefixes = sorted(
            prefix
            for prefix in prefixes
            if any(key.startswith(prefix) for key in missing_keys)
        )
        if missing_prefixes:
            missing_groups.append(f"{label}: {', '.join(missing_prefixes)}")
    if missing_groups:
        raise RuntimeError(
            "Checkpoint is missing required current-version APLER weights. "
            f"Please retrain with the current code/configuration before use. Missing: {'; '.join(missing_groups)}"
        )


def load_checkpoint_into_model(model, checkpoint_path, require_lora=False, require_current_modules=False):
    """Load a compact checkpoint and reject empty or incomplete adapter loads."""
    state_dict = load_checkpoint_state_dict(checkpoint_path)
    legacy_keys = sorted(
        key
        for key in state_dict
        if any(marker in key for marker in LEGACY_CHECKPOINT_MARKERS)
    )
    if legacy_keys:
        preview = ", ".join(legacy_keys[:8])
        suffix = " ..." if len(legacy_keys) > 8 else ""
        raise RuntimeError(
            "Checkpoint contains removed legacy APLER weights "
            f"({preview}{suffix}). Retrain with the current event-handle implementation."
        )
    load_result = model.load_state_dict(state_dict, strict=False)
    unexpected = set(getattr(load_result, "unexpected_keys", []))
    matched_keys = [key for key in state_dict if key not in unexpected]
    if not matched_keys:
        raise RuntimeError(
            f"Checkpoint {str(checkpoint_path)!r} has no parameters matching the current model."
        )
    if require_lora and not any("lora_" in key for key in matched_keys):
        raise RuntimeError(
            f"Checkpoint {str(checkpoint_path)!r} does not contain matching LoRA weights."
        )
    if require_current_modules:
        validate_current_checkpoint_load(load_result, model, checkpoint_path)
    return load_result


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(
        default="auto",
        metadata={
            "help": "Comma-separated LoRA target modules. Use 'auto' for the default backbone-specific set."
        },
    )
    full_precision: bool = field(
        default=True,
        metadata={"help": "Load normal floating-point weights. Set False for NF4 4-bit loading."},
    )
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "Initialize a new LoRA adapter. When False, adapter_name_or_path is required if use_lora=True."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path or Hugging Face id of a PEFT LoRA adapter to load."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})

@dataclass
class DataArguments:
    data_name: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_split: str = field(
        default="test", metadata={"help": "Emergency extraction split used during evaluation."}
    )
    task_type: str = field(
        default="auto",
        metadata={"help": "Task routing: auto or emergency."},
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    debug_data_size: int = field(
        default=32,
        metadata={"help": "Maximum examples loaded when debug_data=True."},
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="apler_qwen3b_bridge_denoise_startcont_softalign_lr16_ep5",
        metadata={"help": "Experiment name"},
    )
    selection_checkpoint_enable: bool = field(
        default=False,
        metadata={
            "help": "Save model-only late-training snapshots for offline checkpoint selection."
        },
    )
    selection_checkpoint_start_epoch: float = field(
        default=3.0,
        metadata={"help": "First epoch eligible for a model-only selection snapshot."},
    )
    selection_checkpoint_interval_epochs: float = field(
        default=0.5,
        metadata={"help": "Epoch interval between model-only selection snapshots."},
    )
    write_eval_report: bool = field(default=True, metadata={"help": "Write emergency evaluation summary and good-sample traces to a txt report."})
    eval_report_dir: str = field(default="", metadata={"help": "Directory for emergency evaluation txt reports. Empty defaults to output_dir/eval_reports."})
    write_eval_json_report: bool = field(default=True, metadata={"help": "Write machine-readable emergency evaluation metrics and per-document counts to JSON."})
    eval_json_report_dir: str = field(default="", metadata={"help": "Directory for emergency evaluation JSON reports. Empty reuses eval_report_dir or output_dir/eval_reports."})
    eval_json_report_path: str = field(default="", metadata={"help": "Exact machine-readable evaluation JSON output path. When set, overrides eval_json_report_dir and timestamp naming."})
    good_sample_f1_threshold: float = field(default=0.50, metadata={"help": "Minimum relaxed diagnostic F1 required for every relaxed metric in saved good-sample traces."})
    good_sample_relaxed_peak_threshold: float = field(default=0.60, metadata={"help": "At least one relaxed diagnostic F1 must exceed this threshold for saved good-sample traces."})
    max_good_sample_reports: int = field(default=8, metadata={"help": "Maximum number of good-sample document traces saved in the evaluation report."})
    num_latent: int = field(default=9, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_layer_strategy: str = field(default="full", metadata={"help": "Layer set for latent/reference distillation: full, selected, or last."})
    distill_selected_layer_count: int = field(default=4, metadata={"help": "Number of representative layers to capture when distill_layer_strategy=selected."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    generation_temperature: float = field(default=0.7, metadata={"help": "Sampling temperature for event extraction inference."})
    generation_top_k: int = field(default=40, metadata={"help": "Top-k sampling for event extraction inference."})
    generation_top_p: float = field(default=0.95, metadata={"help": "Top-p sampling for event extraction inference."})
    generation_max_tokens: int = field(default=1024, metadata={"help": "Maximum generated tokens for event extraction inference."})
    force_json_prefix: bool = field(default=False, metadata={"help": "Force-feed the full JSON root prefix during generation. Disabled by default to avoid nested events roots."})
    json_start_bias: float = field(default=1.5, metadata={"help": "Small logit bias for JSON opening tokens at the first generation step."})
    semantic_json_stop: bool = field(default=True, metadata={"help": "During event inference, only stop on complete JSON when it contains a non-empty event payload."})
    event_boundary_control: str = field(
        default="off",
        metadata={
            "help": "Shared count-survival control: off, handle_soft (legacy), or handle_hierarchical (native boundary commitment)."
        },
    )
    native_boundary_loss_enable: bool = field(
        default=False,
        metadata={
            "help": "Supervise unfused decoder continue/stop group mass at every gold event boundary."
        },
    )
    native_boundary_loss_factor: float = field(
        default=0.10,
        metadata={
            "help": "Weight for per-sample-normalized native continue/stop boundary BCE."
        },
    )
    adaptive_latent: bool = field(default=False, metadata={"help": "Use a halt head to learn adaptive latent reasoning steps."})
    latent_budget_loss_factor: float = field(default=0.2, metadata={"help": "Weight for document-complexity latent-budget prediction."})
    latent_budget_ordinal_loss_factor: float = field(default=0.25, metadata={"help": "Weight for expected-budget SmoothL1 inside the latent-budget objective."})
    latent_budget_condition_factor: float = field(default=0.10, metadata={"help": "Inject a planned latent-step budget embedding into latent reasoning."})
    latent_budget_condition_teacher_forcing: float = field(default=0.5, metadata={"help": "Gold-budget mixing ratio for latent-budget conditioning during training."})
    decoder_exit_mix_enable: bool = field(default=True, metadata={"help": "Train JSON decoding from a warmup-gated mix of gold and predicted latent exits."})
    decoder_exit_mix_prob: float = field(default=0.35, metadata={"help": "Per-sample probability of using the predicted latent-budget exit for the existing JSON CE."})
    decoder_exit_mix_warmup_ratio: float = field(default=0.25, metadata={"help": "Training progress ratio before predicted decoder exits can be sampled."})
    decoder_exit_mix_min_step: int = field(default=3, metadata={"help": "Minimum predicted latent step eligible for decoder-exit mixing."})
    max_event_count: int = field(default=9, metadata={"help": "Maximum event count class for event planning and structure supervision."})
    event_handle_condition_factor: float = field(default=0.25, metadata={"help": "Inject lightweight event-handle tokens into latent reasoning."})
    event_handle_condition_teacher_forcing: float = field(default=0.5, metadata={"help": "Gold-anchor mixing ratio for lightweight event handles during training."})
    event_handle_teacher_forcing_decay_ratio: float = field(default=0.35, metadata={"help": "Fraction of training steps used to anneal event-handle gold-anchor mixing to zero."})
    event_handle_anchor_loss_factor: float = field(default=0.20, metadata={"help": "Weight for set-matched event-handle anchor supervision."})
    event_handle_exist_loss_factor: float = field(default=0.15, metadata={"help": "Weight for event-handle existence supervision."})
    event_handle_exist_count_loss_factor: float = field(default=0.80, metadata={"help": "Weight for gold-count NLL under the event-handle count distribution."})
    event_handle_match_anchor_cost: float = field(default=1.0, metadata={"help": "Anchor component weight in permutation-invariant event-handle matching."})
    event_handle_latent_attention_factor: float = field(default=0.20, metadata={"help": "Let latent reasoning attend to lightweight event-handle memory."})
    decoder_bridge_factor: float = field(default=0.0, metadata={"help": "Insert trainable event-level latent-to-JSON bridge tokens before answer decoding."})
    document_summary_fusion_factor: float = field(default=0.35, metadata={"help": "Fuse pooled document-token evidence with the last-token summary for event heads and latent initialization."})
    complex_sample_loss_factor: float = field(default=0.25, metadata={"help": "Extra loss weight for complex multi-event training samples."})
    complex_sample_weight_mode: str = field(default="full_objective", metadata={"help": "How complex sample weights enter training: full_objective or ce_only."})
    sample_loss_weight_min: float = field(default=0.5, metadata={"help": "Lower clamp for per-sample loss weights."})
    sample_loss_weight_max: float = field(default=1.6, metadata={"help": "Upper clamp for per-sample loss weights. Keeps noisy complex samples from dominating batch-1 training."})
    event_type_count: int = field(default=5, metadata={"help": "Number of event type labels for auxiliary supervision."})
    role_type_count: int = field(default=12, metadata={"help": "Number of role labels for auxiliary supervision."})
    max_sentence_markers: int = field(default=32, metadata={"help": "Maximum sentence markers supervised for evidence selection."})
    json_structure_token_weight: float = field(default=2.0, metadata={"help": "Relative CE weight for JSON keys and delimiters in event extraction answers."})
    event_boundary_token_weight: float = field(default=4.0, metadata={"help": "Relative CE weight applied equally to semantic event continue and stop decisions."})
    json_value_token_weight: float = field(default=2.0, metadata={"help": "Relative CE weight for trigger, evidence ids, and argument value tokens in event JSON."})
    ce_token_weight_warmup_ratio: float = field(default=0.08, metadata={"help": "Warm up JSON/value/boundary token weights from 1.0 to their configured values to avoid early gradient spikes."})
    ce_token_weight_warmup_steps: int = field(default=80, metadata={"help": "Absolute optimizer-step warmup for JSON/value/boundary token weights; takes precedence over ce_token_weight_warmup_ratio when >0."})
    json_prefix_robust_enable: bool = field(default=True, metadata={"help": "Enable bridge-conditioned structured denoising on teacher-forced JSON prefixes during training."})
    json_prefix_value_perturb: bool = field(default=True, metadata={"help": "Perturb value-related decoder-prefix tokens during JSON denoising."})
    json_prefix_value_dropout: float = field(default=-1.0, metadata={"help": "Dropout probability for value-prefix perturbation. Negative falls back to json_prefix_robust_dropout."})
    json_prefix_key_perturb: bool = field(default=True, metadata={"help": "Perturb JSON field-key context tokens during bridge-conditioned denoising."})
    json_prefix_key_dropout: float = field(default=0.08, metadata={"help": "Dropout probability for JSON field-key context perturbation."})
    json_prefix_event_start_perturb: bool = field(default=True, metadata={"help": "Perturb event-object expansion contexts near JSON root and event starts."})
    json_prefix_event_start_dropout: float = field(default=0.05, metadata={"help": "Dropout probability for event-object expansion context perturbation."})
    json_perturb_partition: bool = field(default=True, metadata={"help": "Keep start-continuation, event-start, key, and value perturbations from overlapping on the same decoder positions."})
    json_prefix_robust_dropout: float = field(default=0.20, metadata={"help": "Backward-compatible value-prefix dropout probability for bridge-conditioned JSON denoising."})
    json_prefix_robust_warmup_ratio: float = field(default=0.10, metadata={"help": "Fraction of training used to warm up json_prefix_robust_dropout."})
    json_start_continuation_enable: bool = field(default=True, metadata={"help": "Enable bridge-warmed start-continuation training from the JSON event-array prefix."})
    json_start_continuation_prob: float = field(default=0.12, metadata={"help": "Sample probability for start-continuation embedding replacement after warmup."})
    json_start_continuation_max_tokens: int = field(default=96, metadata={"help": "Maximum number of tokens replaced after the JSON event-array start prefix."})
    json_start_continuation_warmup_ratio: float = field(default=0.30, metadata={"help": "Training progress ratio before start-continuation replacement can be enabled."})
    json_start_continuation_min_bridge_gate: float = field(default=0.60, metadata={"help": "Minimum sample-level bridge gate required before start-continuation replacement is used."})
    lm_head_chunk_size: int = field(default=16, metadata={"help": "Decoder-token chunk size for memory-efficient LM-head CE computation."})
    process_distill_loss_factor: float = field(default=0.4, metadata={"help": "Weight for process-guided latent distillation."})
    process_distill_normalize_hidden: bool = field(default=True, metadata={"help": "L2-normalize hidden states before process-guided latent distillation to avoid scale-dominated losses."})
    process_alignment: str = field(default="soft_linear", metadata={"help": "Process distillation alignment: hard or soft_linear."})
    process_soft_align_factor: float = field(default=1.0, metadata={"help": "Blend factor for soft_linear process-stage targets; 0 keeps hard targets and 1 uses full soft targets."})
    process_soft_min_steps: int = field(default=4, metadata={"help": "Minimum latent target steps required before soft_linear process alignment is used."})
    adaptive_bridge_gate: bool = field(default=True, metadata={"help": "Use a sample-level scalar gate for decoder-bridge residual contribution."})
    bridge_gate_init: float = field(default=0.9, metadata={"help": "Initial value for the sample-level decoder-bridge contribution gate."})
    budget_conditioned_handle_count: bool = field(default=True, metadata={"help": "Use predicted latent-budget distribution as a soft condition for event-handle existence logits."})
    budget_handle_count_factor: float = field(default=0.20, metadata={"help": "Strength of budget-conditioned correction applied to event-handle existence logits."})
    auxiliary_loss_gate_threshold: float = field(default=1.0, metadata={"help": "CE threshold for the single auxiliary-loss gate."})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    use_chat_template: bool = field(default=False, metadata={"help": "Wrap prompts in the backbone tokenizer's native chat template before latent reasoning."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    groups = {
        "lora": 0,
        "apler": 0,
        "embedding_or_lm_head": 0,
        "other": 0,
    }
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"] += param.numel()
        elif any(key in name for key in ("embed", "wte", "lm_head")):
            groups["embedding_or_lm_head"] += param.numel()
        elif any(
            name.startswith(prefix)
            for prefix in (
                "prj.",
                "latent_budget_",
                "event_",
                "role_",
                "evidence_",
                "handle_latent_",
                "decoder_bridge_",
                "document_summary_",
                "latent_boundary_",
            )
        ):
            groups["apler"] += param.numel()
        else:
            groups["other"] += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    print(
        "trainable breakdown: "
        + " | ".join(
            f"{name}={count} ({100 * count / max(trainable_parameters, 1):.2f}%)"
            for name, count in groups.items()
            if count
        )
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def cast_trainable_parameters_to_fp32(model):
    for _, param in model.named_parameters():
        if param.requires_grad and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()


def select_decoder_exit_steps(
    target_exit_steps,
    budget_logits,
    *,
    training,
    mix_enable,
    mix_prob,
    mix_warmup_ratio,
    mix_min_step,
    step_ratio,
    num_latent,
    random_values=None,
):
    """Choose the latent step used by the existing JSON CE.

    This does not add another generation loss. It only moves the single decoder
    supervision exit from the gold budget to a warmup-gated mixture of the gold
    budget and the model-predicted budget, matching the inference path better.
    """
    target_steps = target_exit_steps.long().clamp(
        min(MIN_LATENT_BUDGET, max(1, int(num_latent))),
        max(1, int(num_latent)),
    )
    mix_mask = torch.zeros_like(target_steps, dtype=torch.bool)
    if (
        not bool(training)
        or not bool(mix_enable)
        or budget_logits is None
        or target_steps.numel() == 0
    ):
        return target_steps, mix_mask

    prob = float(max(0.0, min(1.0, mix_prob)))
    if prob <= 0.0:
        return target_steps, mix_mask

    warmup = float(max(0.0, mix_warmup_ratio))
    if step_ratio is None or float(step_ratio) < warmup:
        return target_steps, mix_mask

    predicted_steps = (
        latent_budget_statistics(budget_logits.detach())["proposal"]
        .to(device=target_steps.device)
        .long()
    ).clamp(min(MIN_LATENT_BUDGET, max(1, int(num_latent))), max(1, int(num_latent)))
    if predicted_steps.shape != target_steps.shape:
        return target_steps, mix_mask

    min_step = max(1, int(mix_min_step))
    eligible = predicted_steps.ge(min_step)
    if random_values is None:
        random_values = torch.rand(target_steps.shape, device=target_steps.device)
    else:
        random_values = random_values.to(device=target_steps.device).float()
        if random_values.shape != target_steps.shape:
            random_values = random_values.reshape(target_steps.shape)
    mix_mask = eligible & random_values.lt(prob)
    return torch.where(mix_mask, predicted_steps, target_steps), mix_mask


def mask_invalid_latent_budget_logits(
    budget_logits,
    min_step=MIN_LATENT_BUDGET,
):
    """Mask budget classes that cannot be training targets or inference exits."""
    if budget_logits is None or budget_logits.size(-1) <= 1:
        return budget_logits
    first_valid = min(max(1, int(min_step)), int(budget_logits.size(-1))) - 1
    if first_valid <= 0:
        return budget_logits
    masked = budget_logits.clone()
    masked[..., :first_valid] = float("-inf")
    return masked


def latent_budget_statistics(
    budget_logits,
    min_step=MIN_LATENT_BUDGET,
):
    """Return one shared ordered interpretation of latent-budget logits."""

    if budget_logits is None or budget_logits.dim() < 2:
        raise ValueError("Latent-budget logits must have shape [...,T].")
    masked_logits = mask_invalid_latent_budget_logits(
        budget_logits.float(),
        min_step=min_step,
    )
    probabilities = torch.softmax(masked_logits, dim=-1)
    step_values = torch.arange(
        1,
        masked_logits.size(-1) + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    expected = (probabilities * step_values).sum(dim=-1)
    proposal = torch.floor(expected + 0.5).long().clamp(
        min=min(max(1, int(min_step)), masked_logits.size(-1)),
        max=masked_logits.size(-1),
    )
    top1 = masked_logits.argmax(dim=-1).long() + 1
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    if probabilities.size(-1) > 1:
        top_probabilities = probabilities.topk(k=2, dim=-1).values
        top1_margin = top_probabilities[..., 0] - top_probabilities[..., 1]
    else:
        top1_margin = torch.ones_like(expected)
    return {
        "logits": masked_logits,
        "probabilities": probabilities,
        "expected": expected,
        "proposal": proposal,
        "top1": top1,
        "entropy": entropy,
        "top1_margin": top1_margin,
    }


def exclude_sampled_json_perturbations(candidate_mask, *earlier_drop_masks, enabled=True):
    """Exclude only positions actually sampled by higher-priority perturbations."""
    if candidate_mask is None or not bool(enabled):
        return candidate_mask
    filtered = candidate_mask
    for drop_mask in earlier_drop_masks:
        if drop_mask is not None:
            filtered = filtered & ~drop_mask.to(device=filtered.device).bool()
    return filtered


class CODI(torch.nn.Module):
    # Hugging Face Trainer accesses these PreTrainedModel compatibility
    # attributes while reporting checkpoint load warnings. CODI is an outer
    # nn.Module wrapper, so declare the neutral defaults explicitly.
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        validate_event_configuration(
            training_args.event_type_count,
            training_args.role_type_count,
            training_args.max_event_count,
            training_args.max_sentence_markers,
        )
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        if model_args.train and bool(training_args.gradient_checkpointing):
            raise ValueError(
                "gradient_checkpointing is incompatible with CODI/APLER's differentiable "
                "KV-cache latent rollout. Use gradient accumulation to reduce memory instead."
            )
        if str(getattr(model_args, "separate_decoder_name", "") or "").strip():
            raise NotImplementedError(
                "separate_decoder_name is not implemented. CODI/APLER currently shares one "
                "causal LM across document encoding, latent reasoning, and JSON decoding."
            )
        model_wrapper_class = AutoModelForCausalLM 
        if training_args.bf16:
            requested_dtype = torch.bfloat16
        elif training_args.fp16:
            requested_dtype = torch.float16
        else:
            requested_dtype = torch.float32
        # Keep the frozen backbone in the requested mixed-precision dtype.  Only
        # trainable LoRA/APLER parameters are cast to fp32 below; otherwise Qwen
        # 3B wastes several GB by storing frozen base weights in fp32.
        model_dtype = requested_dtype
        head_dtype = torch.float32 if model_args.train and (training_args.fp16 or training_args.bf16) else model_dtype
        self.model_dtype = model_dtype
        model_load_kwargs = {
            "torch_dtype": model_dtype,
            "token": getattr(model_args, "token", None),
            "cache_dir": getattr(training_args, "cache_dir", None),
        }
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                **model_load_kwargs,
            )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                self.model_name,
                **model_load_kwargs,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=requested_dtype,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4',
                )
            )


        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        try:
            self.codi.resize_token_embeddings(
                ori_vocab_size + 3,
                mean_resizing=False,
            )  # dummy values for mem tokens
        except TypeError:
            self.codi.resize_token_embeddings(
                ori_vocab_size + 3
            )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = load_tokenizer_with_fallback(
            self.model_name,
            token=getattr(model_args, "token", None),
            cache_dir=getattr(training_args, "cache_dir", None),
            model_max_length=getattr(training_args, "model_max_length", None),
        )

        if not model_args.full_precision and model_args.train:
            self.codi = prepare_model_for_kbit_training(
                self.codi,
                use_gradient_checkpointing=False,
            )

        # LoRA
        if training_args.use_lora:
            adapter_path = str(getattr(model_args, "adapter_name_or_path", "") or "").strip()
            if lora_config is not None and adapter_path:
                raise ValueError(
                    "--lora_init True and --adapter_name_or_path are mutually exclusive."
                )
            if lora_config is not None:
                self.codi = get_peft_model(self.codi, lora_config)
            elif adapter_path:
                self.codi = PeftModel.from_pretrained(
                    self.codi,
                    adapter_path,
                    is_trainable=bool(self.training),
                )
            else:
                raise ValueError(
                    "use_lora=True requires either --lora_init True or "
                    "--adapter_name_or_path PATH."
                )

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            self.prj.to(dtype=head_dtype)
                
        # Losses
        self.adaptive_latent = training_args.adaptive_latent
        self.latent_budget_loss_factor = training_args.latent_budget_loss_factor
        self.latent_budget_ordinal_loss_factor = (
            training_args.latent_budget_ordinal_loss_factor
        )
        self.latent_budget_condition_factor = training_args.latent_budget_condition_factor
        self.latent_budget_condition_teacher_forcing = training_args.latent_budget_condition_teacher_forcing
        self.decoder_exit_mix_enable = bool(training_args.decoder_exit_mix_enable)
        self.decoder_exit_mix_prob = float(training_args.decoder_exit_mix_prob)
        self.decoder_exit_mix_warmup_ratio = float(training_args.decoder_exit_mix_warmup_ratio)
        self.decoder_exit_mix_min_step = int(training_args.decoder_exit_mix_min_step)
        self.max_event_count = training_args.max_event_count
        self.event_handle_condition_factor = training_args.event_handle_condition_factor
        self.event_handle_condition_teacher_forcing = training_args.event_handle_condition_teacher_forcing
        self.event_handle_teacher_forcing_decay_ratio = training_args.event_handle_teacher_forcing_decay_ratio
        self.native_boundary_loss_enable = bool(
            training_args.native_boundary_loss_enable
        )
        self.native_boundary_loss_factor = float(
            training_args.native_boundary_loss_factor
        )
        self.event_handle_anchor_loss_factor = training_args.event_handle_anchor_loss_factor
        self.event_handle_exist_loss_factor = training_args.event_handle_exist_loss_factor
        self.event_handle_exist_count_loss_factor = training_args.event_handle_exist_count_loss_factor
        self.event_handle_match_anchor_cost = training_args.event_handle_match_anchor_cost
        self.event_handle_latent_attention_factor = training_args.event_handle_latent_attention_factor
        self.decoder_bridge_factor = training_args.decoder_bridge_factor
        self.document_summary_fusion_factor = training_args.document_summary_fusion_factor
        self.complex_sample_weight_mode = str(
            getattr(training_args, "complex_sample_weight_mode", "ce_only") or "ce_only"
        ).lower()
        self.sample_loss_weight_min = training_args.sample_loss_weight_min
        self.sample_loss_weight_max = training_args.sample_loss_weight_max
        self.event_type_count = training_args.event_type_count
        self.role_type_count = training_args.role_type_count
        self.process_distill_loss_factor = training_args.process_distill_loss_factor
        self.process_distill_normalize_hidden = training_args.process_distill_normalize_hidden
        self.process_alignment = str(training_args.process_alignment or "hard").lower()
        self.process_soft_align_factor = float(training_args.process_soft_align_factor)
        self.process_soft_min_steps = max(1, int(training_args.process_soft_min_steps))
        self.adaptive_bridge_gate = bool(training_args.adaptive_bridge_gate)
        self.bridge_gate_init = float(training_args.bridge_gate_init)
        self.budget_conditioned_handle_count = bool(training_args.budget_conditioned_handle_count)
        self.budget_handle_count_factor = float(training_args.budget_handle_count_factor)
        self.auxiliary_loss_gate_threshold = training_args.auxiliary_loss_gate_threshold
        if self.adaptive_latent:
            self.latent_budget_head = nn.Linear(self.dim, self.num_latent).to(dtype=head_dtype)
            self.latent_budget_embedding = nn.Embedding(self.num_latent + 1, self.dim).to(dtype=head_dtype)
            self._use_latent_budget_head = True
        if self.event_handle_condition_factor > 0:
            self.event_handle_query = nn.Embedding(self.max_event_count, self.dim).to(dtype=head_dtype)
            self.event_handle_embedding = nn.Embedding(self.max_event_count, self.dim).to(dtype=head_dtype)
            self.event_handle_active_embedding = nn.Embedding(2, self.dim).to(dtype=head_dtype)
            self.event_handle_anchor_proj = nn.Linear(self.dim, self.dim).to(dtype=head_dtype)
            self.event_handle_query_norm = nn.LayerNorm(self.dim).to(dtype=head_dtype)
            self.event_handle_condition_proj = nn.Linear(self.dim, self.dim).to(dtype=head_dtype)
            self.event_handle_exist_head = nn.Linear(self.dim, 1).to(dtype=head_dtype)
            if self.budget_conditioned_handle_count:
                self.event_handle_budget_handle_proj = nn.Linear(self.dim, self.dim, bias=False).to(dtype=head_dtype)
                self.event_handle_budget_budget_proj = nn.Linear(self.dim, self.dim, bias=False).to(dtype=head_dtype)
                self.event_handle_budget_out = nn.Linear(self.dim, 1).to(dtype=head_dtype)
            with torch.no_grad():
                # Most documents use only a small fraction of the available
                # instance carriers, so start from a sparse existence prior.
                self.event_handle_exist_head.bias.fill_(-2.0)
        if self.event_handle_latent_attention_factor > 0:
            self.handle_latent_query_proj = nn.Linear(self.dim, self.dim, bias=False).to(dtype=head_dtype)
            self.handle_latent_key_proj = nn.Linear(self.dim, self.dim, bias=False).to(dtype=head_dtype)
            self.handle_latent_value_proj = nn.Linear(self.dim, self.dim, bias=False).to(dtype=head_dtype)
            self.handle_latent_gate = nn.Linear(self.dim * 2, self.dim).to(dtype=head_dtype)
        if self.decoder_bridge_factor > 0:
            self.decoder_bridge_proj = nn.Linear(self.dim, self.dim).to(dtype=head_dtype)
            self.decoder_bridge_gate = nn.Linear(self.dim * 2, self.dim).to(dtype=head_dtype)
            if self.adaptive_bridge_gate:
                self.decoder_bridge_contribution_gate = nn.Linear(self.dim * 2, 1).to(dtype=head_dtype)
                init_value = min(max(float(self.bridge_gate_init), 1e-4), 1.0 - 1e-4)
                init_logit = math.log(init_value / (1.0 - init_value))
                with torch.no_grad():
                    self.decoder_bridge_contribution_gate.weight.zero_()
                    self.decoder_bridge_contribution_gate.bias.fill_(init_logit)
        if self.document_summary_fusion_factor > 0:
            self.document_summary_gate = nn.Linear(self.dim * 2, self.dim).to(dtype=head_dtype)
        self.latent_boundary_embedding = nn.Embedding(2, self.dim).to(dtype=head_dtype)
        with torch.no_grad():
            base_embedding = self.get_embd(self.codi, self.model_name)
            fallback_id = self.tokenizer.eos_token_id
            if fallback_id is None:
                fallback_id = self.tokenizer.bos_token_id
            if fallback_id is None:
                fallback_id = 0
            fallback = base_embedding.weight[int(fallback_id)].detach().to(
                dtype=self.latent_boundary_embedding.weight.dtype
            )
            self.latent_boundary_embedding.weight[0].copy_(fallback)
            self.latent_boundary_embedding.weight[1].copy_(fallback)
        self.multilabel_loss_fct = nn.BCEWithLogitsLoss()

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        
        # Process self-distillation
        self.distill_layer_strategy = str(training_args.distill_layer_strategy).lower()
        self.distill_selected_layer_count = max(1, int(training_args.distill_selected_layer_count))

        # general 
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training and (training_args.fp16 or training_args.bf16):
            cast_trainable_parameters_to_fp32(self)

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError(
                f"Cannot locate input embeddings for backbone {model_name!r}."
            )

    def get_final_hidden_module(self):
        try:
            base_model = self.codi.get_base_model()
        except Exception:
            base_model = self.codi

        candidates = []
        if hasattr(base_model, "model"):
            candidates.append(getattr(base_model, "model"))
        if hasattr(base_model, "transformer"):
            candidates.append(getattr(base_model, "transformer"))
        if hasattr(base_model, "gpt_neox"):
            candidates.append(getattr(base_model, "gpt_neox"))
        candidates.append(base_model)

        for module in candidates:
            for attr in ("norm", "ln_f", "final_layer_norm"):
                if hasattr(module, attr):
                    return getattr(module, attr)
        return None

    def get_transformer_layer_modules(self):
        try:
            base_model = self.codi.get_base_model()
        except Exception:
            base_model = self.codi

        candidates = []
        if hasattr(base_model, "model"):
            candidates.append(getattr(base_model, "model"))
        if hasattr(base_model, "transformer"):
            candidates.append(getattr(base_model, "transformer"))
        if hasattr(base_model, "gpt_neox"):
            candidates.append(getattr(base_model, "gpt_neox"))
        candidates.append(base_model)

        for module in candidates:
            for attr in ("layers", "h"):
                layers = getattr(module, attr, None)
                if layers is not None and len(layers) > 0:
                    return list(layers)
        return []

    def get_causal_lm_base(self):
        try:
            return self.codi.get_base_model()
        except Exception:
            return self.codi

    def get_backbone_model(self):
        """Return the transformer backbone without the LM head when available.

        Training APLER repeatedly encodes long documents and one-token latent
        steps.  Those passes only need hidden states and KV cache; computing
        full vocabulary logits for Qwen's large vocab wastes several GB on
        4k-token inputs.  Calling the backbone keeps LoRA-injected modules in
        place while avoiding the unnecessary lm_head projection.
        """
        base_model = self.get_causal_lm_base()
        for attr in ("model", "transformer", "gpt_neox"):
            backbone = getattr(base_model, attr, None)
            if backbone is not None:
                return backbone
        return None

    def backbone_forward(self, **kwargs):
        backbone = self.get_backbone_model()
        kwargs.setdefault("return_dict", True)
        if backbone is None:
            return self.codi(**kwargs)
        return backbone(**kwargs)

    def get_lm_head(self):
        base_model = self.get_causal_lm_base()
        for attr in ("lm_head", "embed_out", "score"):
            head = getattr(base_model, attr, None)
            if head is not None:
                return head
        raise AttributeError(f"Cannot find LM head for {self.model_name!r}.")

    def lm_head_logits(self, hidden_states):
        lm_head = self.get_lm_head()
        return lm_head(hidden_states.to(dtype=lm_head.weight.dtype))

    def get_distill_hidden_modules(self):
        strategy = str(getattr(self, "distill_layer_strategy", "full")).lower()
        final_module = self.get_final_hidden_module()
        if strategy == "last":
            return [final_module] if final_module is not None else []
        if strategy != "selected":
            return []

        layers = self.get_transformer_layer_modules()
        modules = []
        if layers:
            layer_count = len(layers)
            selected_count = min(max(1, self.distill_selected_layer_count - 1), layer_count)
            if selected_count == 1:
                indices = [layer_count - 1]
            else:
                indices = sorted(
                    {
                        round(i * (layer_count - 1) / (selected_count - 1))
                        for i in range(selected_count)
                    }
                )
            modules.extend(layers[int(index)] for index in indices)
        if final_module is not None and final_module not in modules:
            modules.append(final_module)
        return modules

    def uses_compact_distill_layers(self):
        return str(getattr(self, "distill_layer_strategy", "full")).lower() in {"selected", "last"}

    def pool_hidden_ranges(self, hidden, ranges):
        """Mean-pool hidden states over [start, end) stage ranges."""
        if ranges is None:
            return hidden
        ranges = ranges.to(hidden.device).long()
        if ranges.dim() != 3 or ranges.size(-1) != 2:
            raise ValueError("Process self-distillation ranges must have shape [batch, stages, 2].")
        seq_len = hidden.size(1)
        if seq_len <= 0:
            raise ValueError("Cannot pool process stages from an empty hidden sequence.")
        starts = ranges[..., 0].clamp(0, seq_len - 1)
        ends = ranges[..., 1].clamp(0, seq_len)
        ends = torch.maximum(ends, (starts + 1).clamp(max=seq_len))
        prefix = torch.cat(
            (
                torch.zeros(
                    hidden.size(0),
                    1,
                    hidden.size(-1),
                    dtype=torch.float32,
                    device=hidden.device,
                ),
                hidden.float().cumsum(dim=1),
            ),
            dim=1,
        )
        gather_shape = starts.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        start_sum = prefix.gather(1, gather_shape)
        end_sum = prefix.gather(1, ends.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        lengths = (ends - starts).clamp_min(1).unsqueeze(-1).float()
        return ((end_sum - start_sum) / lengths).to(dtype=hidden.dtype)

    def codi_with_distill_hiddens(self, gather_positions=None, gather_ranges=None, return_logits=True, **kwargs):
        """Run the LM while capturing only selected distillation states.

        Hugging Face returns every layer when output_hidden_states=True.  For
        long-document distillation this is expensive.  APLER captures only
        representative layers and, when possible, only supervised positions.
        """
        hook_modules = self.get_distill_hidden_modules()
        captured = [None] * len(hook_modules)
        handles = []

        def capture_hidden(index):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if gather_ranges is not None:
                    hidden = self.pool_hidden_ranges(hidden, gather_ranges)
                elif gather_positions is not None:
                    positions = gather_positions.to(hidden.device).long()
                    if positions.dim() == 1:
                        positions = positions.unsqueeze(1)
                    positions = positions.clamp(0, hidden.size(1) - 1)
                    hidden = hidden.gather(
                        1,
                        positions.unsqueeze(-1).expand(
                            positions.size(0),
                            positions.size(1),
                            hidden.size(-1),
                        ),
                    )
                captured[index] = hidden
            return hook

        for index, module in enumerate(hook_modules):
            handles.append(module.register_forward_hook(capture_hidden(index)))
        try:
            if return_logits:
                outputs = self.codi(output_hidden_states=False, **kwargs)
            else:
                outputs = self.backbone_forward(output_hidden_states=False, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        hidden_states = tuple(hidden for hidden in captured if hidden is not None)
        if not hidden_states:
            if return_logits:
                outputs = self.codi(output_hidden_states=True, **kwargs)
            else:
                outputs = self.backbone_forward(output_hidden_states=True, **kwargs)
            hidden_states = outputs.hidden_states
        return outputs, hidden_states

    def embed_input_ids(self, input_ids):
        embeddings = self.get_embd(self.codi, self.model_name)(input_ids)
        if not hasattr(self, "latent_boundary_embedding"):
            return embeddings
        bot_mask = input_ids.eq(self.bot_id).unsqueeze(-1)
        eot_mask = input_ids.eq(self.eot_id).unsqueeze(-1)
        bot_embedding = self.latent_boundary_embedding.weight[0].to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        eot_embedding = self.latent_boundary_embedding.weight[1].to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        embeddings = torch.where(bot_mask, bot_embedding, embeddings)
        return torch.where(eot_mask, eot_embedding, embeddings)

    def fuse_document_summary(
        self,
        encoder_states,
        encoder_summary,
        attention_mask=None,
        document_token_mask=None,
    ):
        """Blend the last-token summary with a pooled document-token summary.

        Chat backbones place the assistant-generation marker at the end of the
        prompt.  Using only that final token is brittle for long documents, so
        event planning heads and latent reasoning get a gated view of the
        actual S0/S1/... document region as well.
        """
        if (
            not hasattr(self, "document_summary_gate")
            or float(getattr(self, "document_summary_fusion_factor", 0.0)) <= 0
            or encoder_states is None
            or encoder_summary is None
        ):
            return encoder_summary

        if attention_mask is not None:
            valid_mask = attention_mask.to(encoder_states.device).bool()
        else:
            valid_mask = torch.ones(
                encoder_states.shape[:2],
                dtype=torch.bool,
                device=encoder_states.device,
            )
        if document_token_mask is not None:
            doc_mask = document_token_mask.to(encoder_states.device).bool()
            if doc_mask.dim() == 1:
                doc_mask = doc_mask.unsqueeze(0)
            doc_mask = doc_mask[:, : encoder_states.size(1)]
            if doc_mask.size(1) < encoder_states.size(1):
                pad = torch.zeros(
                    (doc_mask.size(0), encoder_states.size(1) - doc_mask.size(1)),
                    dtype=torch.bool,
                    device=encoder_states.device,
                )
                doc_mask = torch.cat((doc_mask, pad), dim=1)
            valid_mask = valid_mask & doc_mask

        fallback_mask = (
            attention_mask.to(encoder_states.device).bool()
            if attention_mask is not None
            else torch.ones_like(valid_mask)
        )
        has_document_tokens = valid_mask.any(dim=1, keepdim=True)
        valid_mask = torch.where(has_document_tokens, valid_mask, fallback_mask)

        weights = valid_mask.float()
        pooled = (encoder_states.float() * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary_float = encoder_summary.float()
        gate_input = torch.cat((summary_float, pooled), dim=-1).to(
            dtype=self.document_summary_gate.weight.dtype,
            device=encoder_summary.device,
        )
        gate = torch.sigmoid(self.document_summary_gate(gate_input)).float()
        gated = gate * summary_float + (1.0 - gate) * pooled
        factor = max(0.0, min(1.0, float(self.document_summary_fusion_factor)))
        fused = (1.0 - factor) * summary_float + factor * gated
        return fused.to(dtype=encoder_summary.dtype, device=encoder_summary.device)

    def normalize_past_key_values(self, past_key_values):
        if past_key_values is None or DynamicCache is None:
            return past_key_values
        # DynamicCache is the supported custom-generation path. During
        # training, keep the differentiable legacy tensors used by CODI's
        # sequential latent rollout instead of mutating an inference cache.
        if self.training:
            return past_key_values
        if not any(name in self.model_name.lower() for name in ("qwen", "llama", "mistral")):
            return past_key_values
        if Cache is not None and isinstance(past_key_values, Cache):
            return past_key_values
        try:
            return DynamicCache.from_legacy_cache(past_key_values)
        except (AttributeError, TypeError, ValueError):
            return past_key_values

    @staticmethod
    def clone_past_key_values(past_key_values):
        """Fork a rollout cache before an auxiliary decoder mutates its container."""
        if past_key_values is None:
            return None

        def clone_tree(value):
            if torch.is_tensor(value):
                return value.clone()
            if isinstance(value, tuple):
                return tuple(clone_tree(item) for item in value)
            if isinstance(value, list):
                return [clone_tree(item) for item in value]
            return value

        if Cache is not None and isinstance(past_key_values, Cache):
            to_legacy_cache = getattr(past_key_values, "to_legacy_cache", None)
            if callable(to_legacy_cache):
                legacy_cache = to_legacy_cache()
                from_legacy_cache = getattr(type(past_key_values), "from_legacy_cache", None)
                if callable(from_legacy_cache):
                    return from_legacy_cache(legacy_cache)
                if DynamicCache is not None:
                    return DynamicCache.from_legacy_cache(legacy_cache)
            raise TypeError(
                f"Cannot safely clone cache type {type(past_key_values).__name__} for auxiliary decoding."
            )
        return clone_tree(past_key_values)

    @staticmethod
    def past_key_values_length(past_key_values):
        if past_key_values is None:
            return 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            return int(get_seq_length())
        try:
            return int(past_key_values[0][0].size(-2))
        except (IndexError, TypeError, AttributeError):
            return None

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            load_result = load_checkpoint_into_model(
                self,
                self.training_args.restore_from,
                require_lora=bool(
                    self.training_args.use_lora
                    and not self.model_args.adapter_name_or_path
                ),
                require_current_modules=True,
            )
            missing_keys = list(getattr(load_result, "missing_keys", []))
            unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
            if missing_keys:
                print(f"Base-model keys loaded from model_name_or_path, not compact checkpoint: {len(missing_keys)} missing keys")
            if unexpected_keys:
                print(f"Unexpected checkpoint keys ignored: {unexpected_keys}")
            print(f"Finished loading from {self.training_args.restore_from}")

    def latent_stage_targets(self, step_index, target_steps, stage_count, batch_size, device):
        # target_steps is the document-specific latent budget. stage_count is
        # the fixed teacher process granularity. This maps a variable-length
        # latent path onto the ordered process stages without making T fixed.
        if target_steps is None:
            target_steps = torch.full(
                (batch_size,),
                self.num_latent,
                dtype=torch.long,
                device=device,
            )
        target_steps = target_steps.to(device).long().clamp(1, max(1, self.num_latent))
        stage_targets = torch.full(
            (batch_size,),
            max(stage_count - 1, 0),
            dtype=torch.long,
            device=device,
        )
        for batch_index, step_count_tensor in enumerate(target_steps):
            step_count = int(step_count_tensor.item())
            if stage_count == 5 and step_count == 1:
                schedule = [4]
            elif stage_count == 5 and step_count == 2:
                schedule = [1, 4]
            elif stage_count == 5 and step_count == 3:
                schedule = [1, 3, 4]
            elif stage_count == 5 and step_count == 4:
                schedule = [1, 2, 3, 4]
            else:
                schedule = torch.linspace(
                    0,
                    max(stage_count - 1, 0),
                    steps=step_count,
                    device=device,
                ).round().long().tolist()
            stage_targets[batch_index] = int(
                schedule[min(step_index, len(schedule) - 1)]
            )
        return stage_targets

    def select_process_teacher_hidden(
        self,
        ref_hidden,
        step_index,
        target_steps,
        stage_count,
        batch_size,
        device,
    ):
        """Select the teacher process target for one latent step.

        hard keeps the established discrete stage schedule. soft_linear is a
        conservative softening of that schedule: short latent paths keep the
        old hard targets, while longer paths interpolate from stage 1 to the
        final stage so stage 0 does not consume scarce latent capacity.
        """
        hard_indices = self.latent_stage_targets(
            step_index,
            target_steps,
            stage_count,
            batch_size,
            device,
        ).clamp(0, max(stage_count - 1, 0))
        hard_selected = ref_hidden.gather(
            1,
            hard_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, ref_hidden.size(-1)),
        )
        align_mode = str(getattr(self, "process_alignment", "hard") or "hard").lower()
        soft_factor = float(getattr(self, "process_soft_align_factor", 0.0))
        if align_mode != "soft_linear" or soft_factor <= 0.0 or stage_count <= 1:
            return hard_selected

        if target_steps is None:
            target_steps = torch.full(
                (batch_size,),
                self.num_latent,
                dtype=torch.long,
                device=device,
            )
        target_steps = target_steps.to(device).long().clamp(1, max(1, self.num_latent))
        soft_min_steps = max(1, int(getattr(self, "process_soft_min_steps", 4)))
        use_soft = target_steps >= soft_min_steps
        if not bool(use_soft.any()):
            return hard_selected

        denom = (target_steps.float() - 1.0).clamp_min(1.0)
        progress = (float(step_index) / denom).clamp(0.0, 1.0)
        start_stage = 1.0 if stage_count > 1 else 0.0
        end_stage = float(stage_count - 1)
        stage_position = start_stage + progress * (end_stage - start_stage)
        lower_indices = stage_position.floor().long().clamp(0, stage_count - 1)
        upper_indices = stage_position.ceil().long().clamp(0, stage_count - 1)
        upper_weight = (stage_position - lower_indices.float()).clamp(0.0, 1.0)
        lower_selected = ref_hidden.gather(
            1,
            lower_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, ref_hidden.size(-1)),
        )
        upper_selected = ref_hidden.gather(
            1,
            upper_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, ref_hidden.size(-1)),
        )
        soft_selected = (
            (1.0 - upper_weight).view(-1, 1, 1) * lower_selected
            + upper_weight.view(-1, 1, 1) * upper_selected
        )
        factor = max(0.0, min(1.0, soft_factor))
        if factor >= 1.0:
            blended = soft_selected
        else:
            blended = (1.0 - factor) * hard_selected + factor * soft_selected
        return torch.where(use_soft.view(-1, 1, 1), blended, hard_selected)

    def build_event_handle_plan(
        self,
        sentence_hidden,
        valid_sentences,
        gold_handle_anchor_targets=None,
        gold_counts=None,
        teacher_forcing_ratio=0.0,
        global_context=None,
        budget_condition=None,
    ):
        """Build lightweight, permutation-invariant event handles.

        These handles carry only existence/count and evidence-anchor signals.
        Event content is still generated by the LLM decoder, so this module
        remains a soft instance scaffold instead of a field-filling extractor.
        """
        if not hasattr(self, "event_handle_embedding"):
            return None

        batch_size, sentence_count, hidden_size = sentence_hidden.shape
        max_handles = int(self.max_event_count)
        device = sentence_hidden.device
        valid_sentence_rows = valid_sentences.any(dim=1)
        handle_ids = torch.arange(max_handles, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        static_queries = F.normalize(
            self.event_handle_query.weight.float(),
            dim=-1,
            eps=1e-6,
        )
        projected_markers = F.normalize(
            self.event_handle_anchor_proj(
                sentence_hidden.to(dtype=self.event_handle_anchor_proj.weight.dtype)
            ).float(),
            dim=-1,
            eps=1e-6,
        )

        def gather_anchor_vectors(indices):
            safe_indices = indices.clamp_min(0)
            gathered = sentence_hidden.float().gather(
                1,
                safe_indices.unsqueeze(-1).expand(-1, -1, hidden_size),
            )
            return gathered * indices.ge(0).unsqueeze(-1).float()

        query_seed = static_queries.unsqueeze(0).expand(batch_size, -1, -1)
        if global_context is not None:
            context_seed = global_context
            if context_seed.dim() == 3:
                context_seed = context_seed[:, -1, :]
            context_direction = F.normalize(
                context_seed.to(device=device).float(),
                dim=-1,
                eps=1e-6,
            )
            query_seed = query_seed + context_direction.unsqueeze(1)
        initial_queries, query_seed_norm = stabilize_handle_query_seed(query_seed)
        query_seed_guard_rate = (
            query_seed_norm.squeeze(-1)
            .lt(HANDLE_QUERY_SEED_NORM_FLOOR)
            .float()
            .mean()
        )
        initial_attention_logits = torch.einsum(
            "bsd,bmd->bsm", initial_queries, projected_markers
        ) * math.sqrt(float(hidden_size))
        initial_attention_logits = initial_attention_logits.masked_fill(
            ~valid_sentences.unsqueeze(1), -1e4
        )
        initial_attention = torch.softmax(initial_attention_logits.float(), dim=-1)
        initial_attention = initial_attention * valid_sentence_rows[:, None, None].to(
            initial_attention.dtype
        )
        query_context = torch.matmul(initial_attention, sentence_hidden.float())
        handle_hidden = self.event_handle_query_norm(
            (query_seed + query_context).to(dtype=self.event_handle_query_norm.weight.dtype)
        ).float()
        refined_queries = F.normalize(
            handle_hidden,
            dim=-1,
            eps=1e-6,
        )
        handle_anchor_logits = torch.einsum(
            "bmd,bsd->bsm", projected_markers, refined_queries
        ) * math.sqrt(float(hidden_size))
        handle_anchor_logits = handle_anchor_logits.masked_fill(
            ~valid_sentences.unsqueeze(1), -1e4
        )
        handle_exist_logits = self.event_handle_exist_head(
            handle_hidden.to(dtype=self.event_handle_exist_head.weight.dtype)
        ).squeeze(-1).float()
        if (
            bool(getattr(self, "budget_conditioned_handle_count", False))
            and float(getattr(self, "budget_handle_count_factor", 0.0)) > 0.0
            and budget_condition is not None
            and hasattr(self, "event_handle_budget_handle_proj")
            and hasattr(self, "event_handle_budget_budget_proj")
            and hasattr(self, "event_handle_budget_out")
        ):
            budget_context = budget_condition
            if budget_context.dim() == 3:
                budget_context = budget_context[:, -1, :]
            budget_context = budget_context.to(device=device).detach()
            budget_dtype = self.event_handle_budget_handle_proj.weight.dtype
            budget_hidden = torch.tanh(
                self.event_handle_budget_handle_proj(handle_hidden.to(dtype=budget_dtype))
                + self.event_handle_budget_budget_proj(
                    budget_context.to(dtype=budget_dtype)
                ).unsqueeze(1)
            )
            budget_delta = self.event_handle_budget_out(budget_hidden).squeeze(-1).float()
            handle_exist_logits = (
                handle_exist_logits
                + float(self.budget_handle_count_factor) * budget_delta
            )

        predicted_indices = handle_anchor_logits.argmax(dim=-1)

        matched_anchor_targets = torch.full(
            (batch_size, max_handles), -100, dtype=torch.long, device=device
        )
        matched_handles = torch.zeros(
            (batch_size, max_handles), dtype=torch.bool, device=device
        )
        if gold_handle_anchor_targets is not None and gold_counts is not None:
            gold_anchor_values = gold_handle_anchor_targets.to(device).long()
            anchor_log_probs = torch.log_softmax(handle_anchor_logits.float(), dim=-1)
            for batch_index in range(batch_size):
                gold_limit = min(max_handles, int(gold_counts[batch_index].item()))
                valid_gold = [
                    gold_index
                    for gold_index in range(gold_limit)
                    if (
                        0 <= int(gold_anchor_values[batch_index, gold_index].item()) < sentence_count
                        and bool(
                            valid_sentences[
                                batch_index,
                                int(gold_anchor_values[batch_index, gold_index].item()),
                            ].item()
                        )
                    )
                ]
                if not valid_gold:
                    continue
                gold_anchor_index = gold_anchor_values[batch_index, valid_gold]
                total_cost = -float(self.event_handle_match_anchor_cost) * (
                    anchor_log_probs[batch_index][:, gold_anchor_index]
                )
                for prediction_index, local_gold_index in minimum_cost_bipartite_assignment(total_cost):
                    original_gold_index = valid_gold[local_gold_index]
                    matched_handles[batch_index, prediction_index] = True
                    matched_anchor_targets[batch_index, prediction_index] = gold_anchor_values[
                        batch_index, original_gold_index
                    ]

        predicted_vectors = gather_anchor_vectors(predicted_indices)
        ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        conditioning_vectors = predicted_vectors
        if matched_handles.any():
            matched_vectors = gather_anchor_vectors(
                torch.where(
                    matched_handles,
                    matched_anchor_targets,
                    torch.full_like(matched_anchor_targets, -1),
                )
            )
            conditioning_vectors = torch.where(
                matched_handles.unsqueeze(-1),
                ratio * matched_vectors + (1.0 - ratio) * predicted_vectors,
                predicted_vectors,
            )
        conditioning_vectors = self.event_handle_query_norm(
            (handle_hidden + conditioning_vectors).to(dtype=self.event_handle_query_norm.weight.dtype)
        ).float()

        exist_probs = torch.sigmoid(handle_exist_logits)
        predicted_active = torch.zeros_like(exist_probs, dtype=torch.bool)
        predicted_counts = exist_probs.sum(dim=1).round().long().clamp(1, max_handles)
        predicted_counts = torch.where(
            valid_sentence_rows,
            predicted_counts,
            torch.zeros_like(predicted_counts),
        )
        for batch_index in range(batch_size):
            required_count = int(predicted_counts[batch_index].item())
            ranked_handles = exist_probs[batch_index].argsort(descending=True)
            predicted_active[batch_index, ranked_handles[:required_count]] = True
        plan_active = predicted_active
        use_gold_rows = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        if self.training and matched_handles.any() and ratio > 0:
            use_gold_rows = (
                (torch.rand((batch_size,), device=device) < ratio)
                & valid_sentence_rows
                & matched_handles.any(dim=1)
            )
            plan_active = torch.where(
                use_gold_rows.unsqueeze(1), matched_handles, predicted_active
            )

        projected_anchors = self.event_handle_condition_proj(
            conditioning_vectors.to(dtype=self.event_handle_condition_proj.weight.dtype)
        )
        active_mask = plan_active.unsqueeze(-1).to(projected_anchors.dtype)
        plan = (
            self.event_handle_embedding(handle_ids)
            + self.event_handle_active_embedding(plan_active.long())
            + float(self.event_handle_condition_factor) * projected_anchors * active_mask
        )

        # Pack active handles in document order for deterministic memory access.
        # Hungarian supervision remains permutation-invariant: handle_i is not
        # assumed to be the i-th event in canonical JSON order.
        ordering_indices = predicted_indices
        if matched_handles.any():
            gold_ordering_indices = torch.where(
                matched_handles & matched_anchor_targets.ge(0),
                matched_anchor_targets,
                predicted_indices,
            )
            ordering_indices = torch.where(
                use_gold_rows.unsqueeze(1), gold_ordering_indices, predicted_indices
            )
        safe_anchor_order = ordering_indices.clamp_min(0)
        chronological_score = (
            float(sentence_count) - safe_anchor_order.float()
        ) / float(max(sentence_count, 1))
        order_score = (
            plan_active.float() * float(sentence_count + 2)
            + chronological_score
            + 0.01 * exist_probs
        )
        handle_order = order_score.argsort(dim=1, descending=True)
        gather_hidden = handle_order.unsqueeze(-1).expand(-1, -1, hidden_size)
        packed_plan = plan.gather(1, gather_hidden)
        packed_active = plan_active.gather(1, handle_order)
        packed_indices = predicted_indices.gather(1, handle_order)
        packed_ordering_indices = ordering_indices.gather(1, handle_order)
        packed_exist_probs = exist_probs.gather(1, handle_order)
        active_plan_length = max(
            1, int(packed_active.long().sum(dim=1).max().detach().item())
        )

        return {
            "plan": packed_plan[:, :active_plan_length],
            "predicted_indices": packed_indices[:, :active_plan_length],
            "ordering_anchor_indices": packed_ordering_indices[:, :active_plan_length],
            "handle_order": handle_order[:, :active_plan_length],
            "raw_predicted_indices": predicted_indices,
            "plan_active": packed_active[:, :active_plan_length],
            "packed_exist_probs": packed_exist_probs[:, :active_plan_length],
            "handle_exist_logits": handle_exist_logits,
            "handle_exist_probs": exist_probs,
            "predicted_active": predicted_active,
            "dynamic_counts": predicted_active.long().sum(dim=1),
            "handle_anchor_logits": handle_anchor_logits,
            "raw_handle_hidden": handle_hidden,
            "valid_sentence_rows": valid_sentence_rows,
            "valid_gold_handles": matched_handles,
            "matched_handles": matched_handles,
            "matched_anchor_handles": matched_handles & matched_anchor_targets.ge(0),
            "matched_anchor_targets": matched_anchor_targets,
            "query_seed_guard_rate": query_seed_guard_rate.detach(),
        }

    def apply_handle_latent_controller(self, latent_hidden, handle_memory, handle_active_mask=None):
        """Refine a latent state by attending to lightweight event handles."""
        if (
            handle_memory is None
            or not hasattr(self, "handle_latent_query_proj")
            or float(getattr(self, "event_handle_latent_attention_factor", 0.0)) <= 0
        ):
            return latent_hidden
        if handle_memory.size(1) == 0:
            return latent_hidden

        proj_dtype = self.handle_latent_query_proj.weight.dtype
        query = self.handle_latent_query_proj(latent_hidden.to(dtype=proj_dtype)).float().unsqueeze(1)
        keys = self.handle_latent_key_proj(handle_memory.to(dtype=proj_dtype)).float()
        values = self.handle_latent_value_proj(handle_memory.to(dtype=proj_dtype)).float()
        attn_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(float(keys.size(-1)))

        if handle_active_mask is not None:
            active_mask = handle_active_mask.to(device=handle_memory.device).bool()
            if active_mask.size(1) != handle_memory.size(1):
                active_mask = active_mask[:, : handle_memory.size(1)]
            valid_rows = active_mask.any(dim=1)
            attn_logits = attn_logits.masked_fill(~active_mask, -1e4)
        else:
            valid_rows = torch.ones((handle_memory.size(0),), dtype=torch.bool, device=handle_memory.device)

        if not bool(valid_rows.any()):
            return latent_hidden

        attn_weights = torch.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn_weights.unsqueeze(1), values).squeeze(1)
        gate_input = torch.cat((latent_hidden.float(), context), dim=-1)
        gate = torch.sigmoid(self.handle_latent_gate(gate_input.to(dtype=proj_dtype)).float())
        updated = latent_hidden.float() + float(self.event_handle_latent_attention_factor) * gate * context
        updated = torch.where(valid_rows.unsqueeze(-1), updated, latent_hidden.float())
        return updated.to(dtype=latent_hidden.dtype, device=latent_hidden.device)

    def build_decoder_bridge_embeds(
        self,
        latent_hidden,
        handle_memory=None,
        handle_active_mask=None,
        return_gate=False,
        return_mask=False,
    ):
        """Create event-level bridge tokens that condition JSON decoding on latent state and handles."""
        if (
            latent_hidden is None
            or not hasattr(self, "decoder_bridge_proj")
            or float(getattr(self, "decoder_bridge_factor", 0.0)) <= 0
        ):
            if return_gate and return_mask:
                return None, None, None
            if return_gate:
                return None, None
            if return_mask:
                return None, None
            return None

        bridge_dtype = self.decoder_bridge_proj.weight.dtype
        latent_float = latent_hidden.float()
        latent_tokens = latent_float.unsqueeze(1)
        bridge_mask = torch.ones(
            (latent_float.size(0), 1),
            dtype=torch.bool,
            device=latent_hidden.device,
        )
        handle_values = None
        if handle_memory is not None and handle_memory.size(1) > 0:
            handle_values = handle_memory.float()
            if handle_active_mask is not None:
                active_mask = handle_active_mask.to(device=handle_memory.device).bool()
                if active_mask.size(1) != handle_memory.size(1):
                    active_mask = active_mask[:, : handle_memory.size(1)]
                    if active_mask.size(1) < handle_memory.size(1):
                        active_mask = F.pad(
                            active_mask,
                            (0, handle_memory.size(1) - active_mask.size(1)),
                            value=False,
                        )
            else:
                active_mask = torch.ones(
                    handle_values.shape[:2],
                    dtype=torch.bool,
                    device=handle_memory.device,
                )
            bridge_mask = active_mask.to(device=latent_hidden.device).clone()
            no_active_rows = ~bridge_mask.any(dim=1)
            if no_active_rows.any():
                bridge_mask[no_active_rows, 0] = True
            active_values = handle_values.to(device=latent_hidden.device) * active_mask.to(
                device=latent_hidden.device
            ).unsqueeze(-1).float()
            latent_tokens = latent_float.unsqueeze(1).expand(-1, handle_values.size(1), -1)
            context = latent_tokens + active_values
            weights = active_mask.to(device=latent_hidden.device).float()
            valid_rows = weights.sum(dim=1, keepdim=True) > 0
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            handle_summary = (weights.unsqueeze(-1) * handle_values.to(device=latent_hidden.device)).sum(dim=1)
            sample_context = torch.where(valid_rows, latent_float + handle_summary, latent_float)
        else:
            context = latent_tokens
            sample_context = latent_float

        projected = self.decoder_bridge_proj(context.to(dtype=bridge_dtype)).float()
        gate_input = torch.cat((latent_tokens.float(), context.float()), dim=-1)
        gate = torch.sigmoid(self.decoder_bridge_gate(gate_input.to(dtype=bridge_dtype)).float())
        delta = float(self.decoder_bridge_factor) * gate * projected
        output_gate = torch.ones(
            (latent_float.size(0), 1),
            dtype=delta.dtype,
            device=delta.device,
        )
        if self.adaptive_bridge_gate and hasattr(self, "decoder_bridge_contribution_gate"):
            sample_gate_input = torch.cat((latent_float, sample_context.float()), dim=-1)
            output_gate = torch.sigmoid(
                self.decoder_bridge_contribution_gate(sample_gate_input.to(dtype=bridge_dtype)).float()
            )
        bridge = latent_tokens + output_gate.view(-1, 1, 1) * delta
        bridge = bridge.to(device=latent_hidden.device, dtype=latent_hidden.dtype)
        bridge_mask = bridge_mask.to(device=latent_hidden.device)
        if return_gate and return_mask:
            return bridge, output_gate, bridge_mask
        if return_gate:
            return bridge, output_gate
        if return_mask:
            return bridge, bridge_mask
        return bridge

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        process_ref_ranges: Optional[torch.LongTensor] = None,
        latent_target_steps: Optional[torch.LongTensor] = None,
        event_count_targets: Optional[torch.LongTensor] = None,
        sentence_rep_positions: Optional[torch.LongTensor] = None,
        event_handle_anchor_targets: Optional[torch.LongTensor] = None,
        document_token_mask: Optional[torch.Tensor] = None,
        decoder_token_weights: Optional[torch.Tensor] = None,
        decoder_prefix_robust_mask: Optional[torch.Tensor] = None,
        decoder_prefix_key_robust_mask: Optional[torch.Tensor] = None,
        decoder_prefix_event_start_robust_mask: Optional[torch.Tensor] = None,
        decoder_start_continuation_mask: Optional[torch.Tensor] = None,
        decoder_boundary_label_positions: Optional[torch.Tensor] = None,
        decoder_boundary_targets: Optional[torch.Tensor] = None,
        decoder_boundary_completed_counts: Optional[torch.Tensor] = None,
        decoder_boundary_continue_ids: Optional[torch.Tensor] = None,
        decoder_boundary_stop_ids: Optional[torch.Tensor] = None,
        decoder_proxy_trigger_mask: Optional[torch.Tensor] = None,
        decoder_proxy_arg_i_mask: Optional[torch.Tensor] = None,
        decoder_proxy_arg_c_mask: Optional[torch.Tensor] = None,
        decoder_proxy_span_mask: Optional[torch.Tensor] = None,
        decoder_proxy_head_mask: Optional[torch.Tensor] = None,
        sample_loss_weights: Optional[torch.Tensor] = None,
        step: int = None,
        step_ratio: float = None
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None
        # Keep the latent rollout on the KV-cache path for memory efficiency.
        # Recomputing the full prefix for micro-batches > 1 avoids a Qwen SDPA
        # mask/cache corner case but is too memory-expensive for 4k documents.
        # Use gradient accumulation, not larger micro-batches, for a larger
        # effective batch in this long-document latent setup.
        use_kv_cache_rollout = True
        if (
            self.training
            and encoder_input_ids is not None
            and encoder_input_ids.size(0) > 1
            and "qwen" in str(self.model_name).lower()
        ):
            raise ValueError(
                "Qwen APLER latent rollout currently supports per_device_train_batch_size=1. "
                "Use gradient_accumulation_steps to increase the effective batch size."
            )

        # Encode the question
        past_key_values = None
        encoder_input_embeds = self.embed_input_ids(encoder_input_ids)
        outputs = self.backbone_forward(
            inputs_embeds=encoder_input_embeds,
            use_cache=use_kv_cache_rollout,
            output_hidden_states=False,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = self.normalize_past_key_values(outputs.past_key_values) if use_kv_cache_rollout else None
        reasoning_attention_mask = (
            encoder_attention_mask.bool()
            if self.fix_attn_mask and encoder_attention_mask is not None
            else None
        )
        rollout_prefix_embeds = encoder_input_embeds
        rollout_attention_mask = reasoning_attention_mask
        encoder_states = outputs.last_hidden_state
        if encoder_attention_mask is not None:
            encoder_lengths = encoder_attention_mask.to(encoder_states.device).long().sum(dim=1).clamp_min(1)
            summary_positions = (encoder_lengths - 1).clamp(max=encoder_states.size(1) - 1)
            encoder_summary = encoder_states.gather(
                1,
                summary_positions.view(-1, 1, 1).expand(-1, 1, encoder_states.size(-1)),
            ).squeeze(1)
        else:
            encoder_summary = encoder_states[:, -1, :]
        encoder_summary = self.fuse_document_summary(
            encoder_states,
            encoder_summary,
            attention_mask=encoder_attention_mask,
            document_token_mask=document_token_mask,
        )
        latent_embd = encoder_summary.unsqueeze(1) # as the next input
        if self.use_prj:
            latent_embd = self.prj(latent_embd)

        # Iterate over the latent embeddings
        zero_loss = torch.zeros((), device=encoder_input_ids.device)
        process_distill_loss_total = zero_loss.clone()
        complexity_loss_total = zero_loss.clone()
        event_handle_anchor_loss_total = zero_loss.clone()
        event_handle_exist_loss_total = zero_loss.clone()
        event_handle_exist_bce_loss_metric = 0.0
        event_handle_count_nll_loss_metric = 0.0
        event_handle_gold_count_probability_metric = 0.0
        event_handle_count_entropy_metric = 0.0
        event_handle_count_reliability_metric = 0.0
        event_handle_expected_count_mae_metric = 0.0
        ce_loss_total = zero_loss.clone()
        ce_loss_for_gate_total = zero_loss.clone()
        native_boundary_bce_total = zero_loss.clone()
        predicted_complexity_metric = 0.0
        latent_budget_accuracy_metric = 0.0
        latent_budget_top1_mean_metric = 0.0
        latent_budget_expected_mean_metric = 0.0
        latent_budget_expected_mae_metric = 0.0
        latent_budget_proposal_accuracy_metric = 0.0
        latent_budget_proposal_mae_metric = 0.0
        latent_budget_entropy_metric = 0.0
        latent_budget_top1_margin_metric = 0.0
        latent_budget_ce_loss_metric = 0.0
        latent_budget_ordinal_loss_metric = 0.0
        latent_budget_target_hist_metrics = {
            step: 0.0
            for step in range(
                min(MIN_LATENT_BUDGET, max(1, int(self.num_latent))),
                int(self.num_latent) + 1,
            )
        }
        latent_budget_proposal_hist_metrics = dict(
            latent_budget_target_hist_metrics
        )
        event_handle_anchor_accuracy_metric = 0.0
        event_handle_exist_accuracy_metric = 0.0
        event_handle_exist_pos_probability_metric = 0.0
        event_handle_exist_neg_probability_metric = 0.0
        event_handle_conf_ge30_metric = 0.0
        event_handle_conf_ge50_metric = 0.0
        event_handle_control_recall_metric = 0.0
        event_handle_count_accuracy_metric = 0.0
        event_handle_query_seed_guard_rate_metric = 0.0
        decoder_bridge_gate_sum_metric = zero_loss.clone()
        decoder_bridge_delta_sum_metric = zero_loss.clone()
        decoder_bridge_sample_count_metric = zero_loss.clone()
        json_prefix_robust_drop_sum_metric = zero_loss.clone()
        json_prefix_robust_candidate_sum_metric = zero_loss.clone()
        json_prefix_value_drop_sum_metric = zero_loss.clone()
        json_prefix_value_candidate_sum_metric = zero_loss.clone()
        json_prefix_key_drop_sum_metric = zero_loss.clone()
        json_prefix_key_candidate_sum_metric = zero_loss.clone()
        json_prefix_event_start_drop_sum_metric = zero_loss.clone()
        json_prefix_event_start_candidate_sum_metric = zero_loss.clone()
        json_start_continuation_drop_sum_metric = zero_loss.clone()
        json_start_continuation_candidate_sum_metric = zero_loss.clone()
        boundary_active_count_metric = zero_loss.clone()
        boundary_candidate_count_metric = zero_loss.clone()
        boundary_reliability_sum_metric = zero_loss.clone()
        boundary_continue_count_metric = zero_loss.clone()
        boundary_agreement_sum_metric = zero_loss.clone()
        boundary_agreement_weight_metric = zero_loss.clone()
        boundary_qwen_continue_sum_metric = zero_loss.clone()
        boundary_blended_continue_sum_metric = zero_loss.clone()
        native_boundary_gold_count_metric = zero_loss.clone()
        native_boundary_valid_count_metric = zero_loss.clone()
        native_boundary_proposal_count_metric = zero_loss.clone()
        native_boundary_correct_count_metric = zero_loss.clone()
        native_boundary_continue_probability_on_continue_sum_metric = (
            zero_loss.clone()
        )
        native_boundary_continue_target_count_metric = zero_loss.clone()
        native_boundary_continue_probability_on_stop_sum_metric = (
            zero_loss.clone()
        )
        native_boundary_stop_target_count_metric = zero_loss.clone()
        native_boundary_low_count_correct_metric = zero_loss.clone()
        native_boundary_low_count_total_metric = zero_loss.clone()
        native_boundary_high_count_correct_metric = zero_loss.clone()
        native_boundary_high_count_total_metric = zero_loss.clone()
        native_boundary_first_correct_metric = zero_loss.clone()
        native_boundary_first_total_metric = zero_loss.clone()
        native_boundary_later_correct_metric = zero_loss.clone()
        native_boundary_later_total_metric = zero_loss.clone()
        event_boundary_mode = normalize_event_boundary_control(
            getattr(self.training_args, "event_boundary_control", "off")
        )
        train_proxy_exact_hits = {
            "trigger": zero_loss.clone(),
            "arg_i": zero_loss.clone(),
            "arg_c": zero_loss.clone(),
            "span": zero_loss.clone(),
            "head": zero_loss.clone(),
        }
        train_proxy_topk_hits = {
            "trigger": zero_loss.clone(),
            "arg_i": zero_loss.clone(),
            "arg_c": zero_loss.clone(),
            "span": zero_loss.clone(),
            "head": zero_loss.clone(),
        }
        train_proxy_counts = {
            "trigger": zero_loss.clone(),
            "arg_i": zero_loss.clone(),
            "arg_c": zero_loss.clone(),
            "span": zero_loss.clone(),
            "head": zero_loss.clone(),
        }
        process_supervision_steps = 0
        logits = None
        ce_token_weight_alpha_metric = 1.0
        decoder_exit_mix_rate_metric = 0.0
        decoder_exit_gap_metric = 0.0
        decoder_exit_used_metric = 0.0

        def current_ce_token_weight_alpha():
            warmup_steps = int(
                max(
                    getattr(
                        self.training_args,
                        "ce_token_weight_warmup_steps",
                        0,
                    ),
                    0,
                )
            )
            warmup_ratio = float(
                max(
                    getattr(
                        self.training_args,
                        "ce_token_weight_warmup_ratio",
                        0.0,
                    ),
                    0.0,
                )
            )
            if self.training and warmup_steps > 0 and step is not None:
                return min(
                    1.0,
                    max(0.0, float(step) / float(max(warmup_steps, 1))),
                )
            if self.training and warmup_ratio > 0 and step_ratio is not None:
                return min(1.0, max(0.0, float(step_ratio) / warmup_ratio))
            return 1.0

        def scheduled_ce_token_weights(token_weights):
            nonlocal ce_token_weight_alpha_metric
            if token_weights is None:
                return None
            weights = token_weights.float()
            alpha = current_ce_token_weight_alpha()
            ce_token_weight_alpha_metric = alpha
            weights = 1.0 + alpha * (weights - 1.0)
            return weights

        def scheduled_json_prefix_dropout(base_prob):
            if not bool(getattr(self.training_args, "json_prefix_robust_enable", True)):
                return 0.0
            prob = float(max(0.0, min(1.0, base_prob)))
            if prob <= 0.0:
                return 0.0
            warmup_ratio = float(max(0.0, getattr(self.training_args, "json_prefix_robust_warmup_ratio", 0.0)))
            if step_ratio is not None and warmup_ratio > 0.0:
                prob *= min(1.0, max(0.0, float(step_ratio)) / warmup_ratio)
            return prob

        def scheduled_json_prefix_value_dropout():
            configured = float(getattr(self.training_args, "json_prefix_value_dropout", -1.0))
            if configured < 0.0:
                configured = float(getattr(self.training_args, "json_prefix_robust_dropout", 0.0))
            if not bool(getattr(self.training_args, "json_prefix_value_perturb", True)):
                return 0.0
            return scheduled_json_prefix_dropout(configured)

        def scheduled_json_start_continuation_prob():
            if not bool(getattr(self.training_args, "json_start_continuation_enable", False)):
                return 0.0
            prob = float(max(0.0, min(1.0, getattr(self.training_args, "json_start_continuation_prob", 0.0))))
            if prob <= 0.0:
                return 0.0
            warmup_ratio = float(max(0.0, getattr(self.training_args, "json_start_continuation_warmup_ratio", 0.0)))
            if step_ratio is not None and float(step_ratio) < warmup_ratio:
                return 0.0
            return prob

        sample_weight_mode = str(
            getattr(self, "complex_sample_weight_mode", "ce_only") or "ce_only"
        ).lower()
        full_objective_sample_weighting = sample_weight_mode in {
            "full",
            "full_objective",
            "objective",
        }
        ce_only_sample_weighting = sample_weight_mode in {
            "ce",
            "ce_only",
            "legacy",
        }

        def sequence_cross_entropy(
            logits_tensor,
            label_tensor,
            active_mask=None,
            token_weights=None,
            return_base_loss=False,
        ):
            labels_for_loss = label_tensor.to(logits_tensor.device)
            token_loss = F.cross_entropy(
                logits_tensor.float().reshape(-1, logits_tensor.size(-1)),
                labels_for_loss.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels_for_loss.size(0), labels_for_loss.size(1))
            valid_tokens = labels_for_loss.ne(-100)
            if active_mask is not None:
                active = active_mask.to(logits_tensor.device).bool()
                valid_tokens = valid_tokens & active.unsqueeze(1)
            else:
                active = valid_tokens.any(dim=1)
            if not valid_tokens.any():
                if return_base_loss:
                    empty = zero_loss.clone()
                    return empty, empty
                return zero_loss.clone()

            loss_weights = valid_tokens.float()
            if token_weights is not None:
                scheduled_weights = scheduled_ce_token_weights(token_weights).to(logits_tensor.device).clamp_min(0.0)
                loss_weights = loss_weights * scheduled_weights
            per_sample_loss = (token_loss * loss_weights).sum(dim=1)
            per_sample_den = loss_weights.sum(dim=1).clamp_min(1.0)
            per_sample_loss = per_sample_loss / per_sample_den
            sample_mask = active & valid_tokens.any(dim=1)
            base_loss = per_sample_loss[sample_mask].mean() if sample_mask.any() else zero_loss.clone()
            if sample_loss_weights is not None and ce_only_sample_weighting:
                weights = sample_loss_weights.to(logits_tensor.device).float().clamp(
                    float(self.sample_loss_weight_min),
                    float(self.sample_loss_weight_max),
                )
                per_sample_loss = per_sample_loss * weights
            weighted_loss = per_sample_loss[sample_mask].mean() if sample_mask.any() else zero_loss.clone()
            if return_base_loss:
                return weighted_loss, base_loss
            return weighted_loss

        def sequence_lm_head_cross_entropy(
            hidden_tensor,
            label_tensor,
            active_mask=None,
            token_weights=None,
            chunk_size=None,
            return_base_loss=False,
        ):
            nonlocal boundary_active_count_metric
            nonlocal boundary_candidate_count_metric
            nonlocal boundary_reliability_sum_metric
            nonlocal boundary_continue_count_metric
            nonlocal boundary_agreement_sum_metric
            nonlocal boundary_agreement_weight_metric
            nonlocal boundary_qwen_continue_sum_metric
            nonlocal boundary_blended_continue_sum_metric
            nonlocal native_boundary_gold_count_metric
            nonlocal native_boundary_valid_count_metric
            nonlocal native_boundary_proposal_count_metric
            nonlocal native_boundary_correct_count_metric
            nonlocal native_boundary_continue_probability_on_continue_sum_metric
            nonlocal native_boundary_continue_target_count_metric
            nonlocal native_boundary_continue_probability_on_stop_sum_metric
            nonlocal native_boundary_stop_target_count_metric
            nonlocal native_boundary_low_count_correct_metric
            nonlocal native_boundary_low_count_total_metric
            nonlocal native_boundary_high_count_correct_metric
            nonlocal native_boundary_high_count_total_metric
            nonlocal native_boundary_first_correct_metric
            nonlocal native_boundary_first_total_metric
            nonlocal native_boundary_later_correct_metric
            nonlocal native_boundary_later_total_metric
            chunk_size = int(chunk_size or getattr(self.training_args, "lm_head_chunk_size", 16) or 16)
            labels_for_loss = label_tensor.to(hidden_tensor.device)
            valid_tokens = labels_for_loss.ne(-100)
            if active_mask is not None:
                active = active_mask.to(hidden_tensor.device).bool()
                valid_tokens = valid_tokens & active.unsqueeze(1)
            else:
                active = valid_tokens.any(dim=1)
            if not valid_tokens.any():
                if return_base_loss:
                    empty = zero_loss.clone()
                    return empty, empty, empty
                return zero_loss.clone()

            token_loss = torch.zeros(
                labels_for_loss.shape,
                dtype=torch.float,
                device=hidden_tensor.device,
            )
            boundary_positions = None
            boundary_probabilities = None
            boundary_count_distribution = None
            boundary_reliability = None
            boundary_warmup_scale = 1.0
            if (
                decoder_boundary_label_positions is not None
                and decoder_boundary_targets is not None
                and decoder_boundary_completed_counts is not None
                and decoder_boundary_continue_ids is not None
                and decoder_boundary_stop_ids is not None
            ):
                boundary_positions = (
                    decoder_boundary_label_positions.to(
                        device=hidden_tensor.device,
                        dtype=torch.long,
                    )
                    - 1
                )
            if (
                event_boundary_mode != "off"
                and event_handle_exist_probabilities_for_boundary is not None
                and boundary_positions is not None
            ):
                boundary_probabilities = (
                    event_handle_exist_probabilities_for_boundary.to(
                        device=hidden_tensor.device,
                        dtype=torch.float,
                    )
                )
                boundary_count_distribution = event_count_distribution_tensor(
                    boundary_probabilities
                )
                boundary_reliability = event_count_reliability_tensor(
                    boundary_count_distribution
                )
                if self.training and step_ratio is not None:
                    decay_ratio = max(
                        float(self.event_handle_teacher_forcing_decay_ratio),
                        1e-6,
                    )
                    boundary_warmup_scale = min(
                        1.0,
                        max(0.0, float(step_ratio) / decay_ratio),
                    )
                if boundary_reliability is not None:
                    boundary_reliability = (
                        boundary_reliability * float(boundary_warmup_scale)
                    ).clamp(0.0, 1.0)
            boundary_token_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
            if boundary_positions is not None:
                boundary_targets_for_mask = decoder_boundary_targets.to(
                    device=hidden_tensor.device,
                    dtype=torch.long,
                )
                valid_boundary_positions = (
                    boundary_positions.ge(0)
                    & boundary_positions.lt(hidden_tensor.size(1))
                    & boundary_targets_for_mask.ge(0)
                )
                if active_mask is not None:
                    valid_boundary_positions = (
                        valid_boundary_positions
                        & active_mask.to(hidden_tensor.device).bool().unsqueeze(1)
                    )
                if valid_boundary_positions.any():
                    boundary_rows, boundary_slots = (
                        valid_boundary_positions.nonzero(as_tuple=True)
                    )
                    boundary_token_mask[
                        boundary_rows,
                        boundary_positions[boundary_rows, boundary_slots],
                    ] = True
            native_boundary_row_losses = []
            native_boundary_valid_rows = []
            native_boundary_sample_rows = []
            for start in range(0, hidden_tensor.size(1), max(1, chunk_size)):
                end = min(start + max(1, chunk_size), hidden_tensor.size(1))
                if not valid_tokens[:, start:end].any():
                    continue
                chunk_hidden = hidden_tensor[:, start:end, :]
                chunk_logits = self.lm_head_logits(chunk_hidden)
                if boundary_positions is not None:
                    boundary_targets = decoder_boundary_targets.to(
                        device=hidden_tensor.device,
                        dtype=torch.long,
                    )
                    boundary_entries = (
                        boundary_positions.ge(start)
                        & boundary_positions.lt(end)
                        & boundary_targets.ge(0)
                    )
                    if active_mask is not None:
                        boundary_entries = (
                            boundary_entries
                            & active_mask.to(hidden_tensor.device).bool().unsqueeze(1)
                        )
                    boundary_candidate_count_metric = (
                        boundary_candidate_count_metric
                        + boundary_entries.float().sum().detach()
                    )
                    if boundary_entries.any():
                        sample_rows, boundary_slots = boundary_entries.nonzero(
                            as_tuple=True
                        )
                        local_positions = (
                            boundary_positions[sample_rows, boundary_slots] - start
                        )
                        continue_ids = decoder_boundary_continue_ids.to(
                            device=hidden_tensor.device,
                            dtype=torch.long,
                        )[sample_rows, boundary_slots]
                        stop_ids = decoder_boundary_stop_ids.to(
                            device=hidden_tensor.device,
                            dtype=torch.long,
                        )[sample_rows, boundary_slots]
                        sparse_logits = chunk_logits[
                            sample_rows,
                            local_positions,
                            :,
                        ]
                        gold_targets_all = boundary_targets[
                            sample_rows,
                            boundary_slots,
                        ]
                        (
                            native_row_loss,
                            native_continue_probability,
                            native_valid_rows,
                        ) = native_event_boundary_statistics(
                            sparse_logits,
                            continue_ids,
                            stop_ids,
                            gold_targets_all,
                        )
                        native_boundary_row_losses.append(
                            native_row_loss
                            if (
                                self.native_boundary_loss_enable
                                and self.native_boundary_loss_factor > 0.0
                            )
                            else native_row_loss.detach()
                        )
                        native_boundary_valid_rows.append(native_valid_rows)
                        native_boundary_sample_rows.append(sample_rows)
                        native_boundary_gold_count_metric = (
                            native_boundary_gold_count_metric
                            + native_valid_rows.new_tensor(
                                native_valid_rows.numel(),
                                dtype=torch.float,
                            )
                        )
                        native_boundary_valid_count_metric = (
                            native_boundary_valid_count_metric
                            + native_valid_rows.detach().float().sum()
                        )
                        native_proposal_rows = event_boundary_proposal_mask(
                            sparse_logits,
                            continue_ids,
                            stop_ids,
                            proposal_vocab_size=len(self.tokenizer),
                        )
                        native_boundary_proposal_count_metric = (
                            native_boundary_proposal_count_metric
                            + (
                                native_proposal_rows
                                & native_valid_rows
                            ).detach().float().sum()
                        )
                        native_predictions = native_continue_probability.ge(0.5)
                        native_correct = (
                            native_predictions
                            == gold_targets_all.eq(1)
                        ) & native_valid_rows
                        native_boundary_correct_count_metric = (
                            native_boundary_correct_count_metric
                            + native_correct.detach().float().sum()
                        )
                        gold_continue_rows = (
                            native_valid_rows & gold_targets_all.eq(1)
                        )
                        gold_stop_rows = (
                            native_valid_rows & gold_targets_all.eq(0)
                        )
                        native_boundary_continue_probability_on_continue_sum_metric = (
                            native_boundary_continue_probability_on_continue_sum_metric
                            + native_continue_probability[
                                gold_continue_rows
                            ].detach().sum()
                        )
                        native_boundary_continue_target_count_metric = (
                            native_boundary_continue_target_count_metric
                            + gold_continue_rows.detach().float().sum()
                        )
                        native_boundary_continue_probability_on_stop_sum_metric = (
                            native_boundary_continue_probability_on_stop_sum_metric
                            + native_continue_probability[
                                gold_stop_rows
                            ].detach().sum()
                        )
                        native_boundary_stop_target_count_metric = (
                            native_boundary_stop_target_count_metric
                            + gold_stop_rows.detach().float().sum()
                        )
                        if event_count_targets is not None:
                            boundary_gold_counts = event_count_targets.to(
                                device=hidden_tensor.device,
                                dtype=torch.long,
                            )[sample_rows]
                            low_count_rows = (
                                native_valid_rows & boundary_gold_counts.le(2)
                            )
                            high_count_rows = (
                                native_valid_rows & boundary_gold_counts.ge(3)
                            )
                            native_boundary_low_count_correct_metric = (
                                native_boundary_low_count_correct_metric
                                + (
                                    native_correct & low_count_rows
                                ).detach().float().sum()
                            )
                            native_boundary_low_count_total_metric = (
                                native_boundary_low_count_total_metric
                                + low_count_rows.detach().float().sum()
                            )
                            native_boundary_high_count_correct_metric = (
                                native_boundary_high_count_correct_metric
                                + (
                                    native_correct & high_count_rows
                                ).detach().float().sum()
                            )
                            native_boundary_high_count_total_metric = (
                                native_boundary_high_count_total_metric
                                + high_count_rows.detach().float().sum()
                            )
                        completed_counts_all = (
                            decoder_boundary_completed_counts.to(
                                device=hidden_tensor.device,
                                dtype=torch.long,
                            )[sample_rows, boundary_slots]
                        )
                        first_rows = (
                            native_valid_rows & completed_counts_all.eq(1)
                        )
                        later_rows = (
                            native_valid_rows & completed_counts_all.ge(2)
                        )
                        native_boundary_first_correct_metric = (
                            native_boundary_first_correct_metric
                            + (
                                native_correct & first_rows
                            ).detach().float().sum()
                        )
                        native_boundary_first_total_metric = (
                            native_boundary_first_total_metric
                            + first_rows.detach().float().sum()
                        )
                        native_boundary_later_correct_metric = (
                            native_boundary_later_correct_metric
                            + (
                                native_correct & later_rows
                            ).detach().float().sum()
                        )
                        native_boundary_later_total_metric = (
                            native_boundary_later_total_metric
                            + later_rows.detach().float().sum()
                        )

                        if boundary_count_distribution is not None:
                            proposal_rows = torch.ones(
                                sparse_logits.size(0),
                                dtype=torch.bool,
                                device=hidden_tensor.device,
                            )
                            if event_boundary_mode == "handle_hierarchical":
                                proposal_rows = native_proposal_rows
                            if proposal_rows.any():
                                proposal_indices = proposal_rows.nonzero(
                                    as_tuple=True
                                )[0]
                                selected_sample_rows = sample_rows[proposal_indices]
                                selected_boundary_slots = boundary_slots[proposal_indices]
                                selected_local_positions = local_positions[proposal_indices]
                                completed_counts = (
                                    decoder_boundary_completed_counts.to(
                                        device=hidden_tensor.device,
                                        dtype=torch.long,
                                    )[selected_sample_rows, selected_boundary_slots]
                                )
                                handle_continue = (
                                    conditional_event_continue_probability_tensor(
                                        boundary_count_distribution[
                                            selected_sample_rows
                                        ],
                                        completed_counts,
                                    )
                                )
                                reliability = boundary_reliability[
                                    selected_sample_rows
                                ]
                                (
                                    controlled_logits,
                                    qwen_continue,
                                    blended_continue,
                                    valid_rows,
                                ) = blend_event_boundary_logits(
                                    sparse_logits[proposal_indices],
                                    continue_ids[proposal_indices],
                                    stop_ids[proposal_indices],
                                    handle_continue,
                                    reliability,
                                )
                            else:
                                valid_rows = torch.zeros(
                                    0,
                                    dtype=torch.bool,
                                    device=hidden_tensor.device,
                                )
                        else:
                            valid_rows = torch.zeros(
                                0,
                                dtype=torch.bool,
                                device=hidden_tensor.device,
                            )
                        if valid_rows.any():
                            chunk_logits = chunk_logits.clone()
                            valid_sample_rows = selected_sample_rows[valid_rows]
                            valid_boundary_slots = selected_boundary_slots[valid_rows]
                            valid_local_positions = selected_local_positions[valid_rows]
                            chunk_logits[
                                valid_sample_rows,
                                valid_local_positions,
                                :,
                            ] = controlled_logits[valid_rows]
                            valid_reliability = reliability[valid_rows].detach()
                            gold_targets = boundary_targets[
                                valid_sample_rows,
                                valid_boundary_slots,
                            ].float()
                            prior_agreement = (
                                handle_continue[valid_rows].ge(0.5).float()
                                == gold_targets
                            ).float()
                            boundary_active_count_metric = (
                                boundary_active_count_metric
                                + valid_rows.float().sum().detach()
                            )
                            boundary_reliability_sum_metric = (
                                boundary_reliability_sum_metric
                                + valid_reliability.sum()
                            )
                            boundary_continue_count_metric = (
                                boundary_continue_count_metric
                                + gold_targets.detach().sum()
                            )
                            boundary_agreement_sum_metric = (
                                boundary_agreement_sum_metric
                                + (prior_agreement.detach() * valid_reliability).sum()
                            )
                            boundary_agreement_weight_metric = (
                                boundary_agreement_weight_metric
                                + valid_reliability.sum()
                            )
                            boundary_qwen_continue_sum_metric = (
                                boundary_qwen_continue_sum_metric
                                + qwen_continue[valid_rows].detach().sum()
                            )
                            boundary_blended_continue_sum_metric = (
                                boundary_blended_continue_sum_metric
                                + blended_continue[valid_rows].detach().sum()
                            )
                chunk_loss = F.cross_entropy(
                    chunk_logits.float().reshape(-1, chunk_logits.size(-1)),
                    labels_for_loss[:, start:end].reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(labels_for_loss.size(0), end - start)
                token_loss[:, start:end] = chunk_loss
                del chunk_logits, chunk_loss

            if native_boundary_row_losses:
                native_boundary_bce = mean_event_boundary_loss_per_sample(
                    torch.cat(native_boundary_row_losses, dim=0),
                    torch.cat(native_boundary_valid_rows, dim=0),
                    torch.cat(native_boundary_sample_rows, dim=0),
                    hidden_tensor.size(0),
                )
            else:
                native_boundary_bce = zero_loss.clone()
            loss_weights = valid_tokens.float()
            if token_weights is not None:
                scheduled_weights = scheduled_ce_token_weights(token_weights).to(hidden_tensor.device).clamp_min(0.0)
                loss_weights = loss_weights * scheduled_weights
            per_sample_loss = (token_loss * loss_weights).sum(dim=1)
            per_sample_den = loss_weights.sum(dim=1).clamp_min(1.0)
            per_sample_loss = per_sample_loss / per_sample_den
            sample_mask = active & valid_tokens.any(dim=1)
            base_loss = auxiliary_gate_cross_entropy(
                token_loss,
                loss_weights,
                boundary_token_mask,
                sample_mask,
            )
            if sample_loss_weights is not None and ce_only_sample_weighting:
                weights = sample_loss_weights.to(hidden_tensor.device).float().clamp(
                    float(self.sample_loss_weight_min),
                    float(self.sample_loss_weight_max),
                )
                per_sample_loss = per_sample_loss * weights
            weighted_loss = per_sample_loss[sample_mask].mean() if sample_mask.any() else zero_loss.clone()
            if return_base_loss:
                return weighted_loss, base_loss, native_boundary_bce
            return weighted_loss

        decoder_proxy_masks = {
            "trigger": decoder_proxy_trigger_mask,
            "arg_i": decoder_proxy_arg_i_mask,
            "arg_c": decoder_proxy_arg_c_mask,
            "span": decoder_proxy_span_mask,
            "head": decoder_proxy_head_mask,
        }

        def update_train_proxy_metrics(hidden_tensor, label_tensor, active_mask=None):
            available_masks = {
                name: mask
                for name, mask in decoder_proxy_masks.items()
                if mask is not None
            }
            if not available_masks:
                return
            labels_for_proxy = label_tensor.to(hidden_tensor.device)
            valid_tokens = labels_for_proxy.ne(-100)
            if active_mask is not None:
                valid_tokens = valid_tokens & active_mask.to(hidden_tensor.device).bool().unsqueeze(1)
            if not valid_tokens.any():
                return

            shifted_masks = {}
            any_proxy_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
            for name, mask in available_masks.items():
                mask = mask.to(hidden_tensor.device).bool()
                if mask.size(1) == label_tensor.size(1) + 1:
                    mask = mask[:, 1:]
                elif mask.size(1) != label_tensor.size(1):
                    mask = mask[:, : label_tensor.size(1)]
                    if mask.size(1) < label_tensor.size(1):
                        pad = torch.zeros(
                            mask.size(0),
                            label_tensor.size(1) - mask.size(1),
                            dtype=torch.bool,
                            device=mask.device,
                        )
                        mask = torch.cat((mask, pad), dim=1)
                mask = mask & valid_tokens
                shifted_masks[name] = mask
                any_proxy_mask = any_proxy_mask | mask
            if not any_proxy_mask.any():
                return

            with torch.no_grad():
                pred_ids = torch.full_like(labels_for_proxy, -100)
                topk_hits = torch.zeros_like(valid_tokens, dtype=torch.bool)
                chunk_size = int(getattr(self.training_args, "lm_head_chunk_size", 16) or 16)
                for start in range(0, hidden_tensor.size(1), max(1, chunk_size)):
                    end = min(start + max(1, chunk_size), hidden_tensor.size(1))
                    if not any_proxy_mask[:, start:end].any():
                        continue
                    chunk_hidden = hidden_tensor[:, start:end, :].detach()
                    chunk_logits = self.lm_head_logits(chunk_hidden).float()
                    pred_ids[:, start:end] = chunk_logits.argmax(dim=-1)
                    k = min(5, chunk_logits.size(-1))
                    topk = torch.topk(chunk_logits, k=k, dim=-1).indices
                    chunk_labels = labels_for_proxy[:, start:end]
                    topk_hits[:, start:end] = topk.eq(chunk_labels.unsqueeze(-1)).any(dim=-1)
                    del chunk_logits, topk

                exact_hits = pred_ids.eq(labels_for_proxy) & valid_tokens
                for name, mask in shifted_masks.items():
                    count = mask.float().sum()
                    if count <= 0:
                        continue
                    train_proxy_counts[name] = train_proxy_counts[name] + count.detach()
                    train_proxy_exact_hits[name] = (
                        train_proxy_exact_hits[name]
                        + (exact_hits & mask).float().sum().detach()
                    )
                    train_proxy_topk_hits[name] = (
                        train_proxy_topk_hits[name]
                        + (topk_hits & mask).float().sum().detach()
                    )

        def scheduled_teacher_forcing_ratio(base_ratio, decay_ratio=1.0):
            ratio = float(max(0.0, min(1.0, base_ratio)))
            if step_ratio is not None:
                decay_ratio = float(max(decay_ratio, 1e-6))
                ratio = ratio * max(0.0, 1.0 - float(step_ratio) / decay_ratio)
            return ratio

        budget_logits = None
        budget_statistics = None
        if self.adaptive_latent and hasattr(self, "latent_budget_head") and latent_target_steps is not None:
            complexity_target = latent_target_steps.to(encoder_summary.device).long().clamp(
                min(MIN_LATENT_BUDGET, self.num_latent),
                self.num_latent,
            )
            budget_hidden = encoder_summary.to(dtype=self.latent_budget_head.weight.dtype)
            budget_statistics = latent_budget_statistics(
                self.latent_budget_head(budget_hidden).float(),
            )
            budget_logits = budget_statistics["logits"]
            budget_classification_loss = F.cross_entropy(
                budget_logits,
                complexity_target - 1,
            )
            budget_ordinal_loss = F.smooth_l1_loss(
                budget_statistics["expected"],
                complexity_target.float(),
            )
            complexity_loss_total = (
                budget_classification_loss
                + float(self.latent_budget_ordinal_loss_factor)
                * budget_ordinal_loss
            )
            latent_budget_ce_loss_metric = (
                budget_classification_loss.detach().item()
            )
            latent_budget_ordinal_loss_metric = budget_ordinal_loss.detach().item()
            budget_proposal = budget_statistics["proposal"]
            budget_top1 = budget_statistics["top1"]
            predicted_complexity_metric = (
                budget_proposal.detach().float().mean().item()
            )
            latent_budget_accuracy_metric = (
                budget_top1 == complexity_target
            ).float().mean().detach().item()
            latent_budget_top1_mean_metric = (
                budget_top1.detach().float().mean().item()
            )
            latent_budget_expected_mean_metric = (
                budget_statistics["expected"].detach().mean().item()
            )
            latent_budget_expected_mae_metric = (
                (budget_statistics["expected"].detach() - complexity_target.float())
                .abs()
                .mean()
                .item()
            )
            latent_budget_proposal_accuracy_metric = (
                budget_proposal == complexity_target
            ).float().mean().detach().item()
            latent_budget_proposal_mae_metric = (
                (budget_proposal.detach().float() - complexity_target.float())
                .abs()
                .mean()
                .item()
            )
            latent_budget_entropy_metric = (
                budget_statistics["entropy"].detach().mean().item()
            )
            latent_budget_top1_margin_metric = (
                budget_statistics["top1_margin"].detach().mean().item()
            )
            for budget_step in latent_budget_target_hist_metrics:
                latent_budget_target_hist_metrics[budget_step] = (
                    complexity_target.eq(budget_step)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                latent_budget_proposal_hist_metrics[budget_step] = (
                    budget_proposal.eq(budget_step)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )

        if (
            self.adaptive_latent
            and hasattr(self, "latent_budget_embedding")
            and self.latent_budget_condition_factor > 0
        ):
            pred_budget_condition = (
                budget_statistics["proposal"].detach().long()
                if budget_statistics is not None
                else None
            )
            gold_budget_condition = (
                latent_target_steps.to(encoder_input_ids.device).long().clamp(
                    min(MIN_LATENT_BUDGET, self.num_latent),
                    self.num_latent,
                )
                if latent_target_steps is not None
                else None
            )
            budget_condition_embd = None
            if gold_budget_condition is not None and pred_budget_condition is not None:
                gold_budget_embd = self.latent_budget_embedding(gold_budget_condition)
                pred_budget_embd = self.latent_budget_embedding(pred_budget_condition)
                tf_ratio = scheduled_teacher_forcing_ratio(self.latent_budget_condition_teacher_forcing)
                budget_condition_embd = tf_ratio * gold_budget_embd + (1.0 - tf_ratio) * pred_budget_embd
            elif pred_budget_condition is not None:
                budget_condition_embd = self.latent_budget_embedding(pred_budget_condition)
            elif gold_budget_condition is not None:
                budget_condition_embd = self.latent_budget_embedding(gold_budget_condition)
            if budget_condition_embd is not None:
                budget_condition_embd = budget_condition_embd.to(
                    device=latent_embd.device,
                    dtype=latent_embd.dtype,
                ).unsqueeze(1)
                latent_embd = latent_embd + float(self.latent_budget_condition_factor) * budget_condition_embd

        handle_budget_condition = None
        if (
            self.adaptive_latent
            and budget_logits is not None
            and hasattr(self, "latent_budget_embedding")
            and bool(getattr(self, "budget_conditioned_handle_count", False))
            and float(getattr(self, "budget_handle_count_factor", 0.0)) > 0.0
        ):
            budget_probs = budget_statistics["probabilities"].detach().float()
            budget_ids = torch.arange(
                1,
                self.num_latent + 1,
                dtype=torch.long,
                device=budget_logits.device,
            )
            budget_table = self.latent_budget_embedding(budget_ids).detach().to(
                device=budget_logits.device,
                dtype=budget_probs.dtype,
            )
            handle_budget_condition = torch.matmul(budget_probs, budget_table)

        sentence_hidden_for_handles = None
        valid_sentences_for_handles = None
        if sentence_rep_positions is not None:
            rep_positions = sentence_rep_positions.to(encoder_input_ids.device)
            valid_sentences = rep_positions >= 0
            if valid_sentences.any():
                safe_positions = rep_positions.clamp_min(0)
                sentence_hidden_for_handles = encoder_states.gather(
                    1,
                    safe_positions.unsqueeze(-1).expand(-1, -1, encoder_states.size(-1)),
                ).float()
                valid_sentences_for_handles = valid_sentences

        event_handle_memory = None
        event_handle_active_mask = None
        event_handle_exist_probabilities_for_boundary = None
        if (
            self.event_handle_condition_factor > 0
            and sentence_hidden_for_handles is not None
            and valid_sentences_for_handles is not None
            and hasattr(self, "event_handle_embedding")
        ):
            gold_handle_counts = (
                event_count_targets.to(encoder_input_ids.device).long()
                if event_count_targets is not None
                else None
            )
            handle_teacher_forcing = scheduled_teacher_forcing_ratio(
                self.event_handle_condition_teacher_forcing,
                self.event_handle_teacher_forcing_decay_ratio,
            )
            event_handle_result = self.build_event_handle_plan(
                sentence_hidden_for_handles,
                valid_sentences_for_handles,
                gold_handle_anchor_targets=event_handle_anchor_targets,
                gold_counts=gold_handle_counts,
                teacher_forcing_ratio=handle_teacher_forcing,
                global_context=latent_embd[:, -1, :],
                budget_condition=handle_budget_condition,
            )
            event_handle_plan = event_handle_result["plan"] if event_handle_result is not None else None
            if event_handle_result is not None:
                event_handle_active_mask = event_handle_result.get("plan_active")
                event_handle_query_seed_guard_rate_metric = float(
                    event_handle_result["query_seed_guard_rate"].detach().float().item()
                )
                event_handle_exist_probabilities_for_boundary = event_handle_result[
                    "handle_exist_probs"
                ].detach()
            if (
                event_handle_result is not None
                and event_handle_anchor_targets is not None
                and gold_handle_counts is not None
            ):
                handle_targets = event_handle_result["matched_anchor_targets"]
                valid_gold_handles = event_handle_result["matched_handles"]
                valid_anchor_handles = event_handle_result["matched_anchor_handles"]
                handle_exist_targets = valid_gold_handles.float()
                valid_handle_rows = event_handle_result["valid_sentence_rows"].bool()
                valid_handle_entries = valid_handle_rows.unsqueeze(1).expand_as(
                    handle_exist_targets
                )
                raw_handle_exist_logits = event_handle_result["handle_exist_logits"][
                    valid_handle_entries
                ].float()
                handle_exist_classification_loss = F.binary_cross_entropy_with_logits(
                    raw_handle_exist_logits,
                    handle_exist_targets[valid_handle_entries],
                )
                handle_exist_probabilities = torch.sigmoid(
                    event_handle_result["handle_exist_logits"].float()
                )
                handle_count_distribution = event_count_distribution_tensor(
                    handle_exist_probabilities
                )
                target_handle_cardinality = gold_handle_counts.to(
                    device=handle_count_distribution.device,
                    dtype=torch.long,
                ).clamp(0, self.max_event_count)
                handle_exist_count_loss = event_count_target_nll(
                    handle_count_distribution,
                    target_handle_cardinality,
                    valid_rows=valid_handle_rows,
                )
                event_handle_exist_loss_total = (
                    handle_exist_classification_loss
                    + float(self.event_handle_exist_count_loss_factor) * handle_exist_count_loss
                )
                event_handle_exist_bce_loss_metric = (
                    handle_exist_classification_loss.detach().item()
                )
                event_handle_count_nll_loss_metric = (
                    handle_exist_count_loss.detach().item()
                )
                if valid_handle_rows.any():
                    valid_count_distribution = handle_count_distribution[
                        valid_handle_rows
                    ]
                    valid_target_counts = target_handle_cardinality[
                        valid_handle_rows
                    ]
                    target_count_probabilities = valid_count_distribution.gather(
                        1,
                        valid_target_counts.unsqueeze(1),
                    ).squeeze(1)
                    count_entropy = -(
                        valid_count_distribution
                        * valid_count_distribution.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                    count_reliability = event_count_reliability_tensor(
                        valid_count_distribution
                    )
                    predicted_handle_cardinality = handle_exist_probabilities.sum(
                        dim=1
                    )
                    event_handle_gold_count_probability_metric = (
                        target_count_probabilities.detach().mean().item()
                    )
                    event_handle_count_entropy_metric = (
                        count_entropy.detach().mean().item()
                    )
                    event_handle_count_reliability_metric = (
                        count_reliability.detach().mean().item()
                    )
                    event_handle_expected_count_mae_metric = (
                        (
                            predicted_handle_cardinality[valid_handle_rows].detach()
                            - valid_target_counts.float()
                        )
                        .abs()
                        .mean()
                        .item()
                    )
                handle_exist_predictions = event_handle_result["predicted_active"]
                event_handle_exist_accuracy_metric = (
                    handle_exist_predictions[valid_handle_entries]
                    == valid_gold_handles[valid_handle_entries]
                ).float().mean().detach().item()
                if valid_gold_handles.any():
                    positive_probabilities = handle_exist_probabilities[valid_gold_handles]
                    event_handle_exist_pos_probability_metric = (
                        positive_probabilities.mean().detach().item()
                    )
                    event_handle_conf_ge30_metric = (
                        positive_probabilities
                        .ge(0.30)
                        .float()
                        .mean()
                        .detach()
                        .item()
                    )
                    event_handle_conf_ge50_metric = (
                        positive_probabilities
                        .ge(0.50)
                        .float()
                        .mean()
                        .detach()
                        .item()
                    )
                    event_handle_control_recall_metric = (
                        positive_probabilities
                        .ge(0.75)
                        .float()
                        .mean()
                        .detach()
                        .item()
                    )
                negative_gold_handles = valid_handle_entries & ~valid_gold_handles
                if negative_gold_handles.any():
                    event_handle_exist_neg_probability_metric = (
                        handle_exist_probabilities[negative_gold_handles]
                        .mean()
                        .detach()
                        .item()
                    )
                if gold_handle_counts is not None:
                    event_handle_count_accuracy_metric = (
                        event_handle_result["dynamic_counts"][valid_handle_rows].long()
                        == gold_handle_counts[valid_handle_rows].clamp(0, self.max_event_count)
                    ).float().mean().detach().item()
                if valid_gold_handles is not None and valid_gold_handles.any():
                    handle_anchor_logits = event_handle_result["handle_anchor_logits"]
                    handle_predictions = handle_anchor_logits.argmax(dim=-1)
                    handle_anchor_losses = []
                    if valid_anchor_handles.any():
                        handle_anchor_losses.append(
                            (
                                float(self.event_handle_match_anchor_cost),
                                F.cross_entropy(
                                    handle_anchor_logits[valid_anchor_handles],
                                    handle_targets[valid_anchor_handles],
                                ),
                            )
                        )
                        event_handle_anchor_accuracy_metric = (
                            handle_predictions[valid_anchor_handles]
                            == handle_targets[valid_anchor_handles]
                        ).float().mean().detach().item()

                    if handle_anchor_losses:
                        grounding_weight = sum(weight for weight, _loss in handle_anchor_losses)
                        event_handle_anchor_loss_total = sum(
                            weight * loss for weight, loss in handle_anchor_losses
                        ) / max(grounding_weight, 1e-6)

            if event_handle_plan is not None:
                event_handle_plan = event_handle_plan.to(
                    device=latent_embd.device,
                    dtype=latent_embd.dtype,
                )
                event_handle_memory = event_handle_plan
                if event_handle_active_mask is not None:
                    event_handle_active_mask = event_handle_active_mask.to(device=event_handle_plan.device).bool()
                handle_attention = None
                if reasoning_attention_mask is not None:
                    if event_handle_active_mask is not None:
                        handle_attention = event_handle_active_mask
                    else:
                        handle_attention = torch.ones(
                            (event_handle_plan.size(0), event_handle_plan.size(1)),
                            dtype=torch.bool,
                            device=event_handle_plan.device,
                        )
                if use_kv_cache_rollout:
                    handle_outputs = self.backbone_forward(
                        inputs_embeds=event_handle_plan,
                        use_cache=True,
                        output_hidden_states=False,
                        past_key_values=past_key_values,
                        attention_mask=(
                            torch.cat((reasoning_attention_mask, handle_attention), dim=1)
                            if reasoning_attention_mask is not None
                            else None
                        ),
                    )
                    past_key_values = self.normalize_past_key_values(handle_outputs.past_key_values)
                else:
                    rollout_prefix_embeds = torch.cat((rollout_prefix_embeds, event_handle_plan), dim=1)
                if reasoning_attention_mask is not None:
                    reasoning_attention_mask = torch.cat((reasoning_attention_mask, handle_attention), dim=1)
                    rollout_attention_mask = reasoning_attention_mask

        if process_ref_ranges is not None:
            process_ref_ranges = process_ref_ranges.to(encoder_input_ids.device)

        if process_ref_ranges is None:
            raise ValueError("Process self-distillation requires completed-stage ranges.")

        with torch.no_grad():
            if self.uses_compact_distill_layers():
                ref_outputs, ref_hidden_states = self.codi_with_distill_hiddens(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                    gather_ranges=process_ref_ranges,
                    return_logits=False,
                )
            else:
                ref_outputs = self.backbone_forward(
                    input_ids=ref_input_ids,
                    output_hidden_states=True,
                    attention_mask=ref_attention_mask,
                )
                ref_hidden_states = tuple(
                    self.pool_hidden_ranges(layer_hidden, process_ref_ranges)
                    for layer_hidden in ref_outputs.hidden_states
                )
        ref_compact_hidden_states = tuple(hidden.detach() for hidden in ref_hidden_states)
        del ref_outputs, ref_hidden_states
      
        num_latent = self.num_latent
        if num_latent > 0:
            if self.adaptive_latent and latent_target_steps is not None:
                target_exit_steps = latent_target_steps.to(encoder_input_ids.device).long().clamp(1, num_latent)
            else:
                target_exit_steps = torch.full(
                    (encoder_input_ids.size(0),),
                    num_latent,
                    dtype=torch.long,
                    device=encoder_input_ids.device,
                )
            decoder_exit_steps, decoder_exit_mix_mask = select_decoder_exit_steps(
                target_exit_steps,
                budget_logits,
                training=self.training,
                mix_enable=self.decoder_exit_mix_enable,
                mix_prob=self.decoder_exit_mix_prob,
                mix_warmup_ratio=self.decoder_exit_mix_warmup_ratio,
                mix_min_step=self.decoder_exit_mix_min_step,
                step_ratio=step_ratio,
                num_latent=num_latent,
            )
            decoder_exit_mix_rate_metric = (
                decoder_exit_mix_mask.detach().float().mean().item()
                if decoder_exit_mix_mask.numel() > 0
                else 0.0
            )
            decoder_exit_gap_metric = (
                (decoder_exit_steps.detach().float() - target_exit_steps.detach().float())
                .abs()
                .mean()
                .item()
            )
            decoder_exit_used_metric = decoder_exit_steps.detach().float().mean().item()

        def decoder_exit_losses(decode_past_key_values, exit_mask, latent_steps_so_far, bridge_hidden=None):
            nonlocal decoder_bridge_delta_sum_metric
            nonlocal decoder_bridge_gate_sum_metric
            nonlocal decoder_bridge_sample_count_metric
            nonlocal json_prefix_robust_drop_sum_metric
            nonlocal json_prefix_robust_candidate_sum_metric
            nonlocal json_prefix_value_drop_sum_metric
            nonlocal json_prefix_value_candidate_sum_metric
            nonlocal json_prefix_key_drop_sum_metric
            nonlocal json_prefix_key_candidate_sum_metric
            nonlocal json_prefix_event_start_drop_sum_metric
            nonlocal json_prefix_event_start_candidate_sum_metric
            nonlocal json_start_continuation_drop_sum_metric
            nonlocal json_start_continuation_candidate_sum_metric
            if not exit_mask.any():
                return (
                    zero_loss.clone(),
                    zero_loss.clone(),
                    zero_loss.clone(),
                    None,
                )

            embds = self.embed_input_ids(decoder_input_ids)
            bridge_embds, bridge_out_gate, bridge_token_mask = self.build_decoder_bridge_embeds(
                bridge_hidden,
                event_handle_memory,
                event_handle_active_mask,
                return_gate=True,
                return_mask=True,
            )
            if bridge_embds is not None and bridge_hidden is not None:
                bridge_delta = (
                    bridge_embds.float() - bridge_hidden.float().unsqueeze(1)
                ).norm(dim=-1) / bridge_hidden.float().norm(dim=-1).clamp_min(1e-6).unsqueeze(1)
                if bridge_token_mask is not None:
                    delta_mask = bridge_token_mask.to(device=bridge_delta.device).bool()
                else:
                    delta_mask = torch.ones_like(bridge_delta, dtype=torch.bool)
                bridge_delta = (
                    bridge_delta.masked_fill(~delta_mask, 0.0).sum(dim=1)
                    / delta_mask.float().sum(dim=1).clamp_min(1.0)
                )
                decoder_bridge_delta_sum_metric = (
                    decoder_bridge_delta_sum_metric
                    + bridge_delta[exit_mask].detach().sum()
                )
                if bridge_out_gate is not None:
                    decoder_bridge_gate_sum_metric = (
                        decoder_bridge_gate_sum_metric
                        + bridge_out_gate.squeeze(-1)[exit_mask].detach().float().sum()
                    )
                decoder_bridge_sample_count_metric = (
                    decoder_bridge_sample_count_metric
                    + exit_mask.detach().float().sum()
                )
            bridge_summary_for_replacement = None

            def get_bridge_summary_for_replacement():
                nonlocal bridge_summary_for_replacement
                if bridge_summary_for_replacement is not None:
                    return bridge_summary_for_replacement
                if bridge_token_mask is not None:
                    bridge_weights = bridge_token_mask.to(device=embds.device).float()
                    bridge_summary_for_replacement = (
                        bridge_embds.to(device=embds.device).float()
                        * bridge_weights.unsqueeze(-1)
                    ).sum(dim=1) / bridge_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                else:
                    bridge_summary_for_replacement = bridge_embds.to(device=embds.device).float().mean(dim=1)
                return bridge_summary_for_replacement

            start_cont_drop_mask = None
            start_cont_prob = scheduled_json_start_continuation_prob()
            if (
                self.training
                and start_cont_prob > 0.0
                and decoder_start_continuation_mask is not None
                and bridge_hidden is not None
                and bridge_embds is not None
            ):
                start_mask = decoder_start_continuation_mask.to(device=embds.device).bool()
                if start_mask.shape != decoder_input_ids.shape:
                    start_mask = start_mask[:, : decoder_input_ids.size(1)]
                    if start_mask.size(1) < decoder_input_ids.size(1):
                        pad_width = decoder_input_ids.size(1) - start_mask.size(1)
                        start_mask = F.pad(start_mask, (0, pad_width), value=False)
                start_mask = start_mask & labels.ne(-100).to(device=embds.device)
                start_mask = start_mask & exit_mask.to(device=embds.device).view(-1, 1)
                if bridge_out_gate is not None:
                    min_gate = float(max(0.0, min(1.0, getattr(
                        self.training_args,
                        "json_start_continuation_min_bridge_gate",
                        0.0,
                    ))))
                    gate_mask = bridge_out_gate.squeeze(-1).to(device=embds.device).float() >= min_gate
                    start_mask = start_mask & gate_mask.view(-1, 1)
                candidate_count = start_mask.float().sum()
                if candidate_count > 0:
                    sample_mask = (
                        torch.rand(
                            (start_mask.size(0), 1),
                            device=embds.device,
                        ) < float(start_cont_prob)
                    )
                    drop_mask = start_mask & sample_mask
                    start_cont_drop_mask = drop_mask
                    if drop_mask.any():
                        bridge_summary = get_bridge_summary_for_replacement()
                        bridge_replacement = bridge_summary.to(dtype=embds.dtype).unsqueeze(1).expand_as(embds)
                        embds = torch.where(
                            drop_mask.unsqueeze(-1),
                            bridge_replacement,
                            embds,
                        )
                    json_start_continuation_drop_sum_metric = (
                        json_start_continuation_drop_sum_metric
                        + drop_mask.float().sum().detach()
                    )
                    json_start_continuation_candidate_sum_metric = (
                        json_start_continuation_candidate_sum_metric
                        + candidate_count.detach()
                    )
            value_prob = scheduled_json_prefix_value_dropout()
            key_prob = (
                scheduled_json_prefix_dropout(getattr(self.training_args, "json_prefix_key_dropout", 0.0))
                if bool(getattr(self.training_args, "json_prefix_key_perturb", False))
                else 0.0
            )
            event_start_prob = (
                scheduled_json_prefix_dropout(getattr(self.training_args, "json_prefix_event_start_dropout", 0.0))
                if bool(getattr(self.training_args, "json_prefix_event_start_perturb", False))
                else 0.0
            )
            if (
                self.training
                and max(value_prob, key_prob, event_start_prob) > 0.0
                and bridge_hidden is not None
                and bridge_embds is not None
            ):
                def align_prefix_mask(mask, *, include_left_context=False):
                    if mask is None:
                        return None
                    aligned = mask.to(device=embds.device).bool()
                    if aligned.shape != decoder_input_ids.shape:
                        aligned = aligned[:, : decoder_input_ids.size(1)]
                        if aligned.size(1) < decoder_input_ids.size(1):
                            pad_width = decoder_input_ids.size(1) - aligned.size(1)
                            aligned = F.pad(aligned, (0, pad_width), value=False)
                    if include_left_context:
                        left_context = torch.zeros_like(aligned)
                        left_context[:, :-1] = aligned[:, 1:]
                        aligned = aligned | left_context
                    aligned = aligned & labels.ne(-100).to(device=embds.device)
                    aligned = aligned & exit_mask.to(device=embds.device).view(-1, 1)
                    return aligned

                value_mask = align_prefix_mask(
                    decoder_prefix_robust_mask,
                    include_left_context=True,
                )
                key_mask = align_prefix_mask(decoder_prefix_key_robust_mask)
                event_start_mask = align_prefix_mask(decoder_prefix_event_start_robust_mask)
                partition_enabled = bool(
                    getattr(self.training_args, "json_perturb_partition", True)
                )
                value_mask = exclude_sampled_json_perturbations(
                    value_mask,
                    start_cont_drop_mask,
                    enabled=partition_enabled,
                )
                key_mask = exclude_sampled_json_perturbations(
                    key_mask,
                    start_cont_drop_mask,
                    enabled=partition_enabled,
                )
                event_start_mask = exclude_sampled_json_perturbations(
                    event_start_mask,
                    start_cont_drop_mask,
                    enabled=partition_enabled,
                )

                def sample_mask(mask, prob):
                    if mask is None or prob <= 0.0:
                        empty = torch.zeros_like(embds[..., 0], dtype=torch.bool)
                        return empty, zero_loss.clone(), zero_loss.clone()
                    candidate_count = mask.float().sum()
                    if candidate_count <= 0:
                        empty = torch.zeros_like(mask, dtype=torch.bool)
                        return empty, zero_loss.clone(), candidate_count.detach()
                    sampled = (torch.rand(mask.shape, device=embds.device) < float(prob)) & mask
                    return sampled, sampled.float().sum().detach(), candidate_count.detach()

                event_start_drop_mask, event_start_drop_count, event_start_candidate_count = sample_mask(
                    event_start_mask,
                    event_start_prob,
                )
                key_mask = exclude_sampled_json_perturbations(
                    key_mask,
                    event_start_drop_mask,
                    enabled=partition_enabled,
                )
                key_drop_mask, key_drop_count, key_candidate_count = sample_mask(key_mask, key_prob)
                value_mask = exclude_sampled_json_perturbations(
                    value_mask,
                    event_start_drop_mask,
                    key_drop_mask,
                    enabled=partition_enabled,
                )
                value_drop_mask, value_drop_count, value_candidate_count = sample_mask(value_mask, value_prob)
                candidate_union = None
                for mask in (value_mask, key_mask, event_start_mask):
                    if mask is None:
                        continue
                    candidate_union = mask if candidate_union is None else (candidate_union | mask)
                drop_mask = value_drop_mask | key_drop_mask | event_start_drop_mask
                candidate_count = (
                    candidate_union.float().sum()
                    if candidate_union is not None
                    else zero_loss.clone()
                )
                if candidate_count > 0:
                    if drop_mask.any():
                        bridge_summary = get_bridge_summary_for_replacement()
                        bridge_replacement = bridge_summary.to(dtype=embds.dtype).unsqueeze(1).expand_as(embds)
                        embds = torch.where(
                            drop_mask.unsqueeze(-1),
                            bridge_replacement,
                            embds,
                        )
                    json_prefix_value_drop_sum_metric = (
                        json_prefix_value_drop_sum_metric + value_drop_count
                    )
                    json_prefix_value_candidate_sum_metric = (
                        json_prefix_value_candidate_sum_metric + value_candidate_count
                    )
                    json_prefix_key_drop_sum_metric = (
                        json_prefix_key_drop_sum_metric + key_drop_count
                    )
                    json_prefix_key_candidate_sum_metric = (
                        json_prefix_key_candidate_sum_metric + key_candidate_count
                    )
                    json_prefix_event_start_drop_sum_metric = (
                        json_prefix_event_start_drop_sum_metric + event_start_drop_count
                    )
                    json_prefix_event_start_candidate_sum_metric = (
                        json_prefix_event_start_candidate_sum_metric + event_start_candidate_count
                    )
                    json_prefix_robust_drop_sum_metric = (
                        json_prefix_robust_drop_sum_metric
                        + drop_mask.float().sum().detach()
                    )
                    json_prefix_robust_candidate_sum_metric = (
                        json_prefix_robust_candidate_sum_metric
                        + candidate_count.detach()
                    )
            bridge_len = bridge_embds.size(1) if bridge_embds is not None else 0
            decode_attention_mask = None
            bridge_attention_mask = None
            prefix_len = rollout_prefix_embeds.size(1) if not use_kv_cache_rollout else 0
            if self.fix_attn_mask and encoder_attention_mask is not None:
                if bridge_token_mask is not None:
                    bridge_mask = bridge_token_mask.to(device=encoder_attention_mask.device).bool()
                else:
                    bridge_mask = torch.ones(
                        (embds.size(0), bridge_len),
                        dtype=torch.bool,
                        device=encoder_attention_mask.device,
                    )
                decoder_mask = torch.ones(
                    (embds.size(0), embds.size(1)),
                    dtype=torch.bool,
                    device=encoder_attention_mask.device,
                )
                if use_kv_cache_rollout:
                    bridge_attention_mask = torch.cat(
                        (reasoning_attention_mask, bridge_mask),
                        dim=1,
                    )
                    decode_attention_mask = torch.cat(
                        (reasoning_attention_mask, bridge_mask, decoder_mask),
                        dim=1,
                    )
                else:
                    bridge_attention_mask = None
                    decode_attention_mask = torch.cat(
                        (rollout_attention_mask, bridge_mask, decoder_mask),
                        dim=1,
                    )
            decode_embds = embds
            decode_past = (
                self.clone_past_key_values(decode_past_key_values)
                if use_kv_cache_rollout
                else None
            )
            if use_kv_cache_rollout and reasoning_attention_mask is not None:
                cached_length = self.past_key_values_length(decode_past)
                mask_length = reasoning_attention_mask.size(1)
                if cached_length is not None and cached_length != mask_length:
                    raise RuntimeError(
                        "Latent rollout cache/mask mismatch before auxiliary decoding: "
                        f"cache={cached_length}, mask={mask_length}, "
                        f"latent_step={latent_steps_so_far}."
                    )
            if bridge_embds is not None and use_kv_cache_rollout:
                bridge_outputs = self.backbone_forward(
                    inputs_embeds=bridge_embds,
                    use_cache=True,
                    output_hidden_states=False,
                    past_key_values=decode_past,
                    attention_mask=bridge_attention_mask,
                )
                decode_past = self.normalize_past_key_values(bridge_outputs.past_key_values)
            if not use_kv_cache_rollout:
                if bridge_embds is not None:
                    decode_embds = torch.cat((rollout_prefix_embeds, bridge_embds, embds), dim=1)
                else:
                    decode_embds = torch.cat((rollout_prefix_embeds, embds), dim=1)
                decode_past = None
                prefix_len = prefix_len + bridge_len

            exit_outputs = self.backbone_forward(
                inputs_embeds=decode_embds,
                use_cache=use_kv_cache_rollout,
                output_hidden_states=False,
                past_key_values=decode_past,
                attention_mask=decode_attention_mask,
            )
            exit_weight = exit_mask.float().mean().clamp_min(1e-6)

            if use_kv_cache_rollout:
                effective_hidden = exit_outputs.last_hidden_state[:, :-1, :]
            else:
                effective_hidden = exit_outputs.last_hidden_state[:, prefix_len : prefix_len + decoder_input_ids.size(1) - 1, :]
            effective_labels = labels[:, 1:].clone()
            effective_labels[~exit_mask] = -100
            effective_token_weights = decoder_token_weights[:, 1:] if decoder_token_weights is not None else None
            (
                exit_ce_loss,
                exit_ce_base_loss,
                exit_native_boundary_bce,
            ) = sequence_lm_head_cross_entropy(
                effective_hidden,
                effective_labels,
                exit_mask,
                token_weights=effective_token_weights,
                return_base_loss=True,
            )
            update_train_proxy_metrics(
                effective_hidden,
                effective_labels,
                exit_mask,
            )
            return (
                exit_ce_loss * exit_weight,
                exit_ce_base_loss * exit_weight,
                exit_native_boundary_bce * exit_weight,
                None,
            )

        if self.num_latent != 0:
            max_latent_iterations = int(
                torch.maximum(
                    target_exit_steps.max(),
                    decoder_exit_steps.max(),
                )
                .detach()
                .item()
            )
            for i in range(max_latent_iterations):
                # Implicit CoT generation
                if event_handle_memory is not None:
                    latent_embd = self.apply_handle_latent_controller(
                        latent_embd[:, -1, :],
                        event_handle_memory,
                        event_handle_active_mask,
                    ).unsqueeze(1)
                latent_attention_mask = None
                if reasoning_attention_mask is not None:
                    latent_step_mask = torch.ones(
                        (latent_embd.size(0), latent_embd.size(1)),
                        dtype=torch.bool,
                        device=latent_embd.device,
                    )
                    latent_attention_mask = torch.cat((reasoning_attention_mask, latent_step_mask), dim=1)
                if not use_kv_cache_rollout:
                    rollout_prefix_embeds = torch.cat((rollout_prefix_embeds, latent_embd), dim=1)
                    if reasoning_attention_mask is not None:
                        rollout_attention_mask = latent_attention_mask
                    gather_latent_position = torch.full(
                        (rollout_prefix_embeds.size(0),),
                        rollout_prefix_embeds.size(1) - 1,
                        dtype=torch.long,
                        device=encoder_input_ids.device,
                    )
                if self.uses_compact_distill_layers():
                    outputs, process_student_hidden_states = self.codi_with_distill_hiddens(
                        inputs_embeds=latent_embd if use_kv_cache_rollout else rollout_prefix_embeds,
                        use_cache=use_kv_cache_rollout,
                        past_key_values=past_key_values if use_kv_cache_rollout else None,
                        attention_mask=latent_attention_mask if use_kv_cache_rollout else rollout_attention_mask,
                        gather_positions=None if use_kv_cache_rollout else gather_latent_position,
                        return_logits=False,
                    )
                else:
                    outputs = self.backbone_forward(
                        inputs_embeds=latent_embd if use_kv_cache_rollout else rollout_prefix_embeds,
                        use_cache=use_kv_cache_rollout,
                        output_hidden_states=True,
                        past_key_values=past_key_values if use_kv_cache_rollout else None,
                        attention_mask=latent_attention_mask if use_kv_cache_rollout else rollout_attention_mask,
                    )
                    process_student_hidden_states = outputs.hidden_states
                past_key_values = self.normalize_past_key_values(outputs.past_key_values) if use_kv_cache_rollout else None
                if latent_attention_mask is not None:
                    reasoning_attention_mask = latent_attention_mask
                latent_hidden = process_student_hidden_states[-1][:, -1, :]

                if process_ref_ranges is not None and process_ref_ranges.numel() > 0:
                    stage_count = process_ref_ranges.size(1)
                    stage_targets = self.latent_stage_targets(
                        i,
                        latent_target_steps,
                        stage_count,
                        process_ref_ranges.size(0),
                        encoder_input_ids.device,
                    )
                    stage_ranges = process_ref_ranges.gather(
                        1,
                        stage_targets.unsqueeze(1).unsqueeze(-1).expand(-1, 1, 2),
                    ).squeeze(1)
                    valid_stage = (stage_ranges[:, 1] > stage_ranges[:, 0]) & (i < target_exit_steps)
                    if valid_stage.any():
                        process_distill_loss = zero_loss.clone()
                        for out, ref_out in zip(process_student_hidden_states, ref_compact_hidden_states):
                            ref_selected = self.select_process_teacher_hidden(
                                ref_out,
                                i,
                                latent_target_steps,
                                stage_count,
                                process_ref_ranges.size(0),
                                encoder_input_ids.device,
                            )
                            out_selected = out[:, -1:, :]
                            student_distill_hidden = out_selected[valid_stage].float()
                            teacher_distill_hidden = ref_selected[valid_stage].detach().float()
                            if self.process_distill_normalize_hidden:
                                student_distill_hidden = F.normalize(student_distill_hidden, dim=-1)
                                teacher_distill_hidden = F.normalize(teacher_distill_hidden, dim=-1)
                                process_distill_loss = process_distill_loss + (
                                    1.0
                                    - F.cosine_similarity(
                                        student_distill_hidden,
                                        teacher_distill_hidden,
                                        dim=-1,
                                    ).mean()
                                )
                            else:
                                process_distill_loss = process_distill_loss + F.mse_loss(
                                    student_distill_hidden,
                                    teacher_distill_hidden,
                                )
                        process_distill_loss = process_distill_loss / len(process_student_hidden_states)
                        process_distill_loss_total = process_distill_loss_total + process_distill_loss
                        process_supervision_steps += 1

                latent_embd = latent_hidden.unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                exit_mask = decoder_exit_steps == (i + 1)
                (
                    exit_ce_loss,
                    exit_ce_base_loss,
                    exit_native_boundary_bce,
                    exit_logits,
                ) = decoder_exit_losses(
                    past_key_values,
                    exit_mask,
                    i + 1,
                    bridge_hidden=latent_hidden,
                )
                ce_loss_total = ce_loss_total + exit_ce_loss
                ce_loss_for_gate_total = ce_loss_for_gate_total + exit_ce_base_loss
                native_boundary_bce_total = (
                    native_boundary_bce_total + exit_native_boundary_bce
                )
                if exit_logits is not None:
                    logits = exit_logits


        # Weigh the two retained auxiliary objectives.
        if process_supervision_steps > 0:
            process_distill_loss_total = process_distill_loss_total / process_supervision_steps

        process_distill_loss_total *= self.process_distill_loss_factor
        complexity_loss_total *= self.latent_budget_loss_factor
        event_handle_anchor_loss_total *= self.event_handle_anchor_loss_factor
        event_handle_exist_loss_total *= self.event_handle_exist_loss_factor

        reason_loss = (
            process_distill_loss_total
            + complexity_loss_total
        )
        instance_loss = (
            event_handle_anchor_loss_total
            + event_handle_exist_loss_total
        )
        native_boundary_loss = zero_loss.clone()
        if (
            self.native_boundary_loss_enable
            and self.native_boundary_loss_factor > 0.0
        ):
            native_boundary_loss = (
                native_boundary_bce_total
                * float(self.native_boundary_loss_factor)
                * float(current_ce_token_weight_alpha())
            )
        gate_threshold = float(max(getattr(self, "auxiliary_loss_gate_threshold", 1.0), 1e-6))
        gate_ce_loss = ce_loss_for_gate_total if full_objective_sample_weighting else ce_loss_total
        auxiliary_gate = (
            gate_threshold / gate_ce_loss.detach().float().clamp_min(1e-6)
        ).clamp(0.0, 1.0).to(device=ce_loss_total.device, dtype=ce_loss_total.dtype)
        loss = ce_loss_total + native_boundary_loss + auxiliary_gate * (
            reason_loss + instance_loss
        )
        sample_loss_weight_metric = 1.0
        if sample_loss_weights is not None:
            sample_weights = sample_loss_weights.to(loss.device).float().clamp(
                float(self.sample_loss_weight_min),
                float(self.sample_loss_weight_max),
            )
            sample_loss_weight_metric = sample_weights.detach().mean().float().item()
            if full_objective_sample_weighting:
                loss = loss * sample_weights.to(loss.dtype).mean()

        def to_float(value):
            if isinstance(value, torch.Tensor):
                return value.detach().float().item()
            return float(value)

        def proxy_score(mask_name, *, relaxed=False):
            count = train_proxy_counts[mask_name].detach().float()
            hits = (
                train_proxy_topk_hits[mask_name]
                if relaxed
                else train_proxy_exact_hits[mask_name]
            ).detach().float()
            if count.item() <= 0:
                return 0.0
            return to_float(hits / count.clamp_min(1.0))

        train_proxy_outputs = {}
        strict_proxy_specs = {
            "trigger": "trigger",
            "arg_i": "arg_i",
            "arg_c": "arg_c",
            "span": "span",
            "head": "head",
        }
        for output_name, mask_name in strict_proxy_specs.items():
            score = proxy_score(mask_name, relaxed=False)
            train_proxy_outputs[f"train_proxy_{output_name}_p"] = score
            train_proxy_outputs[f"train_proxy_{output_name}_r"] = score
            train_proxy_outputs[f"train_proxy_{output_name}_f1"] = score
        relaxed_proxy_specs = {
            "soft_trigger": "trigger",
            "relaxed_span": "span",
            "normalized_arg_i": "arg_i",
            "normalized_arg_c": "arg_c",
            "partial_argument": "head",
        }
        for output_name, mask_name in relaxed_proxy_specs.items():
            score = proxy_score(mask_name, relaxed=True)
            train_proxy_outputs[f"train_proxy_{output_name}_p"] = score
            train_proxy_outputs[f"train_proxy_{output_name}_r"] = score
            train_proxy_outputs[f"train_proxy_{output_name}_f1"] = score

        outputs = {
            "loss": loss,
            "logits": logits,
            "ce_loss": to_float(ce_loss_total),
            "native_boundary_loss": to_float(native_boundary_loss),
            "native_boundary_bce": to_float(native_boundary_bce_total),
            "process_distill_loss": to_float(process_distill_loss_total),
            "complexity_loss": to_float(complexity_loss_total),
            "reason_loss": to_float(reason_loss),
            "instance_loss": to_float(instance_loss),
            "event_handle_anchor_loss": to_float(event_handle_anchor_loss_total),
            "event_handle_exist_loss": to_float(event_handle_exist_loss_total),
            "event_handle_exist_bce_loss": event_handle_exist_bce_loss_metric,
            "event_handle_count_nll_loss": event_handle_count_nll_loss_metric,
            "event_handle_gold_count_probability": (
                event_handle_gold_count_probability_metric
            ),
            "event_handle_count_entropy": event_handle_count_entropy_metric,
            "event_handle_count_reliability": event_handle_count_reliability_metric,
            "event_handle_expected_count_mae": (
                event_handle_expected_count_mae_metric
            ),
            "auxiliary_loss_gate": to_float(auxiliary_gate),
            "ce_token_weight_alpha": float(ce_token_weight_alpha_metric),
            "predicted_complexity": predicted_complexity_metric,
            "latent_budget_acc": latent_budget_accuracy_metric,
            "latent_budget_top1_mean": latent_budget_top1_mean_metric,
            "latent_budget_expected_mean": latent_budget_expected_mean_metric,
            "latent_budget_expected_mae": latent_budget_expected_mae_metric,
            "latent_budget_proposal_acc": latent_budget_proposal_accuracy_metric,
            "latent_budget_proposal_mae": latent_budget_proposal_mae_metric,
            "latent_budget_entropy": latent_budget_entropy_metric,
            "latent_budget_top1_margin": latent_budget_top1_margin_metric,
            "latent_budget_ce_loss": latent_budget_ce_loss_metric,
            "latent_budget_ordinal_loss": latent_budget_ordinal_loss_metric,
            "event_handle_anchor_acc": event_handle_anchor_accuracy_metric,
            "event_handle_exist_acc": event_handle_exist_accuracy_metric,
            "event_handle_exist_pos_probability": event_handle_exist_pos_probability_metric,
            "event_handle_exist_neg_probability": event_handle_exist_neg_probability_metric,
            "event_handle_conf_ge30": event_handle_conf_ge30_metric,
            "event_handle_conf_ge50": event_handle_conf_ge50_metric,
            "event_handle_conf_ge75": event_handle_control_recall_metric,
            "event_handle_control_recall": event_handle_control_recall_metric,
            "event_handle_count_acc": event_handle_count_accuracy_metric,
            "event_handle_query_seed_guard_rate": (
                event_handle_query_seed_guard_rate_metric
            ),
            "decoder_bridge_out_gate": to_float(
                decoder_bridge_gate_sum_metric
                / decoder_bridge_sample_count_metric.clamp_min(1.0)
            ),
            "decoder_bridge_delta": to_float(
                decoder_bridge_delta_sum_metric
                / decoder_bridge_sample_count_metric.clamp_min(1.0)
            ),
            "json_prefix_robust_rate": to_float(
                json_prefix_robust_drop_sum_metric
                / json_prefix_robust_candidate_sum_metric.clamp_min(1.0)
            ),
            "json_prefix_value_robust_rate": to_float(
                json_prefix_value_drop_sum_metric
                / json_prefix_value_candidate_sum_metric.clamp_min(1.0)
            ),
            "json_prefix_key_robust_rate": to_float(
                json_prefix_key_drop_sum_metric
                / json_prefix_key_candidate_sum_metric.clamp_min(1.0)
            ),
            "json_prefix_event_start_robust_rate": to_float(
                json_prefix_event_start_drop_sum_metric
                / json_prefix_event_start_candidate_sum_metric.clamp_min(1.0)
            ),
            "json_start_continuation_rate": to_float(
                json_start_continuation_drop_sum_metric
                / json_start_continuation_candidate_sum_metric.clamp_min(1.0)
            ),
            "boundary_active_rate": to_float(
                boundary_active_count_metric
                / boundary_candidate_count_metric.clamp_min(1.0)
            ),
            "boundary_prior_confidence": to_float(
                boundary_reliability_sum_metric
                / boundary_active_count_metric.clamp_min(1.0)
            ),
            "boundary_gold_continue_rate": to_float(
                boundary_continue_count_metric
                / boundary_active_count_metric.clamp_min(1.0)
            ),
            "boundary_gold_stop_rate": to_float(
                (boundary_active_count_metric - boundary_continue_count_metric)
                / boundary_active_count_metric.clamp_min(1.0)
            ),
            "boundary_prior_agreement": to_float(
                boundary_agreement_sum_metric
                / boundary_agreement_weight_metric.clamp_min(1e-6)
            ),
            "boundary_qwen_continue": to_float(
                boundary_qwen_continue_sum_metric
                / boundary_active_count_metric.clamp_min(1.0)
            ),
            "boundary_blended_continue": to_float(
                boundary_blended_continue_sum_metric
                / boundary_active_count_metric.clamp_min(1.0)
            ),
            "native_boundary_coverage": to_float(
                native_boundary_valid_count_metric
                / native_boundary_gold_count_metric.clamp_min(1.0)
            ),
            "native_boundary_commit_rate": to_float(
                native_boundary_proposal_count_metric
                / native_boundary_valid_count_metric.clamp_min(1.0)
            ),
            "native_boundary_choice_acc": to_float(
                native_boundary_correct_count_metric
                / native_boundary_valid_count_metric.clamp_min(1.0)
            ),
            "native_continue_gold_continue": to_float(
                native_boundary_continue_probability_on_continue_sum_metric
                / native_boundary_continue_target_count_metric.clamp_min(1.0)
            ),
            "native_continue_gold_stop": to_float(
                native_boundary_continue_probability_on_stop_sum_metric
                / native_boundary_stop_target_count_metric.clamp_min(1.0)
            ),
            "native_boundary_proposal_failure_rate": to_float(
                torch.where(
                    native_boundary_valid_count_metric.gt(0),
                    1.0
                    - (
                        native_boundary_proposal_count_metric
                        / native_boundary_valid_count_metric.clamp_min(1.0)
                    ),
                    zero_loss,
                )
            ),
            "native_boundary_low_count_acc": to_float(
                native_boundary_low_count_correct_metric
                / native_boundary_low_count_total_metric.clamp_min(1.0)
            ),
            "native_boundary_high_count_acc": to_float(
                native_boundary_high_count_correct_metric
                / native_boundary_high_count_total_metric.clamp_min(1.0)
            ),
            "native_boundary_first_acc": to_float(
                native_boundary_first_correct_metric
                / native_boundary_first_total_metric.clamp_min(1.0)
            ),
            "native_boundary_later_acc": to_float(
                native_boundary_later_correct_metric
                / native_boundary_later_total_metric.clamp_min(1.0)
            ),
            "decoder_exit_mix_rate": float(decoder_exit_mix_rate_metric),
            "decoder_exit_gap": float(decoder_exit_gap_metric),
            "decoder_exit_used": float(decoder_exit_used_metric),
            "sample_loss_weight": sample_loss_weight_metric,
        }
        for budget_step, rate in latent_budget_target_hist_metrics.items():
            outputs[f"latent_budget_target_t{budget_step}_rate"] = rate
        for budget_step, rate in latent_budget_proposal_hist_metrics.items():
            outputs[f"latent_budget_proposal_t{budget_step}_rate"] = rate
        outputs.update(train_proxy_outputs)
        return outputs
