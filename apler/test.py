#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import re
import os
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from accelerate.utils import set_seed


from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    blend_event_boundary_logits,
    conditional_event_continue_probability_tensor,
    event_boundary_proposal_mask,
    event_count_distribution_tensor,
    event_count_reliability_tensor,
    latent_budget_statistics,
    load_checkpoint_into_model,
    load_tokenizer_with_fallback,
)
from src.event_utils import (
    FINAL_JSON_MARKER,
    JSON_EVENT_ARRAY_PREFIX,
    add_standard_metric_counts,
    add_relaxed_metric_counts,
    build_document_token_mask_from_source,
    build_sentence_rep_positions_from_source,
    build_event_prompt,
    canonical_events,
    deduplicate_exact_events,
    event_generation_prefix_reserve,
    empty_standard_metric_counts,
    empty_relaxed_metric_counts,
    encode_json_event_array_prefix,
    event_object_start_candidate_token_ids,
    event_payload_quality_metrics,
    example_complexity,
    extract_json_payload,
    find_first_complete_json_end,
    finalize_standard_metric_counts,
    finalize_relaxed_metric_counts,
    format_prompt_for_model,
    has_semantically_complete_event_payload,
    is_extractable_event_payload,
    filter_positive_event_examples,
    limit_dataset_for_debug,
    load_emergency_examples,
    relaxed_event_metrics,
    standard_event_metrics,
    strip_bos,
    event_tuples,
    require_data_name,
    resolve_event_task,
    tuple_f1,
)
from src.event_decode_state import (
    _decode_single_token,
    _event_boundary_piece_kind,
    at_completed_event_boundary,
    classify_event_token_transition,
    direct_event_boundary_token_groups,
    event_boundary_token_groups,
    event_count_distribution,
    event_count_reliability,
    inspect_event_decode_state,
    normalize_event_boundary_control,
)
from src.eval_report_utils import (
    build_machine_readable_evaluation_payload,
    canonical_record_sha256,
    is_incomplete_json_generation,
    machine_metric_record,
    write_json_atomically,
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def resolve_lora_target_modules(model_name_or_path: str, override: str):
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


def build_lora_config(model_args, training_args):
    if not training_args.use_lora:
        return None
    if not model_args.lora_init:
        if model_args.adapter_name_or_path:
            return None
        raise ValueError(
            "Evaluation with --use_lora True requires --lora_init True or "
            "--adapter_name_or_path PATH."
        )

    task_type = TaskType.CAUSAL_LM
    target_modules = resolve_lora_target_modules(
        model_args.model_name_or_path,
        model_args.lora_target_modules,
    )

    return LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def sample_next_token(logits, greedy, temperature=0.1, top_k=40, top_p=0.95):
    if greedy:
        return torch.argmax(logits, dim=-1).squeeze(-1)

    temperature = float(temperature)
    top_k = int(top_k)
    top_p = float(top_p)
    if temperature <= 0:
        raise ValueError("Sampling temperature must be greater than zero.")
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise RuntimeError("Cannot sample from a logits row with no finite values.")

    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    if top_k > 1:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < min_top_k_value, -float("inf"))

    if top_p < 1.0:
        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove.any():
            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
            sorted_indices_to_remove[:, 0] = False
        for b in range(logits.size(0)):
            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def last_token_logits(logits, vocab_size):
    if logits.dim() == 3:
        return logits[:, -1, :vocab_size]
    if logits.dim() == 2:
        if logits.size(0) == 1:
            return logits[:, :vocab_size]
        return logits[-1:, :vocab_size]
    if logits.dim() == 1:
        return logits.unsqueeze(0)[:, :vocab_size]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def has_complete_json(text):
    start = text.find("{")
    return start >= 0 and find_first_complete_json_end(text, start) >= 0


def has_nonempty_event_payload(text):
    payload = extract_json_payload(text)
    return has_semantically_complete_event_payload(payload)


def should_stop_on_complete_json(text, training_args):
    if not has_complete_json(text):
        return False
    if not bool(getattr(training_args, "semantic_json_stop", True)):
        return True
    return has_nonempty_event_payload(text)


def decoded_event_object_count(text):
    state = inspect_event_decode_state(text)
    return state.completed_event_count + int(state.event_object_open)


def blend_boundary_choice_logits(
    logits,
    continue_ids,
    stop_ids,
    handle_continue_probability,
    handle_reliability,
):
    """Blend continue/stop conditionally while preserving their total logit mass."""
    if not continue_ids or not stop_ids:
        return logits.clone(), None, None
    continue_tensor = torch.tensor(
        [list(continue_ids)],
        dtype=torch.long,
        device=logits.device,
    )
    stop_tensor = torch.tensor(
        [list(stop_ids)],
        dtype=torch.long,
        device=logits.device,
    )
    handle_probability = (
        0.5
        if handle_continue_probability is None
        else float(handle_continue_probability)
    )
    reliability = (
        0.0
        if handle_continue_probability is None
        else float(handle_reliability)
    )
    controlled, qwen_continue, blended_continue, valid_rows = (
        blend_event_boundary_logits(
            logits,
            continue_tensor,
            stop_tensor,
            torch.tensor(
                [handle_probability],
                dtype=torch.float,
                device=logits.device,
            ),
            torch.tensor(
                [reliability],
                dtype=torch.float,
                device=logits.device,
            ),
        )
    )
    if not bool(valid_rows[0]):
        return controlled, None, None
    return controlled, qwen_continue, blended_continue


def qwen_proposes_event_boundary(logits, continue_ids, stop_ids):
    if not continue_ids or not stop_ids:
        return False
    continue_tensor = torch.tensor(
        [list(continue_ids)],
        dtype=torch.long,
        device=logits.device,
    )
    stop_tensor = torch.tensor(
        [list(stop_ids)],
        dtype=torch.long,
        device=logits.device,
    )
    return bool(
        event_boundary_proposal_mask(
            logits,
            continue_tensor,
            stop_tensor,
        )[0].item()
    )


def handle_boundary_statistics(
    handle_exist_probabilities,
    completed_event_count,
):
    """Compute the shared count-survival prior for every boundary mode.

    Hierarchical mode differs only in when a native proposal is committed; it
    must not reinterpret one handle's marginal probability as a sequence hazard.
    """
    if not handle_exist_probabilities:
        return None
    probabilities = torch.tensor(
        [list(handle_exist_probabilities)],
        dtype=torch.float,
    )
    expected_count = probabilities.sum(dim=-1)
    distribution = event_count_distribution_tensor(probabilities)
    continue_probability = conditional_event_continue_probability_tensor(
        distribution,
        torch.tensor([int(completed_event_count)], dtype=torch.long),
    )
    reliability = event_count_reliability_tensor(distribution)
    return {
        "expected_count": float(expected_count[0].item()),
        "reliability": float(reliability[0].item()),
        "continue_probability": float(continue_probability[0].item()),
    }


def apply_handle_event_boundary_control(
    logits,
    tokenizer,
    pred_tokens,
    training_args,
    handle_exist_probabilities=None,
    proposal_logits=None,
):
    """Blend one boundary decision using native proposals and controlled sampling."""

    if proposal_logits is None:
        proposal_logits = logits
    if proposal_logits.shape != logits.shape:
        raise ValueError(
            "Boundary proposal logits must match controlled logits: "
            f"proposal={tuple(proposal_logits.shape)}, "
            f"controlled={tuple(logits.shape)}."
        )
    mode = normalize_event_boundary_control(
        getattr(training_args, "event_boundary_control", "off")
    )
    details = {
        "mode": mode,
        "stage": None,
        "active": False,
        "opportunity": False,
        "hard_cap": False,
        "completed_event_count": 0,
        "direct_continue_candidates": 0,
        "direct_stop_candidates": 0,
        "handle_expected_count": None,
        "handle_reliability": 0.0,
        "handle_continue_probability": None,
        "qwen_continue_probability": None,
        "blended_continue_probability": None,
    }
    if mode == "off":
        return logits, details

    decoded_so_far = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    state = inspect_event_decode_state(decoded_so_far)

    max_event_count = max(1, int(getattr(training_args, "max_event_count", 1)))
    direct_groups = direct_event_boundary_token_groups(
        tokenizer,
        logits.size(-1),
        decoded_so_far,
        state=state,
    )
    direct_continue_ids = direct_groups["continue"]
    direct_stop_ids = direct_groups["stop"]
    completed_after_current = int(state.completed_event_count) + 1
    if mode == "handle_hierarchical" and (
        direct_continue_ids or direct_stop_ids
    ):
        details["opportunity"] = True
        details["stage"] = "direct_opportunity"
        details["completed_event_count"] = completed_after_current
        details["direct_continue_candidates"] = len(direct_continue_ids)
        details["direct_stop_candidates"] = len(direct_stop_ids)
        raw_top_id = int(proposal_logits.float().argmax(dim=-1)[0].item())
        details["raw_proposal_token_id"] = raw_top_id
        raw_commits_boundary = qwen_proposes_event_boundary(
            proposal_logits,
            direct_continue_ids,
            direct_stop_ids,
        )

        # A native boundary proposal is a structural commitment. Once Qwen's
        # own top token enters this branch, the handle prior may choose between
        # continue and stop, but it must not send decoding back to content.
        if completed_after_current >= max_event_count:
            if not raw_commits_boundary or not direct_stop_ids:
                controlled = logits.clone()
                if direct_continue_ids:
                    controlled[:, list(direct_continue_ids)] = -float("inf")
                return controlled, details
            controlled = torch.full_like(logits, -float("inf"))
            controlled[:, list(direct_stop_ids)] = logits[
                :, list(direct_stop_ids)
            ]
            details["stage"] = "direct_boundary"
            details["active"] = True
            details["hard_cap"] = True
            details["handle_continue_probability"] = 0.0
            details["blended_continue_probability"] = 0.0
            return controlled, details
        if (
            not raw_commits_boundary
            or not direct_continue_ids
            or not direct_stop_ids
        ):
            return logits, details

        details["stage"] = "direct_boundary"
        details["active"] = True
        controlled = logits.clone()

        statistics = handle_boundary_statistics(
            handle_exist_probabilities,
            completed_after_current,
        )
        if statistics is not None:
            details["handle_expected_count"] = statistics["expected_count"]
            details["handle_reliability"] = statistics["reliability"]
            details["handle_continue_probability"] = statistics[
                "continue_probability"
            ]
        controlled, qwen_continue, blended_continue = (
            blend_boundary_choice_logits(
                controlled,
                direct_continue_ids,
                direct_stop_ids,
                details["handle_continue_probability"],
                details["handle_reliability"],
            )
        )
        branch_ids = tuple(
            dict.fromkeys(direct_continue_ids + direct_stop_ids)
        )
        committed = torch.full_like(controlled, -float("inf"))
        committed[:, list(branch_ids)] = controlled[:, list(branch_ids)]
        controlled = committed
        if qwen_continue is not None:
            details["qwen_continue_probability"] = float(
                qwen_continue.detach().cpu()[0]
            )
        if blended_continue is not None:
            details["blended_continue_probability"] = float(
                blended_continue.detach().cpu()[0]
            )
        return controlled, details

    if direct_continue_ids and completed_after_current >= max_event_count:
        controlled = logits.clone()
        controlled[:, list(direct_continue_ids)] = -float("inf")
        details["stage"] = "direct_boundary"
        details["active"] = True
        details["hard_cap"] = True
        details["completed_event_count"] = completed_after_current
        details["direct_continue_candidates"] = len(direct_continue_ids)
        details["direct_stop_candidates"] = len(direct_stop_ids)
        details["handle_continue_probability"] = 0.0
        details["blended_continue_probability"] = 0.0
        return controlled, details

    if direct_continue_ids and direct_stop_ids:
        statistics = handle_boundary_statistics(
            handle_exist_probabilities,
            completed_after_current,
        )
        if statistics is not None:
            details["handle_expected_count"] = statistics["expected_count"]
            details["handle_reliability"] = statistics["reliability"]
            details["handle_continue_probability"] = statistics[
                "continue_probability"
            ]
        controlled, qwen_continue, blended_continue = blend_boundary_choice_logits(
            logits,
            direct_continue_ids,
            direct_stop_ids,
            details["handle_continue_probability"],
            details["handle_reliability"],
        )
        details["stage"] = "direct_boundary"
        details["active"] = True
        details["completed_event_count"] = completed_after_current
        details["direct_continue_candidates"] = len(direct_continue_ids)
        details["direct_stop_candidates"] = len(direct_stop_ids)
        if qwen_continue is not None:
            details["qwen_continue_probability"] = float(
                qwen_continue.detach().cpu()[0]
            )
        if blended_continue is not None:
            details["blended_continue_probability"] = float(
                blended_continue.detach().cpu()[0]
            )
        return controlled, details

    if not at_completed_event_boundary(decoded_so_far):
        return logits, details

    completed_count = int(state.completed_event_count)
    groups = event_boundary_token_groups(tokenizer, logits.size(-1))
    continue_ids = groups["continue"]
    stop_ids = groups["stop"]
    whitespace_ids = groups["whitespace"]
    allowed_ids = tuple(dict.fromkeys(continue_ids + stop_ids + whitespace_ids))

    controlled = torch.full_like(logits, -float("inf"))
    controlled[:, list(allowed_ids)] = logits[:, list(allowed_ids)]
    details["stage"] = "completed_boundary"
    details["completed_event_count"] = completed_count
    raw_top_id = int(proposal_logits.float().argmax(dim=-1)[0].item())
    details["raw_proposal_token_id"] = raw_top_id
    if completed_count >= max_event_count:
        hard_capped = torch.full_like(logits, -float("inf"))
        cap_allowed_ids = (
            tuple(stop_ids)
            if mode == "handle_hierarchical"
            else tuple(dict.fromkeys(stop_ids + whitespace_ids))
        )
        hard_capped[:, list(cap_allowed_ids)] = logits[:, list(cap_allowed_ids)]
        details["active"] = True
        details["hard_cap"] = True
        details["handle_continue_probability"] = 0.0
        details["blended_continue_probability"] = 0.0
        return hard_capped, details
    if mode == "handle_hierarchical":
        details["opportunity"] = True
        if raw_top_id in whitespace_ids:
            details["stage"] = "completed_whitespace"
            return controlled, details
        if not qwen_proposes_event_boundary(
            proposal_logits,
            continue_ids,
            stop_ids,
        ):
            return controlled, details
    details["active"] = True

    statistics = handle_boundary_statistics(
        handle_exist_probabilities,
        completed_count,
    )
    if statistics is not None:
        details["handle_expected_count"] = statistics["expected_count"]
        details["handle_reliability"] = statistics["reliability"]

    if statistics is not None:
        details["handle_continue_probability"] = statistics[
            "continue_probability"
        ]

    controlled, qwen_continue, blended_continue = blend_boundary_choice_logits(
        controlled,
        continue_ids,
        stop_ids,
        details["handle_continue_probability"],
        details["handle_reliability"],
    )
    if mode == "handle_hierarchical":
        branch_ids = tuple(dict.fromkeys(continue_ids + stop_ids))
        committed = torch.full_like(controlled, -float("inf"))
        committed[:, list(branch_ids)] = controlled[:, list(branch_ids)]
        controlled = committed
    if qwen_continue is not None:
        details["qwen_continue_probability"] = float(
            qwen_continue.detach().cpu()[0]
        )
    if blended_continue is not None:
        details["blended_continue_probability"] = float(
            blended_continue.detach().cpu()[0]
        )
    return controlled, details


def classify_event_boundary_token(tokenizer, token_id):
    return _event_boundary_piece_kind(_decode_single_token(tokenizer, token_id))


def constrain_event_root_close_logits(logits, tokenizer, pred_tokens, training_args):
    if normalize_event_boundary_control(
        getattr(training_args, "event_boundary_control", "off")
    ) == "off":
        return logits
    decoded_so_far = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    state = inspect_event_decode_state(decoded_so_far)
    if (
        state.events_array_open
        or state.event_object_open
        or state.completed_event_count < 1
        or has_complete_json(decoded_so_far)
        or not str(decoded_so_far or "").rstrip().endswith("]")
    ):
        return logits

    groups = event_boundary_token_groups(tokenizer, logits.size(-1))
    allowed_ids = tuple(
        dict.fromkeys(groups["root_close"] + groups["whitespace"])
    )
    constrained = torch.full_like(logits, -float("inf"))
    constrained[:, list(allowed_ids)] = logits[:, list(allowed_ids)]
    return constrained


def violates_single_layer_json(text):
    if FINAL_JSON_MARKER in text:
        return True
    if len(re.findall(r'"events"\s*:', text)) > 1:
        return True
    root_start = re.search(r'\{\s*"events"\s*:\s*\[', text)
    if root_start:
        nested_region = text[root_start.end() :]
        if re.search(r'\{\s*"?events"?\s*:?$', nested_region):
            return True
        if re.search(r'\{\s*"events"\s*:', nested_region):
            return True
    return False


def current_open_event_object_start(text):
    """Return the start offset of the currently open top-level event object."""

    text = str(text or "")
    root_start = re.search(r'(?<!\\)"events"\s*:\s*\[', text)
    if not root_start:
        return -1
    array_start = root_start.end() - 1
    stack = ["["]
    current_event_start = -1
    in_string = False
    escaped = False
    for index in range(array_start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{" and stack == ["["]:
            current_event_start = index
            stack.append(char)
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not stack or stack[-1] != expected:
                continue
            stack.pop()
            if char == "}" and stack == ["["]:
                current_event_start = -1
            if char == "]" and not stack:
                return -1
    return current_event_start


def top_level_event_key_fragment(text):
    """Return the unfinished top-level event key prefix, or None outside keys."""

    event_start = current_open_event_object_start(text)
    if event_start < 0:
        return None

    depth = 1
    in_string = False
    escaped = False
    string_start = -1
    expect_key = True
    for index in range(event_start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and expect_key:
                    expect_key = False
            continue
        if char == '"':
            in_string = True
            string_start = index
            continue
        if char in "[{":
            depth += 1
            continue
        if char in "]}":
            depth -= 1
            if depth <= 0:
                return None
            continue
        if depth != 1:
            continue
        if expect_key:
            if char.isspace() or char == ",":
                continue
            return None
        if char == ",":
            expect_key = True

    if in_string and depth == 1 and expect_key and string_start >= 0:
        return text[string_start:]
    if depth == 1 and expect_key:
        return ""
    return None


def nested_events_key_candidate_token_ids(tokenizer, vocab_size):
    candidate_texts = set()
    suffixes = ("events", "vents", "ents", "nts", "ts", "s")
    for suffix in suffixes:
        candidate_texts.update(
            {
                suffix,
                f'{suffix}"',
                f'{suffix}":',
                f'{suffix}":[',
                f'"{suffix}',
                f' "{suffix}',
                f'\n"{suffix}',
            }
        )
    candidate_texts.update({'"events"', '"events":', '"events":[', ' "events"', '\n"events"'})

    token_ids = set()
    for text in candidate_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = strip_bos(ids, getattr(tokenizer, "bos_token_id", None))
        if ids:
            token_id = int(ids[0])
            if 0 <= token_id < vocab_size:
                token_ids.add(token_id)
    return token_ids


def token_would_form_nested_events_key(key_fragment, token_text):
    fragment = str(key_fragment or "")
    token_text = str(token_text or "")
    if not fragment:
        candidate = token_text.lstrip()
    else:
        candidate = fragment + token_text
    if not candidate:
        return False

    legal_keys = ('"event_type"', '"trigger"', '"evidence"', '"arguments"')
    legal_bare_keys = tuple(key.strip('"') for key in legal_keys)
    all_legal_keys = legal_keys + legal_bare_keys
    if any(key.startswith(candidate) or candidate.startswith(key) for key in all_legal_keys):
        return False

    nested_keys = ('"events"', "events")
    return any(key.startswith(candidate) or candidate.startswith(key) for key in nested_keys)


def constrain_nested_events_logits(logits, tokenizer, pred_tokens):
    decoded_so_far = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    key_fragment = top_level_event_key_fragment(decoded_so_far)
    if key_fragment is None:
        return logits

    candidate_ids = nested_events_key_candidate_token_ids(tokenizer, logits.size(-1))
    if not candidate_ids:
        return logits

    constrained = logits.clone()
    for token_id in candidate_ids:
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if token_would_form_nested_events_key(key_fragment, token_text):
            constrained[:, token_id] = -float("inf")
    return constrained


def bias_json_start_logits(logits, tokenizer, pred_tokens, bias):
    if pred_tokens or bias is None or float(bias) <= 0:
        return logits
    biased = logits.clone()
    candidate_ids = set()
    for prefix in ("{", '{"', '{"events"', '{"events":', JSON_EVENT_ARRAY_PREFIX):
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        ids = strip_bos(ids, tokenizer.bos_token_id)
        if ids:
            candidate_ids.add(int(ids[0]))
    for token_id in candidate_ids:
        if 0 <= token_id < biased.size(-1):
            biased[:, token_id] += float(bias)
    return biased


def should_constrain_event_object_start(decoded_text):
    """Return True when the next token must start an event object.

    This is a structural decoder guard for the compact JSON protocol.  It only
    fires inside the top-level events array after the opening ``[`` or after an
    inter-event comma.  It does not decide field names or values.
    """

    text = str(decoded_text or "")
    state = inspect_event_decode_state(text)
    if not state.events_array_open or state.event_object_open:
        return False
    stripped = text.rstrip()
    if stripped.endswith("[") and state.completed_event_count == 0:
        return True
    if stripped.endswith(",") and state.completed_event_count >= 1:
        return True
    return False


def event_object_start_token_ids(tokenizer, vocab_size):
    return event_object_start_candidate_token_ids(tokenizer, vocab_size)


def constrain_event_object_start_logits(logits, tokenizer, pred_tokens):
    decoded_so_far = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not should_constrain_event_object_start(decoded_so_far):
        return logits

    allowed_ids = event_object_start_token_ids(tokenizer, logits.size(-1))
    if not allowed_ids:
        return logits

    constrained = torch.full_like(logits, -float("inf"))
    for token_id in allowed_ids:
        constrained[:, token_id] = logits[:, token_id]
    return constrained


def apply_generation_logit_controls(
    logits,
    tokenizer,
    pred_tokens,
    training_args,
    handle_exist_probabilities=None,
    return_details=False,
):
    proposal_logits = logits
    logits = bias_json_start_logits(
        logits,
        tokenizer,
        pred_tokens,
        getattr(training_args, "json_start_bias", 0.0),
    )
    if (
        bool(getattr(training_args, "semantic_json_stop", True))
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        decoded_so_far = tokenizer.decode(
            pred_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not has_complete_json(decoded_so_far):
            logits = logits.clone()
            eos_id = int(tokenizer.eos_token_id)
            if 0 <= eos_id < logits.size(-1):
                logits[:, eos_id] = -float("inf")
    logits = constrain_event_object_start_logits(logits, tokenizer, pred_tokens)
    logits = constrain_nested_events_logits(logits, tokenizer, pred_tokens)
    logits, boundary_details = apply_handle_event_boundary_control(
        logits,
        tokenizer,
        pred_tokens,
        training_args,
        handle_exist_probabilities=handle_exist_probabilities,
        proposal_logits=proposal_logits,
    )
    logits = constrain_event_root_close_logits(
        logits,
        tokenizer,
        pred_tokens,
        training_args,
    )
    if return_details:
        return logits, boundary_details
    return logits


def tokenize_generation_prompt(tokenizer, prompt, training_args, model):
    context_limit = resolve_model_context_limit(model, training_args)
    bot_and_delimiter_tokens = 2 if training_args.remove_eos else 4
    forced_prefix_tokens = 0
    if bool(getattr(training_args, "force_json_prefix", False)):
        forced_prefix_tokens = len(encode_json_event_array_prefix(tokenizer))
    structural_reserve = event_generation_prefix_reserve(
        num_latent=training_args.num_latent,
        max_event_count=training_args.max_event_count,
        fixed_suffix_tokens=bot_and_delimiter_tokens + forced_prefix_tokens,
    )
    prompt_budget = max(1, context_limit - structural_reserve)
    try:
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        input_ids = encoded["input_ids"]
    except TypeError:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    original_len = len(input_ids)
    truncated = original_len > prompt_budget
    if truncated:
        head_len = max(1, int(prompt_budget * 0.35))
        tail_len = max(1, prompt_budget - head_len)
        input_ids = input_ids[:head_len] + input_ids[-tail_len:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_tensor)
    return {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "prompt_tokens": original_len,
        "prompt_budget": prompt_budget,
        "prompt_truncated": truncated,
        "context_limit": context_limit,
    }


def resolve_model_context_limit(model, training_args):
    """Use the backbone's actual positional limit, with a finite CLI fallback."""
    config = getattr(getattr(model, "codi", model), "config", None)
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None) if config is not None else None
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < value < 10**8:
            return value
    fallback = int(getattr(training_args, "model_max_length", 4096))
    return max(1, fallback)


def generate_event_prediction(
    model,
    tokenizer,
    prompt,
    training_args,
    runtime_device,
    example=None,
):
    generation_start_time = time.perf_counter()
    set_seed(42)
    use_greedy = bool(training_args.greedy)
    batch = tokenize_generation_prompt(tokenizer, prompt, training_args, model)
    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(runtime_device)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1).to(runtime_device)

    with torch.no_grad():
        past_key_values = None
        outputs = model.codi(
            inputs_embeds=model.embed_input_ids(batch["input_ids"]),
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=batch["attention_mask"],
        )
        past_key_values = model.normalize_past_key_values(outputs.past_key_values)
        cache_attention_mask = (
            batch["attention_mask"].bool()
            if bool(getattr(training_args, "fix_attn_mask", False))
            else None
        )
        encoder_states = outputs.hidden_states[-1]
        encoder_summary = encoder_states[:, -1, :]
        input_ids_list = batch["input_ids"][0].detach().cpu().tolist()
        document_token_mask = torch.tensor(
            build_document_token_mask_from_source(
                input_ids_list,
                tokenizer,
                prompt,
                max_sentence_markers=getattr(training_args, "max_sentence_markers", 32),
            ),
            dtype=torch.bool,
            device=runtime_device,
        ).unsqueeze(0)
        encoder_summary = model.fuse_document_summary(
            encoder_states,
            encoder_summary,
            attention_mask=batch["attention_mask"],
            document_token_mask=document_token_mask,
        )
        predicted_budget = None
        predicted_budget_id = None
        budget_expected = None
        budget_top1 = None
        budget_entropy = None
        budget_top1_margin = None
        budget_statistics = None
        budget_logits = None
        event_handle_anchors = []
        event_handle_count = None
        event_handle_exist_confidence = None
        event_handle_exist_probabilities = []
        event_handle_memory = None
        event_handle_active_mask = None
        if getattr(model, "adaptive_latent", False) and hasattr(model, "latent_budget_head"):
            budget_hidden = encoder_summary.to(dtype=model.latent_budget_head.weight.dtype)
            budget_statistics = latent_budget_statistics(
                model.latent_budget_head(budget_hidden).float()
            )
            budget_logits = budget_statistics["logits"]
            predicted_budget_id = int(
                budget_statistics["proposal"].detach().cpu()[0]
            )
            predicted_budget = float(predicted_budget_id)
            budget_expected = float(
                budget_statistics["expected"].detach().cpu()[0]
            )
            budget_top1 = int(budget_statistics["top1"].detach().cpu()[0])
            budget_entropy = float(
                budget_statistics["entropy"].detach().cpu()[0]
            )
            budget_top1_margin = float(
                budget_statistics["top1_margin"].detach().cpu()[0]
            )

        latent_embd = encoder_summary.unsqueeze(1)
        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        if (
            predicted_budget_id is not None
            and hasattr(model, "latent_budget_embedding")
            and getattr(model, "latent_budget_condition_factor", 0.0) > 0
        ):
            budget_id = torch.tensor([predicted_budget_id], dtype=torch.long, device=runtime_device)
            budget_embd = model.latent_budget_embedding(budget_id)
            budget_embd = budget_embd.to(device=runtime_device, dtype=latent_embd.dtype).unsqueeze(1)
            latent_embd = latent_embd + float(model.latent_budget_condition_factor) * budget_embd

        handle_budget_condition = None
        if (
            budget_logits is not None
            and hasattr(model, "latent_budget_embedding")
            and bool(getattr(model, "budget_conditioned_handle_count", False))
            and float(getattr(model, "budget_handle_count_factor", 0.0)) > 0.0
        ):
            budget_probs = budget_statistics["probabilities"].detach().float()
            budget_ids = torch.arange(
                1,
                model.num_latent + 1,
                dtype=torch.long,
                device=runtime_device,
            )
            budget_table = model.latent_budget_embedding(budget_ids).detach().to(
                device=runtime_device,
                dtype=budget_probs.dtype,
            )
            handle_budget_condition = torch.matmul(budget_probs, budget_table)

        if (
            example is not None
            and hasattr(model, "event_handle_embedding")
            and getattr(model, "event_handle_condition_factor", 0.0) > 0
        ):
            max_markers = int(getattr(training_args, "max_sentence_markers", 32))
            sentence_rep_positions = build_sentence_rep_positions_from_source(
                input_ids_list,
                tokenizer,
                prompt,
                max_sentence_markers=max_markers,
            )
            valid_sentences = torch.tensor(
                [[position >= 0 for position in sentence_rep_positions]],
                dtype=torch.bool,
                device=runtime_device,
            )
            if valid_sentences.any():
                pos_tensor = torch.tensor(
                    [[max(0, position) for position in sentence_rep_positions]],
                    dtype=torch.long,
                    device=runtime_device,
                )
                sentence_hidden = encoder_states.gather(
                    1,
                    pos_tensor.unsqueeze(-1).expand(-1, -1, encoder_states.size(-1)),
                )
                if sentence_hidden.size(1) > 0:
                    handle_result = model.build_event_handle_plan(
                        sentence_hidden.float(),
                        valid_sentences,
                        global_context=latent_embd[:, -1, :],
                        budget_condition=handle_budget_condition,
                    )
                    handle_exist_probs = (
                        handle_result.get("handle_exist_probs")
                        if handle_result is not None
                        else None
                    )
                    if handle_exist_probs is not None and handle_exist_probs.numel() > 0:
                        event_handle_exist_probabilities = (
                            handle_exist_probs[0]
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                        )
                    handle_plan = handle_result["plan"] if handle_result is not None else None
                    handle_indices = handle_result["predicted_indices"] if handle_result is not None else None
                    event_handle_active_mask = handle_result.get("plan_active") if handle_result is not None else None
                    if handle_plan is not None:
                        dynamic_counts = handle_result.get("dynamic_counts")
                        if dynamic_counts is not None:
                            event_handle_count = int(dynamic_counts.detach().cpu()[0])
                        packed_exist_probs = handle_result.get("packed_exist_probs")
                        if packed_exist_probs is not None:
                            active_probabilities = packed_exist_probs[0]
                            if active_probabilities.numel() > 0:
                                event_handle_exist_confidence = float(
                                    active_probabilities.mean().detach().cpu()
                                )
                        handle_plan = handle_plan.to(device=runtime_device, dtype=latent_embd.dtype)
                        event_handle_memory = handle_plan
                        if event_handle_active_mask is not None:
                            event_handle_active_mask = event_handle_active_mask.to(device=runtime_device).bool()
                        handle_cache_mask = (
                            event_handle_active_mask
                            if event_handle_active_mask is not None
                            else torch.ones(
                                (handle_plan.size(0), handle_plan.size(1)),
                                dtype=torch.bool,
                                device=runtime_device,
                            )
                        )
                        handle_out = model.codi(
                            inputs_embeds=handle_plan,
                            output_hidden_states=False,
                            attention_mask=(
                                torch.cat(
                                    (
                                        cache_attention_mask,
                                        handle_cache_mask,
                                    ),
                                    dim=1,
                                )
                                if cache_attention_mask is not None
                                else None
                            ),
                            use_cache=True,
                            output_attentions=False,
                            past_key_values=past_key_values,
                        )
                        past_key_values = model.normalize_past_key_values(handle_out.past_key_values)
                        if cache_attention_mask is not None:
                            cache_attention_mask = torch.cat(
                                (
                                    cache_attention_mask,
                                    handle_cache_mask,
                                ),
                                dim=1,
                            )
                        event_handle_anchors = [
                            (
                                f"S{int(index)}"
                                if (
                                    0 <= int(index) < len(sentence_rep_positions)
                                    and sentence_rep_positions[int(index)] >= 0
                                )
                                else None
                            )
                            for index in handle_indices[0].detach().cpu().tolist()
                        ]
        max_latent_steps = training_args.inf_latent_iterations or training_args.num_latent
        if predicted_budget_id is not None:
            max_latent_steps = min(int(max_latent_steps), int(predicted_budget_id))
        used_latent_steps = 0
        for i in range(max_latent_steps):
            if event_handle_memory is not None:
                latent_embd = model.apply_handle_latent_controller(
                    latent_embd[:, -1, :],
                    event_handle_memory,
                    event_handle_active_mask,
                ).unsqueeze(1)
            latent_attention_mask = None
            if cache_attention_mask is not None:
                latent_attention_mask = torch.cat(
                    (
                        cache_attention_mask,
                        torch.ones(
                            (latent_embd.size(0), latent_embd.size(1)),
                            dtype=torch.bool,
                            device=runtime_device,
                        ),
                    ),
                    dim=1,
                )
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=latent_attention_mask,
            )
            past_key_values = model.normalize_past_key_values(outputs.past_key_values)
            if latent_attention_mask is not None:
                cache_attention_mask = latent_attention_mask
            latent_hidden = outputs.hidden_states[-1][:, -1, :]
            used_latent_steps = i + 1

            latent_embd = latent_hidden.unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        bridge_hidden = latent_hidden if used_latent_steps > 0 else encoder_summary
        bridge_embeds, bridge_mask = model.build_decoder_bridge_embeds(
            bridge_hidden,
            event_handle_memory,
            event_handle_active_mask,
            return_mask=True,
        )
        if bridge_embeds is not None:
            bridge_mask = (
                bridge_mask.to(device=runtime_device).bool()
                if bridge_mask is not None
                else torch.ones(
                    (bridge_embeds.size(0), bridge_embeds.size(1)),
                    dtype=torch.bool,
                    device=runtime_device,
                )
            )
            bridge_out = model.codi(
                inputs_embeds=bridge_embeds,
                output_hidden_states=False,
                attention_mask=(
                    torch.cat(
                        (
                            cache_attention_mask,
                            bridge_mask,
                        ),
                        dim=1,
                    )
                    if cache_attention_mask is not None
                    else None
                ),
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = model.normalize_past_key_values(bridge_out.past_key_values)
            if cache_attention_mask is not None:
                cache_attention_mask = torch.cat(
                    (
                        cache_attention_mask,
                        bridge_mask,
                    ),
                    dim=1,
                )

        if training_args.remove_eos:
            delimiter_ids = torch.tensor([model.eot_id], dtype=torch.long, device=runtime_device)
        else:
            delimiter_ids = torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=runtime_device)
        delimiter_embeds = model.embed_input_ids(delimiter_ids).unsqueeze(0)
        delimiter_out = model.codi(
            inputs_embeds=delimiter_embeds,
            output_hidden_states=False,
            attention_mask=(
                torch.cat(
                    (
                        cache_attention_mask,
                        torch.ones(
                            (delimiter_embeds.size(0), delimiter_embeds.size(1)),
                            dtype=torch.bool,
                            device=runtime_device,
                        ),
                    ),
                    dim=1,
                )
                if cache_attention_mask is not None
                else None
            ),
            use_cache=True,
            output_attentions=False,
            past_key_values=past_key_values,
        )
        past_key_values = model.normalize_past_key_values(delimiter_out.past_key_values)
        if cache_attention_mask is not None:
            cache_attention_mask = torch.cat(
                (
                    cache_attention_mask,
                    torch.ones(
                        (delimiter_embeds.size(0), delimiter_embeds.size(1)),
                        dtype=torch.bool,
                        device=runtime_device,
                    ),
                ),
                dim=1,
            )
        pending_logits = last_token_logits(delimiter_out.logits, len(tokenizer))
        pred_tokens = []

        # The decoder is trained to emit the JSON body directly.  Full-prefix
        # forcing is optional because it can make weak checkpoints restart the
        # root object and produce nested {"events": ...} payloads.
        json_prefix_ids = []
        if bool(getattr(training_args, "force_json_prefix", False)):
            json_prefix_ids = encode_json_event_array_prefix(tokenizer)
        if json_prefix_ids:
            json_prefix = torch.tensor(json_prefix_ids, dtype=torch.long, device=runtime_device).unsqueeze(0)
            json_prefix_embeds = model.embed_input_ids(json_prefix)
            json_prefix_out = model.codi(
                inputs_embeds=json_prefix_embeds,
                output_hidden_states=False,
                attention_mask=(
                    torch.cat(
                        (
                            cache_attention_mask,
                            torch.ones(
                                (json_prefix_embeds.size(0), json_prefix_embeds.size(1)),
                                dtype=torch.bool,
                                device=runtime_device,
                            ),
                        ),
                        dim=1,
                    )
                    if cache_attention_mask is not None
                    else None
                ),
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = model.normalize_past_key_values(json_prefix_out.past_key_values)
            if cache_attention_mask is not None:
                cache_attention_mask = torch.cat(
                    (
                        cache_attention_mask,
                        torch.ones(
                            (json_prefix_embeds.size(0), json_prefix_embeds.size(1)),
                            dtype=torch.bool,
                            device=runtime_device,
                        ),
                    ),
                    dim=1,
                )
            pending_logits = last_token_logits(json_prefix_out.logits, len(tokenizer))
            pred_tokens = list(json_prefix_ids)

        context_limit = int(batch["context_limit"])
        cached_length = model.past_key_values_length(past_key_values)
        if cached_length is None and cache_attention_mask is not None:
            cached_length = int(cache_attention_mask.size(1))
        if cached_length is None:
            raise RuntimeError("Cannot determine decoder KV-cache length for safe generation.")
        available_generation_tokens = context_limit - int(cached_length)
        if available_generation_tokens < 1:
            raise RuntimeError(
                "No context capacity remains for JSON generation: "
                f"context_limit={context_limit}, cached_length={cached_length}."
            )
        configured_generation_tokens = max(
            1,
            int(getattr(training_args, "generation_max_tokens", 1024)),
        )
        max_new_tokens = min(configured_generation_tokens, available_generation_tokens)
        output = None
        stop_reason = "max_tokens"
        boundary_control_applications = 0
        boundary_opportunities = 0
        direct_boundary_gate_applications = 0
        completed_boundary_gate_applications = 0
        boundary_continue_decisions = 0
        boundary_close_decisions = 0
        boundary_noncommit_decisions = 0
        decoder_handle_advances = 0
        direct_continue_candidates = 0
        direct_stop_candidates = 0
        hard_event_cap_hit = False
        boundary_qwen_continue_probabilities = []
        boundary_handle_continue_probabilities = []
        boundary_blended_continue_probabilities = []
        for _ in range(max_new_tokens):
            if pending_logits is None:
                if output is None:
                    raise RuntimeError("Decoder token embedding is missing after the initial logits step.")
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=(
                        torch.cat(
                            (
                                cache_attention_mask,
                                torch.ones(
                                    (output.size(0), output.size(1)),
                                    dtype=torch.bool,
                                    device=runtime_device,
                                ),
                            ),
                            dim=1,
                        )
                        if cache_attention_mask is not None
                        else None
                    ),
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = model.normalize_past_key_values(out.past_key_values)
                if cache_attention_mask is not None:
                    cache_attention_mask = torch.cat(
                        (
                            cache_attention_mask,
                            torch.ones(
                                (output.size(0), output.size(1)),
                                dtype=torch.bool,
                                device=runtime_device,
                            ),
                        ),
                        dim=1,
                    )
                logits = last_token_logits(out.logits, len(tokenizer))
            else:
                logits = pending_logits
                pending_logits = None
            logits, boundary_details = apply_generation_logit_controls(
                logits,
                tokenizer,
                pred_tokens,
                training_args,
                handle_exist_probabilities=event_handle_exist_probabilities,
                return_details=True,
            )
            if boundary_details.get("opportunity"):
                boundary_opportunities += 1
            if boundary_details["active"]:
                boundary_control_applications += 1
                if boundary_details.get("stage") == "direct_boundary":
                    direct_boundary_gate_applications += 1
                elif boundary_details.get("stage") == "completed_boundary":
                    completed_boundary_gate_applications += 1
                direct_continue_candidates += int(
                    boundary_details.get("direct_continue_candidates", 0)
                    or 0
                )
                direct_stop_candidates += int(
                    boundary_details.get("direct_stop_candidates", 0)
                    or 0
                )
                hard_event_cap_hit = (
                    hard_event_cap_hit or bool(boundary_details["hard_cap"])
                )
                if boundary_details["qwen_continue_probability"] is not None:
                    boundary_qwen_continue_probabilities.append(
                        float(boundary_details["qwen_continue_probability"])
                    )
                if boundary_details["handle_continue_probability"] is not None:
                    boundary_handle_continue_probabilities.append(
                        float(boundary_details["handle_continue_probability"])
                    )
                if boundary_details["blended_continue_probability"] is not None:
                    boundary_blended_continue_probabilities.append(
                        float(boundary_details["blended_continue_probability"])
                    )
            next_token_id = sample_next_token(
                logits,
                use_greedy,
                temperature=getattr(training_args, "generation_temperature", 0.7),
                top_k=getattr(training_args, "generation_top_k", 40),
                top_p=getattr(training_args, "generation_top_p", 0.95),
            ).reshape(1)
            decoded_before_token = tokenizer.decode(
                pred_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            next_token_piece = _decode_single_token(
                tokenizer,
                next_token_id.item(),
            )
            if boundary_details.get("stage") == "direct_boundary":
                boundary_kind = classify_event_token_transition(
                    decoded_before_token,
                    next_token_piece,
                )
            elif boundary_details.get("stage") == "completed_boundary":
                boundary_kind = classify_event_boundary_token(
                    tokenizer,
                    next_token_id.item(),
                )
            else:
                boundary_kind = None
            if boundary_kind == "continue":
                boundary_continue_decisions += 1
                decoder_handle_advances += 1
            elif boundary_kind == "stop":
                boundary_close_decisions += 1
            elif boundary_details["active"]:
                boundary_noncommit_decisions += 1
            pred_tokens.append(next_token_id.item())
            decoded_so_far = tokenizer.decode(
                pred_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if next_token_id.item() == tokenizer.eos_token_id:
                stop_reason = (
                    "eos_complete"
                    if should_stop_on_complete_json(decoded_so_far, training_args)
                    else "eos_incomplete"
                )
                break
            if has_complete_json(decoded_so_far):
                if should_stop_on_complete_json(decoded_so_far, training_args):
                    stop_reason = "complete_json"
                else:
                    stop_reason = "semantic_incomplete_json"
                break
            output = model.embed_input_ids(next_token_id).unsqueeze(1)

    decoded_body = tokenizer.decode(pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded = f"{FINAL_JSON_MARKER}\n{decoded_body}"
    boundary_mode = normalize_event_boundary_control(
        getattr(training_args, "event_boundary_control", "off")
    )
    handle_count_distribution = None
    handle_expected_count = None
    if event_handle_exist_probabilities:
        handle_expected_count = float(sum(event_handle_exist_probabilities))
        handle_count_distribution = event_count_distribution(
            event_handle_exist_probabilities,
        )
    return {
        "text": decoded,
        "latent_steps": used_latent_steps,
        "predicted_budget": predicted_budget,
        "planned_budget": predicted_budget_id,
        "budget_expected": budget_expected,
        "budget_top1": budget_top1,
        "budget_entropy": budget_entropy,
        "budget_top1_margin": budget_top1_margin,
        "event_handle_anchors": event_handle_anchors,
        "event_handle_count": event_handle_count,
        "event_handle_exist_confidence": event_handle_exist_confidence,
        "event_handle_expected_count": handle_expected_count,
        "event_handle_count_reliability": (
            event_count_reliability(handle_count_distribution)
            if handle_count_distribution is not None
            else None
        ),
        "event_boundary_control": boundary_mode,
        "event_boundary_applications": boundary_control_applications,
        "event_boundary_opportunities": boundary_opportunities,
        "direct_boundary_gate_applications": direct_boundary_gate_applications,
        "completed_boundary_gate_applications": completed_boundary_gate_applications,
        "event_boundary_continue_decisions": boundary_continue_decisions,
        "event_boundary_close_decisions": boundary_close_decisions,
        "event_boundary_noncommit_decisions": boundary_noncommit_decisions,
        "decoder_handle_advances": decoder_handle_advances,
        "direct_continue_candidates": direct_continue_candidates,
        "direct_stop_candidates": direct_stop_candidates,
        "event_boundary_qwen_continue_mean": (
            mean_or_zero(boundary_qwen_continue_probabilities)
            if boundary_qwen_continue_probabilities
            else None
        ),
        "event_boundary_handle_continue_mean": (
            mean_or_zero(boundary_handle_continue_probabilities)
            if boundary_handle_continue_probabilities
            else None
        ),
        "event_boundary_blended_continue_mean": (
            mean_or_zero(boundary_blended_continue_probabilities)
            if boundary_blended_continue_probabilities
            else None
        ),
        "hard_event_cap_hit": hard_event_cap_hit,
        "decode_mode": "greedy" if use_greedy else "sample",
        "generated_tokens": len(pred_tokens),
        "generation_token_limit": len(json_prefix_ids) + max_new_tokens,
        "stop_reason": stop_reason,
        "generation_seconds": time.perf_counter() - generation_start_time,
        "prompt_tokens": batch.get("prompt_tokens"),
        "prompt_budget": batch.get("prompt_budget"),
        "prompt_truncated": batch.get("prompt_truncated", False),
    }


def format_float(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def format_duration(seconds):
    try:
        seconds = max(0.0, float(seconds))
    except Exception:
        return "NA"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def beijing_time_text():
    beijing_tz = timezone(timedelta(hours=8), "Asia/Shanghai")
    return datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def beijing_time_for_filename():
    beijing_tz = timezone(timedelta(hours=8), "Asia/Shanghai")
    return datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S_BJT")


def safe_filename_part(value, default="run"):
    text = str(value or default)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._")
    return text[:96] or default


def format_list(values):
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values)


def format_generation_report(prediction):
    parts = [
        f"latent_steps={prediction.get('latent_steps', 0)}",
        f"predicted_budget={format_float(prediction.get('predicted_budget'), 2)}",
        f"planned_budget={prediction.get('planned_budget')}",
        f"budget_expected={format_float(prediction.get('budget_expected'), 2)}",
        f"budget_top1={prediction.get('budget_top1')}",
        f"budget_entropy={format_float(prediction.get('budget_entropy'), 3)}",
        f"budget_margin={format_float(prediction.get('budget_top1_margin'), 3)}",
        f"event_handles={prediction.get('event_handle_count')}",
        f"handle_conf={format_float(prediction.get('event_handle_exist_confidence'), 2)}",
        f"handle_expected={format_float(prediction.get('event_handle_expected_count'), 2)}",
        f"handle_reliability={format_float(prediction.get('event_handle_count_reliability'), 2)}",
        f"handle_anchors={format_list(prediction.get('event_handle_anchors', []))}",
        f"boundary={prediction.get('event_boundary_control', 'off')}",
        f"boundary_apply={int(prediction.get('event_boundary_applications', 0) or 0)}",
        f"boundary_opportunities={int(prediction.get('event_boundary_opportunities', 0) or 0)}",
        f"direct_gate={int(prediction.get('direct_boundary_gate_applications', 0) or 0)}",
        f"completed_gate={int(prediction.get('completed_boundary_gate_applications', 0) or 0)}",
        f"boundary_continue={int(prediction.get('event_boundary_continue_decisions', 0) or 0)}",
        f"boundary_close={int(prediction.get('event_boundary_close_decisions', 0) or 0)}",
        f"boundary_noncommit={int(prediction.get('event_boundary_noncommit_decisions', 0) or 0)}",
        f"handle_advances={int(prediction.get('decoder_handle_advances', 0) or 0)}",
        f"direct_continue_candidates={int(prediction.get('direct_continue_candidates', 0) or 0)}",
        f"direct_stop_candidates={int(prediction.get('direct_stop_candidates', 0) or 0)}",
        f"qwen_continue={format_float(prediction.get('event_boundary_qwen_continue_mean'), 2)}",
        f"handle_continue={format_float(prediction.get('event_boundary_handle_continue_mean'), 2)}",
        f"blended_continue={format_float(prediction.get('event_boundary_blended_continue_mean'), 2)}",
        f"hard_cap={bool(prediction.get('hard_event_cap_hit', False))}",
        f"generated_events={int(prediction.get('generated_event_count', 0) or 0)}",
        f"verified_events={int(prediction.get('verified_event_count', 0) or 0)}",
        f"duplicate_events={int(prediction.get('duplicate_event_count', 0) or 0)}",
        f"decode={prediction.get('decode_mode', 'unknown')}",
        (
            f"tokens={int(prediction.get('generated_tokens', 0) or 0)}"
            f"/{int(prediction.get('generation_token_limit', 0) or 0)}"
        ),
        f"stop={prediction.get('stop_reason', 'unknown')}",
        f"seconds={format_float(prediction.get('generation_seconds', 0.0), 1)}",
    ]
    if prediction.get("prompt_truncated"):
        parts.extend(
            [
                f"prompt_tokens={prediction.get('prompt_tokens')}/{prediction.get('prompt_budget')}",
                "truncated=True",
            ]
        )
    return "--- adaptive latent generation\n" + " ".join(parts)


def format_complexity_report(example, derived_target_steps):
    features = example.get("complexity_features", {})
    feature_text = " ".join(f"{key}={value}" for key, value in sorted(features.items()))
    return (
        f"--- adaptive budget\n"
        f"target_latent_steps={derived_target_steps} "
        f"sentences={len(example.get('sentences', []))} "
        f"events={len(example.get('events', []))} "
        f"{feature_text}"
    )


def format_prediction_report(selected, verified_f1, raw_f1=None):
    detail = selected.get("detail", {})
    detail_text = " ".join(f"{key}={format_float(value, 3)}" for key, value in detail.items())
    raw_f1 = raw_f1 or {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return (
        f"--- prediction\n"
        f"source=single_decode parse_score={format_float(selected.get('score', 0.0), 3)}\n"
        f"raw_tuple_precision={format_float(raw_f1.get('precision', 0.0), 3)} "
        f"raw_tuple_recall={format_float(raw_f1.get('recall', 0.0), 3)} "
        f"raw_tuple_f1={format_float(raw_f1.get('f1', 0.0), 3)}\n"
        f"parsed_tuple_precision={format_float(verified_f1.get('precision', 0.0), 3)} "
        f"parsed_tuple_recall={format_float(verified_f1.get('recall', 0.0), 3)} "
        f"parsed_tuple_f1={format_float(verified_f1.get('f1', 0.0), 3)}\n"
        f"{detail_text}"
    )


STANDARD_MAIN_METRICS = (
    ("trigger", "Trigger"),
    ("arg_i", "Arg-I"),
    ("arg_c", "Arg-C"),
)
STANDARD_APPENDIX_METRICS = (
    ("span", "Span"),
    ("head", "Head"),
)
RELAXED_DIAGNOSTIC_METRICS = (
    ("soft_trigger", "Soft Trigger"),
    ("relaxed_span", "Relaxed Span"),
    ("normalized_arg_i", "Normalized Arg-I"),
    ("normalized_arg_c", "Normalized Arg-C"),
    ("partial_argument", "Partial Arg-C"),
)


def format_metric_triplet(metric, prefix=""):
    return (
        f"{prefix}p={format_float(metric.get('precision', 0.0), 3)} "
        f"{prefix}r={format_float(metric.get('recall', 0.0), 3)} "
        f"{prefix}f1={format_float(metric.get('f1', 0.0), 3)} "
        f"{prefix}hit={int(metric.get('hit', 0))} "
        f"{prefix}pred={int(metric.get('pred', 0))} "
        f"{prefix}gold={int(metric.get('gold', 0))}"
    )


def format_paper_metrics_report(raw_metrics, verified_metrics):
    lines = ["--- paper main table metrics"]
    for key, label in STANDARD_MAIN_METRICS:
        lines.append(
            f"{label} | {format_metric_triplet(raw_metrics.get(key, {}), 'raw_')} "
            f"{format_metric_triplet(verified_metrics.get(key, {}), 'verified_')}"
        )
    lines.append("--- paper appendix table metrics")
    for key, label in STANDARD_APPENDIX_METRICS:
        lines.append(
            f"{label} | {format_metric_triplet(raw_metrics.get(key, {}), 'raw_')} "
            f"{format_metric_triplet(verified_metrics.get(key, {}), 'verified_')}"
        )
    return "\n".join(lines)


def format_relaxed_metrics_report(raw_metrics, verified_metrics):
    lines = ["--- relaxed diagnostic metrics"]
    for key, label in RELAXED_DIAGNOSTIC_METRICS:
        lines.append(
            f"{label} | {format_metric_triplet(raw_metrics.get(key, {}), 'raw_')} "
            f"{format_metric_triplet(verified_metrics.get(key, {}), 'verified_')}"
        )
    return "\n".join(lines)


def metric_from_counts(hit, pred, gold):
    if gold == 0 and pred == 0:
        precision = recall = f1 = 1.0
    elif pred == 0 or gold == 0:
        precision = recall = f1 = 0.0
    else:
        precision = hit / pred
        recall = hit / gold
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hit": float(hit),
        "pred": float(pred),
        "gold": float(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_set_metric_counts(totals, gold_items, pred_items):
    gold_items = set(gold_items or [])
    pred_items = set(pred_items or [])
    totals["hit"] += len(gold_items & pred_items)
    totals["pred"] += len(pred_items)
    totals["gold"] += len(gold_items)


def finalize_set_metric_counts(totals):
    return metric_from_counts(
        int(totals.get("hit", 0)),
        int(totals.get("pred", 0)),
        int(totals.get("gold", 0)),
    )


def set_metric_record(gold_items, pred_items):
    totals = {"hit": 0, "pred": 0, "gold": 0}
    add_set_metric_counts(totals, gold_items, pred_items)
    return machine_metric_record(finalize_set_metric_counts(totals))


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def format_summary_line(label, metric, prefix=""):
    return f"{label} | {format_metric_triplet(metric, prefix)}"


def event_count_bucket_names(gold_event_count):
    names = []
    if gold_event_count <= 2:
        names.append("gold<=2")
    if gold_event_count >= 3:
        names.append("gold>=3")
    if gold_event_count >= 4:
        names.append("gold>=4")
    return names


def empty_event_count_diagnostics():
    return {
        "n": 0,
        "gold_sum": 0.0,
        "pred_sum": 0.0,
        "absolute_error_sum": 0.0,
        "exact": 0,
        "over": 0,
        "under": 0,
    }


def add_event_count_diagnostic(totals, gold_event_count, pred_event_count):
    gold_count = max(0, int(gold_event_count))
    pred_count = max(0, int(pred_event_count))
    totals["n"] += 1
    totals["gold_sum"] += gold_count
    totals["pred_sum"] += pred_count
    totals["absolute_error_sum"] += abs(pred_count - gold_count)
    totals["exact"] += int(pred_count == gold_count)
    totals["over"] += int(pred_count > gold_count)
    totals["under"] += int(pred_count < gold_count)


def format_event_count_diagnostic_summary(
    bucket_totals,
    title="--- event count diagnostics",
):
    lines = [str(title)]
    for bucket_name in ("all", "gold<=2", "gold>=3", "gold>=4"):
        totals = bucket_totals[bucket_name]
        count = int(totals["n"])
        denominator = max(1, count)
        lines.append(
            f"{bucket_name} | n={count} "
            f"gold_mean={totals['gold_sum'] / denominator:.3f} "
            f"pred_mean={totals['pred_sum'] / denominator:.3f} "
            f"mae={totals['absolute_error_sum'] / denominator:.3f} "
            f"exact={totals['exact'] / denominator:.3f} "
            f"over={totals['over'] / denominator:.3f} "
            f"under={totals['under'] / denominator:.3f}"
        )
    return "\n".join(lines)


def format_relaxed_bucket_summary(bucket_counts):
    lines = ["--- verified relaxed metrics by gold event count"]
    for bucket_name in ("gold<=2", "gold>=3", "gold>=4"):
        bucket = bucket_counts[bucket_name]
        summary = finalize_relaxed_metric_counts(bucket["counts"])
        parts = [f"{bucket_name} | n={bucket['n']}"]
        for key, label in RELAXED_DIAGNOSTIC_METRICS:
            short_label = {
                "Soft Trigger": "soft_trigger",
                "Relaxed Span": "relaxed_span",
                "Normalized Arg-I": "norm_arg_i",
                "Normalized Arg-C": "norm_arg_c",
                "Partial Arg-C": "partial_arg_c",
            }.get(label, label.lower().replace(" ", "_"))
            parts.append(f"{short_label}={format_float(summary.get(key, {}).get('f1', 0.0), 3)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def format_paper_summary_report(
    raw_standard_summary,
    verified_standard_summary,
    raw_relaxed_summary,
    verified_relaxed_summary,
    raw_tuple_summary,
    verified_tuple_summary,
    raw_event_summary,
    verified_event_summary,
    detail_summary,
    avg_steps,
    avg_predicted_budget,
    avg_target_steps,
    subset_scores,
):
    lines = ["*** Paper Main Table Metrics ***"]
    for key, label in STANDARD_MAIN_METRICS:
        lines.append(format_summary_line(label, verified_standard_summary.get(key, {})))
    lines.append(format_summary_line("Event-level", verified_event_summary))
    lines.append(format_summary_line("Tuple", verified_tuple_summary))
    lines.append(
        "Schema Validity | "
        f"valid_event_rate={format_float(detail_summary.get('schema', 0.0), 3)}"
    )
    lines.append(
        "Evidence Support | "
        f"citation_validity={format_float(detail_summary.get('citation_validity', 0.0), 3)} "
        f"claim_support={format_float(detail_summary.get('claim_support', 0.0), 3)}"
    )
    latent_parts = [f"used={format_float(avg_steps, 2)}"]
    if avg_predicted_budget is not None:
        latent_parts.append(f"predicted_budget={format_float(avg_predicted_budget, 2)}")
    if avg_target_steps is not None:
        latent_parts.append(f"target={format_float(avg_target_steps, 2)}")
    lines.append("Average Latent Steps | " + " ".join(latent_parts))

    lines.append("*** Paper Appendix Table Metrics ***")
    for key, label in STANDARD_APPENDIX_METRICS:
        lines.append(format_summary_line(label, verified_standard_summary.get(key, {})))
    lines.append(
        "Raw vs Verified Tuple F1 | "
        f"raw={format_float(raw_tuple_summary.get('f1', 0.0), 3)} "
        f"verified={format_float(verified_tuple_summary.get('f1', 0.0), 3)} "
        f"delta={format_float(verified_tuple_summary.get('f1', 0.0) - raw_tuple_summary.get('f1', 0.0), 3)}"
    )
    lines.append(
        "Raw vs Verified Event-level F1 | "
        f"raw={format_float(raw_event_summary.get('f1', 0.0), 3)} "
        f"verified={format_float(verified_event_summary.get('f1', 0.0), 3)} "
        f"delta={format_float(verified_event_summary.get('f1', 0.0) - raw_event_summary.get('f1', 0.0), 3)}"
    )
    lines.append("--- Relaxed Diagnostic Metrics")
    for key, label in RELAXED_DIAGNOSTIC_METRICS:
        raw_metric = raw_relaxed_summary.get(key, {})
        verified_metric = verified_relaxed_summary.get(key, {})
        strict_key = {
            "soft_trigger": "trigger",
            "relaxed_span": "span",
            "normalized_arg_i": "arg_i",
            "normalized_arg_c": "arg_c",
            "partial_argument": "arg_c",
        }.get(key)
        strict_f1 = verified_standard_summary.get(strict_key, {}).get("f1", 0.0) if strict_key else 0.0
        relaxed_f1 = verified_metric.get("f1", 0.0)
        lines.append(
            f"{label} | {format_metric_triplet(verified_metric)} "
            f"raw_f1={format_float(raw_metric.get('f1', 0.0), 3)} "
            f"strict_delta={format_float(relaxed_f1 - strict_f1, 3)}"
        )
    lines.append("--- Complex / Same-type / Multi-event subset F1")
    for subset_name in ("complex", "same_type", "multi_event", "type_confusion", "high_budget", "all"):
        if subset_name not in subset_scores:
            continue
        raw_values = subset_scores[subset_name]["raw"]
        verified_values = subset_scores[subset_name]["verified"]
        lines.append(
            f"{subset_name} | n={len(raw_values)} "
            f"raw_tuple_f1={format_float(mean_or_zero(raw_values), 3)} "
            f"verified_tuple_f1={format_float(mean_or_zero(verified_values), 3)}"
        )
    return "\n".join(lines)


def format_event_payload(payload, title):
    lines = [f"--- {title}"]
    if not payload or not isinstance(payload.get("events"), list) or not payload["events"]:
        lines.append("(no valid events)")
        return "\n".join(lines)

    for event_idx, event in enumerate(payload["events"], start=1):
        lines.append(f"[event {event_idx}]")
        if not isinstance(event, dict):
            lines.append(f"malformed_event: {event}")
            continue
        lines.append(f"event_type: {event.get('event_type', '')}")
        lines.append(f"trigger: {event.get('trigger', '')}")
        lines.append(f"evidence: {format_list(event.get('evidence', []))}")
        arguments = event.get("arguments", {})
        if isinstance(arguments, dict):
            for role in sorted(arguments.keys()):
                lines.append(f"{role}: {format_list(arguments[role])}")
    return "\n".join(lines)


def event_header_set(payload):
    if not payload or not isinstance(payload.get("events"), list):
        return set()
    headers = set()
    for event in payload["events"]:
        if not isinstance(event, dict):
            continue
        headers.add((str(event.get("event_type", "")), str(event.get("trigger", ""))))
    return headers


def set_f1(gold_items, pred_items):
    if not gold_items and not pred_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(gold_items & pred_items)
    precision = hit / len(pred_items) if pred_items else 0.0
    recall = hit / len(gold_items) if gold_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def event_header_f1(gold_payload, pred_payload):
    return set_f1(event_header_set(gold_payload), event_header_set(pred_payload))


def format_document_report(
    idx,
    example,
    eval_split,
    generation,
    selected,
    raw_f1,
    verified_f1,
    raw_standard_metrics,
    verified_standard_metrics,
    raw_relaxed_metrics,
    verified_relaxed_metrics,
    gold_payload,
    derived_target_steps,
):
    lines = [f"*** Document {idx} / {example.get('doc_id')} [{eval_split}] ***"]
    lines.append(format_complexity_report(example, derived_target_steps))
    lines.append(format_generation_report(generation))
    lines.append(format_prediction_report(selected, verified_f1, raw_f1))
    lines.append(format_paper_metrics_report(raw_standard_metrics, verified_standard_metrics))
    lines.append(format_relaxed_metrics_report(raw_relaxed_metrics, verified_relaxed_metrics))
    lines.append(format_event_payload(selected.get("payload"), "predicted events"))
    lines.append(format_event_payload(gold_payload, "gold events"))
    lines.append("*** end document ***")
    return "\n".join(lines)


def inference_only_example(example):
    return {
        "doc_id": example.get("doc_id", "unknown"),
        "sentences": list(example.get("sentences", [])),
    }


def build_direct_prediction(output_text, stop_reason=None, sentences=None):
    if stop_reason in {"eos_incomplete", "semantic_incomplete_json"}:
        return {
            "output": output_text,
            "score": 0.0,
            "detail": {
                "json": 0.0,
                "events": 0.0,
                "schema": 0.0,
                "citation_validity": 0.0,
                "claim_support": 0.0,
            },
            "payload": None,
            "raw_payload": None,
            "generated_event_count_before_dedup": 0,
            "verified_event_count": 0,
            "removed_duplicate_event_count": 0,
        }
    raw_payload = extract_json_payload(output_text)
    quality_metrics = event_payload_quality_metrics(raw_payload, sentences=sentences)
    payload = None
    generated_event_count_before_dedup = 0
    removed_duplicate_event_count = 0
    if is_extractable_event_payload(raw_payload):
        canonical_prediction_events = canonical_events(
            raw_payload.get("events", [])
        )
        generated_event_count_before_dedup = len(canonical_prediction_events)
        verified_events = deduplicate_exact_events(
            canonical_prediction_events
        )
        removed_duplicate_event_count = (
            generated_event_count_before_dedup - len(verified_events)
        )
        payload = {"events": verified_events}
    event_count = len(payload.get("events", [])) if isinstance(payload, dict) else 0
    return {
        "output": output_text,
        "score": 1.0 if payload is not None else 0.0,
        "detail": {
            "json": 1.0 if payload is not None else 0.0,
            "events": float(event_count),
            **quality_metrics,
        },
        "payload": payload,
        "raw_payload": raw_payload,
        "generated_event_count_before_dedup": generated_event_count_before_dedup,
        "verified_event_count": event_count,
        "removed_duplicate_event_count": removed_duplicate_event_count,
    }


def emergency_evaluation(model_args, data_args, training_args):
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args.train = False
    if not training_args.fix_attn_mask:
        logging.warning("Enabling fix_attn_mask for emergency event-handle evaluation.")
        training_args.fix_attn_mask = True
    if not model_args.ckpt_dir:
        raise ValueError(
            "Emergency evaluation requires --ckpt_dir because APLER's latent "
            "reasoning and event-handle heads are task-specific."
        )
    lora_config = build_lora_config(model_args, training_args)
    model = CODI(model_args, training_args, lora_config)

    if model_args.ckpt_dir:
        load_checkpoint_into_model(
            model,
            model_args.ckpt_dir,
            require_lora=bool(
                training_args.use_lora
                and not model_args.adapter_name_or_path
            ),
            require_current_modules=True,
        )
        model.codi.tie_weights()

    tokenizer = load_tokenizer_with_fallback(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    event_boundary_mode = normalize_event_boundary_control(
        getattr(training_args, "event_boundary_control", "off")
    )
    if event_boundary_mode != "off":
        boundary_token_groups = event_boundary_token_groups(
            tokenizer,
            len(tokenizer),
        )
        print(
            "Event boundary control | "
            f"mode={event_boundary_mode} "
            f"continue_tokens={len(boundary_token_groups['continue'])} "
            f"stop_tokens={len(boundary_token_groups['stop'])} "
            f"root_close_tokens={len(boundary_token_groups['root_close'])} "
            f"whitespace_tokens={len(boundary_token_groups['whitespace'])} "
            f"event_tail_tokens={len(boundary_token_groups['event_tail'])} "
            f"multi_close_tokens={len(boundary_token_groups['multi_close'])} "
            f"event_close_tokens={len(boundary_token_groups['event_close'])} "
            f"hard_max_events={int(training_args.max_event_count)}",
            flush=True,
        )

    model = model.to(runtime_device)
    if runtime_device.type == "cuda" and training_args.bf16:
        model.to(torch.bfloat16)
    elif runtime_device.type == "cuda" and training_args.fp16:
        model.to(torch.float16)
    model.eval()

    eval_split = getattr(data_args, "eval_split", "eval")
    examples = load_emergency_examples(data_args.data_name, split=eval_split)
    raw_example_count = len(examples)
    examples, skipped_zero_event = filter_positive_event_examples(examples)
    print(
        "Evaluation event filter | "
        f"raw={raw_example_count} skipped_zero_event={skipped_zero_event} kept={len(examples)}"
    )
    examples = limit_dataset_for_debug(
        examples, data_args.debug_data, data_args.debug_data_size
    )
    if getattr(training_args, "exp_mode", False):
        examples = examples[: int(max(0, getattr(training_args, "exp_data_num", len(examples))))]
    if not examples:
        raise ValueError(
            f"No emergency evaluation examples found for data_name={data_args.data_name!r}, "
            f"split={eval_split!r}."
        )
    parse_scores = []
    raw_f1_scores = []
    verified_f1_scores = []
    raw_header_f1_scores = []
    verified_header_f1_scores = []
    raw_standard_counts = empty_standard_metric_counts()
    verified_standard_counts = empty_standard_metric_counts()
    raw_relaxed_counts = empty_relaxed_metric_counts()
    verified_relaxed_counts = empty_relaxed_metric_counts()
    raw_tuple_counts = {"hit": 0, "pred": 0, "gold": 0}
    verified_tuple_counts = {"hit": 0, "pred": 0, "gold": 0}
    raw_event_counts = {"hit": 0, "pred": 0, "gold": 0}
    verified_event_counts = {"hit": 0, "pred": 0, "gold": 0}
    detail_score_sums = defaultdict(float)
    detail_score_counts = defaultdict(int)
    subset_scores = defaultdict(lambda: {"raw": [], "verified": []})
    event_count_relaxed_buckets = {
        name: {"n": 0, "counts": empty_relaxed_metric_counts()}
        for name in ("gold<=2", "gold>=3", "gold>=4")
    }
    event_count_diagnostics = {
        name: empty_event_count_diagnostics()
        for name in ("all", "gold<=2", "gold>=3", "gold>=4")
    }
    raw_event_count_diagnostics = {
        name: empty_event_count_diagnostics()
        for name in ("all", "gold<=2", "gold>=3", "gold>=4")
    }
    boundary_control_applications = 0
    boundary_opportunities = 0
    direct_boundary_gate_applications = 0
    completed_boundary_gate_applications = 0
    boundary_continue_decisions = 0
    boundary_close_decisions = 0
    boundary_noncommit_decisions = 0
    decoder_handle_advances = 0
    direct_continue_candidates = 0
    direct_stop_candidates = 0
    hard_event_cap_samples = 0
    duplicate_event_count = 0
    repeated_event_samples = 0
    generated_event_count_before_dedup = 0
    verified_event_count = 0
    empty_output_count = 0
    incomplete_json_count = 0
    document_metric_records = []
    handle_expected_count_errors = []
    boundary_qwen_continue_means = []
    boundary_handle_continue_means = []
    boundary_blended_continue_means = []
    latent_steps = []
    predicted_budgets = []
    budget_expected_values = []
    budget_top1_values = []
    budget_entropy_values = []
    budget_top1_margin_values = []
    budget_proposal_histogram = defaultdict(int)
    budget_target_histogram = defaultdict(int)
    derived_target_steps_all = []
    total_examples = len(examples)
    evaluation_start_time = time.perf_counter()
    sample_durations = []
    good_sample_reports = []
    good_sample_threshold = float(getattr(training_args, "good_sample_f1_threshold", 0.50))
    good_sample_peak_threshold = float(getattr(training_args, "good_sample_relaxed_peak_threshold", 0.60))
    print(
        f"*** Emergency Evaluation Progress *** split={eval_split} total={total_examples}",
        flush=True,
    )

    for idx, example in enumerate(examples):
        sample_start_time = time.perf_counter()
        derived_target_steps = example_complexity(example, training_args.num_latent)
        derived_target_steps_all.append(float(derived_target_steps))
        budget_target_histogram[int(derived_target_steps)] += 1
        public_example = inference_only_example(example)
        prompt = build_event_prompt(public_example)
        prompt = format_prompt_for_model(
            prompt,
            tokenizer,
            use_chat_template=training_args.use_chat_template,
        )
        generation = generate_event_prediction(
            model,
            tokenizer,
            prompt,
            training_args,
            runtime_device,
            example=public_example,
        )
        selected = build_direct_prediction(
            generation["text"],
            stop_reason=generation.get("stop_reason"),
            sentences=example.get("sentences", []),
        )
        gold_payload = {"events": canonical_events(example["events"])}
        raw_f1 = tuple_f1(gold_payload, selected.get("raw_payload"))
        verified_f1 = tuple_f1(gold_payload, selected["payload"])
        raw_header_f1 = event_header_f1(gold_payload, selected.get("raw_payload"))
        verified_header_f1 = event_header_f1(gold_payload, selected["payload"])
        raw_standard_metrics = standard_event_metrics(gold_payload, selected.get("raw_payload"))
        verified_standard_metrics = standard_event_metrics(gold_payload, selected["payload"])
        raw_relaxed_metrics = relaxed_event_metrics(gold_payload, selected.get("raw_payload"))
        verified_relaxed_metrics = relaxed_event_metrics(gold_payload, selected["payload"])
        add_standard_metric_counts(raw_standard_counts, raw_standard_metrics)
        add_standard_metric_counts(verified_standard_counts, verified_standard_metrics)
        add_relaxed_metric_counts(raw_relaxed_counts, raw_relaxed_metrics)
        add_relaxed_metric_counts(verified_relaxed_counts, verified_relaxed_metrics)
        gold_event_count = len(gold_payload["events"])
        selected_payload = selected.get("payload")
        selected_events = (
            selected_payload.get("events", [])
            if isinstance(selected_payload, dict)
            and isinstance(selected_payload.get("events", []), list)
            else []
        )
        pred_event_count = len(selected_events)
        sample_generated_event_count = int(
            selected.get(
                "generated_event_count_before_dedup",
                pred_event_count,
            )
            or 0
        )
        sample_duplicate_count = int(
            selected.get(
                "removed_duplicate_event_count",
                max(0, sample_generated_event_count - pred_event_count),
            )
            or 0
        )
        generation["generated_event_count"] = sample_generated_event_count
        generation["verified_event_count"] = pred_event_count
        generation["duplicate_event_count"] = sample_duplicate_count
        generated_event_count_before_dedup += sample_generated_event_count
        verified_event_count += pred_event_count
        duplicate_event_count += sample_duplicate_count
        repeated_event_samples += int(sample_duplicate_count > 0)
        sample_empty_output = pred_event_count == 0
        sample_incomplete_json = is_incomplete_json_generation(
            generation.get("stop_reason")
        )
        empty_output_count += int(sample_empty_output)
        incomplete_json_count += int(sample_incomplete_json)
        add_event_count_diagnostic(
            event_count_diagnostics["all"],
            gold_event_count,
            pred_event_count,
        )
        add_event_count_diagnostic(
            raw_event_count_diagnostics["all"],
            gold_event_count,
            sample_generated_event_count,
        )
        for bucket_name in event_count_bucket_names(gold_event_count):
            add_event_count_diagnostic(
                event_count_diagnostics[bucket_name],
                gold_event_count,
                pred_event_count,
            )
            add_event_count_diagnostic(
                raw_event_count_diagnostics[bucket_name],
                gold_event_count,
                sample_generated_event_count,
            )
            event_count_relaxed_buckets[bucket_name]["n"] += 1
            add_relaxed_metric_counts(
                event_count_relaxed_buckets[bucket_name]["counts"],
                verified_relaxed_metrics,
            )
        boundary_control_applications += int(
            generation.get("event_boundary_applications", 0) or 0
        )
        boundary_opportunities += int(
            generation.get("event_boundary_opportunities", 0) or 0
        )
        direct_boundary_gate_applications += int(
            generation.get("direct_boundary_gate_applications", 0) or 0
        )
        completed_boundary_gate_applications += int(
            generation.get("completed_boundary_gate_applications", 0) or 0
        )
        boundary_continue_decisions += int(
            generation.get("event_boundary_continue_decisions", 0) or 0
        )
        boundary_close_decisions += int(
            generation.get("event_boundary_close_decisions", 0) or 0
        )
        boundary_noncommit_decisions += int(
            generation.get("event_boundary_noncommit_decisions", 0) or 0
        )
        decoder_handle_advances += int(
            generation.get("decoder_handle_advances", 0) or 0
        )
        direct_continue_candidates += int(
            generation.get("direct_continue_candidates", 0) or 0
        )
        direct_stop_candidates += int(
            generation.get("direct_stop_candidates", 0) or 0
        )
        hard_event_cap_samples += int(
            bool(generation.get("hard_event_cap_hit", False))
        )
        handle_expected_count = generation.get("event_handle_expected_count")
        if handle_expected_count is not None:
            handle_expected_count_errors.append(
                abs(float(handle_expected_count) - gold_event_count)
            )
        for generation_key, destination in (
            (
                "event_boundary_qwen_continue_mean",
                boundary_qwen_continue_means,
            ),
            (
                "event_boundary_handle_continue_mean",
                boundary_handle_continue_means,
            ),
            (
                "event_boundary_blended_continue_mean",
                boundary_blended_continue_means,
            ),
        ):
            value = generation.get(generation_key)
            if value is not None:
                destination.append(float(value))
        gold_tuple_items = event_tuples(gold_payload)
        raw_tuple_items = event_tuples(selected.get("raw_payload"))
        verified_tuple_items = event_tuples(selected["payload"])
        gold_event_items = event_header_set(gold_payload)
        raw_event_items = event_header_set(selected.get("raw_payload"))
        verified_event_items = event_header_set(selected["payload"])
        add_set_metric_counts(raw_tuple_counts, gold_tuple_items, raw_tuple_items)
        add_set_metric_counts(verified_tuple_counts, gold_tuple_items, verified_tuple_items)
        add_set_metric_counts(raw_event_counts, gold_event_items, raw_event_items)
        add_set_metric_counts(verified_event_counts, gold_event_items, verified_event_items)
        for detail_name, detail_value in selected.get("detail", {}).items():
            try:
                detail_score_sums[detail_name] += float(detail_value)
                detail_score_counts[detail_name] += 1
            except (TypeError, ValueError):
                pass

        parse_scores.append(selected["score"])
        raw_f1_scores.append(raw_f1["f1"])
        verified_f1_scores.append(verified_f1["f1"])
        raw_header_f1_scores.append(raw_header_f1["f1"])
        verified_header_f1_scores.append(verified_header_f1["f1"])
        features = example.get("complexity_features", {})
        subset_names = ["all"]
        subset_names.append("single_event" if len(example.get("events", [])) == 1 else "multi_event")
        is_complex = (
            len(example.get("events", [])) >= 3
            or derived_target_steps >= 7
            or bool(features.get("same_type_events"))
            or bool(features.get("type_confusion"))
            or bool(features.get("tertiary_event"))
        )
        if is_complex:
            subset_names.append("complex")
        if features.get("same_type_events"):
            subset_names.append("same_type")
        if features.get("type_confusion"):
            subset_names.append("type_confusion")
        if derived_target_steps >= 7:
            subset_names.append("high_budget")
        for subset_name in subset_names:
            subset_scores[subset_name]["raw"].append(raw_f1["f1"])
            subset_scores[subset_name]["verified"].append(verified_f1["f1"])
        document_metric_records.append(
            {
                "evaluation_index": int(idx),
                "record_sha256": canonical_record_sha256(example),
                "doc_id": str(example.get("doc_id", "unknown")),
                "subsets": sorted(set(subset_names)),
                "stop_reason": str(generation.get("stop_reason", "unknown")),
                "parse_score": float(selected.get("score", 0.0)),
                "empty_output": bool(sample_empty_output),
                "incomplete_json": bool(sample_incomplete_json),
                "valid_event_rate": float(
                    selected.get("detail", {}).get("schema", 0.0)
                ),
                "gold_event_count": int(gold_event_count),
                "predicted_event_count": int(pred_event_count),
                "generated_event_count": int(sample_generated_event_count),
                "verified_event_count": int(pred_event_count),
                "duplicate_event_count": int(sample_duplicate_count),
                "boundary_applications": int(
                    generation.get("event_boundary_applications", 0) or 0
                ),
                "boundary_continue_decisions": int(
                    generation.get("event_boundary_continue_decisions", 0) or 0
                ),
                "boundary_close_decisions": int(
                    generation.get("event_boundary_close_decisions", 0) or 0
                ),
                "boundary_noncommit_decisions": int(
                    generation.get("event_boundary_noncommit_decisions", 0) or 0
                ),
                "generated_tokens": int(
                    generation.get("generated_tokens", 0) or 0
                ),
                "raw_tuple": set_metric_record(
                    gold_tuple_items,
                    raw_tuple_items,
                ),
                "verified_tuple": set_metric_record(
                    gold_tuple_items,
                    verified_tuple_items,
                ),
                "verified_arg_c": machine_metric_record(
                    verified_standard_metrics.get("arg_c", {})
                ),
                "verified_event_level": set_metric_record(
                    gold_event_items,
                    verified_event_items,
                ),
            }
        )
        latent_steps.append(generation.get("latent_steps", 0))
        if generation.get("predicted_budget") is not None:
            predicted_budgets.append(generation["predicted_budget"])
            budget_proposal_histogram[int(generation["predicted_budget"])] += 1
        if generation.get("budget_expected") is not None:
            budget_expected_values.append(generation["budget_expected"])
        if generation.get("budget_top1") is not None:
            budget_top1_values.append(generation["budget_top1"])
        if generation.get("budget_entropy") is not None:
            budget_entropy_values.append(generation["budget_entropy"])
        if generation.get("budget_top1_margin") is not None:
            budget_top1_margin_values.append(generation["budget_top1_margin"])

        sample_elapsed = time.perf_counter() - sample_start_time
        sample_durations.append(sample_elapsed)
        elapsed = time.perf_counter() - evaluation_start_time
        completed = idx + 1
        remaining = max(0, total_examples - completed)
        avg_sample_time = sum(sample_durations) / len(sample_durations) if sample_durations else 0.0
        eta = avg_sample_time * remaining
        print(
            f"[{eval_split} progress] {completed}/{total_examples} "
            f"doc_id={example.get('doc_id')} "
            f"tokens={int(generation.get('generated_tokens', 0) or 0)} "
            f"stop={generation.get('stop_reason', 'unknown')} "
            f"sample_time={format_duration(sample_elapsed)} "
            f"avg={format_duration(avg_sample_time)} "
            f"elapsed={format_duration(elapsed)} "
            f"eta={format_duration(eta)}",
            flush=True,
        )

        relaxed_f1_values = {
            key: float(verified_relaxed_metrics.get(key, {}).get("f1", 0.0))
            for key, _label in RELAXED_DIAGNOSTIC_METRICS
        }
        good_sample_min_relaxed = min(relaxed_f1_values.values()) if relaxed_f1_values else 0.0
        good_sample_peak_relaxed = max(relaxed_f1_values.values()) if relaxed_f1_values else 0.0
        good_sample_avg_relaxed = (
            sum(relaxed_f1_values.values()) / len(relaxed_f1_values)
            if relaxed_f1_values
            else 0.0
        )
        is_good_sample = (
            good_sample_min_relaxed >= good_sample_threshold
            and good_sample_peak_relaxed > good_sample_peak_threshold
        )
        document_report = None
        if do_print or is_good_sample:
            document_report = format_document_report(
                idx,
                example,
                eval_split,
                generation,
                selected,
                raw_f1,
                verified_f1,
                raw_standard_metrics,
                verified_standard_metrics,
                raw_relaxed_metrics,
                verified_relaxed_metrics,
                gold_payload,
                derived_target_steps,
            )
        if is_good_sample and document_report is not None:
            relaxed_detail_text = " ".join(
                f"{key}_f1={value:.3f}" for key, value in sorted(relaxed_f1_values.items())
            )
            good_sample_reports.append(
                (
                    (good_sample_avg_relaxed, good_sample_min_relaxed, good_sample_peak_relaxed),
                    (
                        f"good_sample_min_relaxed_f1={good_sample_min_relaxed:.3f} "
                        f"good_sample_avg_relaxed_f1={good_sample_avg_relaxed:.3f} "
                        f"good_sample_peak_relaxed_f1={good_sample_peak_relaxed:.3f} "
                        f"{relaxed_detail_text}\n"
                        f"{document_report}"
                    ),
                )
            )
        if do_print and document_report is not None:
            print(document_report)
            print("")

    avg_parse_score = sum(parse_scores) / len(parse_scores) if parse_scores else 0.0
    avg_raw_f1 = sum(raw_f1_scores) / len(raw_f1_scores) if raw_f1_scores else 0.0
    avg_verified_f1 = sum(verified_f1_scores) / len(verified_f1_scores) if verified_f1_scores else 0.0
    avg_raw_header_f1 = sum(raw_header_f1_scores) / len(raw_header_f1_scores) if raw_header_f1_scores else 0.0
    avg_verified_header_f1 = sum(verified_header_f1_scores) / len(verified_header_f1_scores) if verified_header_f1_scores else 0.0
    avg_steps = sum(latent_steps) / len(latent_steps) if latent_steps else 0.0
    raw_standard_summary = finalize_standard_metric_counts(raw_standard_counts)
    verified_standard_summary = finalize_standard_metric_counts(verified_standard_counts)
    raw_relaxed_summary = finalize_relaxed_metric_counts(raw_relaxed_counts)
    verified_relaxed_summary = finalize_relaxed_metric_counts(verified_relaxed_counts)
    raw_tuple_summary = finalize_set_metric_counts(raw_tuple_counts)
    verified_tuple_summary = finalize_set_metric_counts(verified_tuple_counts)
    raw_event_summary = finalize_set_metric_counts(raw_event_counts)
    verified_event_summary = finalize_set_metric_counts(verified_event_counts)
    detail_summary = {
        key: detail_score_sums[key] / detail_score_counts[key]
        for key in detail_score_sums
        if detail_score_counts[key] > 0
    }
    avg_predicted_budget = sum(predicted_budgets) / len(predicted_budgets) if predicted_budgets else None
    avg_budget_expected = (
        sum(budget_expected_values) / len(budget_expected_values)
        if budget_expected_values
        else None
    )
    avg_budget_top1 = (
        sum(budget_top1_values) / len(budget_top1_values)
        if budget_top1_values
        else None
    )
    avg_budget_entropy = (
        sum(budget_entropy_values) / len(budget_entropy_values)
        if budget_entropy_values
        else None
    )
    avg_budget_top1_margin = (
        sum(budget_top1_margin_values) / len(budget_top1_margin_values)
        if budget_top1_margin_values
        else None
    )
    avg_target_steps = None
    if derived_target_steps_all:
        avg_target_steps = sum(derived_target_steps_all) / len(derived_target_steps_all)
    total_evaluation_time = time.perf_counter() - evaluation_start_time
    avg_sample_time = sum(sample_durations) / len(sample_durations) if sample_durations else 0.0
    min_sample_time = min(sample_durations) if sample_durations else 0.0
    max_sample_time = max(sample_durations) if sample_durations else 0.0
    summary_beijing_time = beijing_time_text()
    summary_lines = [
        "*** Emergency Summary ***",
        f"summary_beijing_time: {summary_beijing_time}",
        f"split: {eval_split}",
        "timing | "
        f"total={format_duration(total_evaluation_time)} "
        f"avg_per_sample={format_duration(avg_sample_time)} "
        f"min_sample={format_duration(min_sample_time)} "
        f"max_sample={format_duration(max_sample_time)} "
        f"samples={len(sample_durations)}",
        f"parse_score: {avg_parse_score:.3f}",
        "raw_tuple_micro | "
        f"p={raw_tuple_summary['precision']:.3f} "
        f"r={raw_tuple_summary['recall']:.3f} "
        f"f1={raw_tuple_summary['f1']:.3f}",
        "verified_tuple_micro | "
        f"p={verified_tuple_summary['precision']:.3f} "
        f"r={verified_tuple_summary['recall']:.3f} "
        f"f1={verified_tuple_summary['f1']:.3f}",
        f"raw_tuple_docavg_f1: {avg_raw_f1:.3f}",
        f"verified_tuple_docavg_f1: {avg_verified_f1:.3f}",
        f"raw_event_header_docavg_f1: {avg_raw_header_f1:.3f}",
        f"verified_event_header_docavg_f1: {avg_verified_header_f1:.3f}",
        f"tuple_f1: {verified_tuple_summary['f1']:.3f}",
        format_paper_summary_report(
            raw_standard_summary,
            verified_standard_summary,
            raw_relaxed_summary,
            verified_relaxed_summary,
            raw_tuple_summary,
            verified_tuple_summary,
            raw_event_summary,
            verified_event_summary,
            detail_summary,
            avg_steps,
            avg_predicted_budget,
            avg_target_steps,
            subset_scores,
        ),
        f"average_adaptive_latent_steps: {avg_steps:.2f}",
    ]
    if avg_predicted_budget is not None:
        summary_lines.append(f"average_predicted_budget: {avg_predicted_budget:.2f}")
    if avg_budget_expected is not None:
        summary_lines.append(f"average_expected_budget: {avg_budget_expected:.2f}")
    if avg_budget_top1 is not None:
        summary_lines.append(f"average_top1_budget: {avg_budget_top1:.2f}")
    if avg_budget_entropy is not None:
        summary_lines.append(f"average_budget_entropy: {avg_budget_entropy:.3f}")
    if avg_budget_top1_margin is not None:
        summary_lines.append(
            f"average_budget_top1_margin: {avg_budget_top1_margin:.3f}"
        )
    if budget_target_histogram or budget_proposal_histogram:
        histogram_steps = sorted(
            set(budget_target_histogram) | set(budget_proposal_histogram)
        )
        summary_lines.append(
            "latent budget histogram | "
            + " ".join(
                f"T{step}:target={budget_target_histogram[step]}"
                f"/proposal={budget_proposal_histogram[step]}"
                for step in histogram_steps
            )
        )
    if avg_target_steps is not None:
        summary_lines.append(f"average_target_latent_steps: {avg_target_steps:.2f}")
    summary_lines.append("--- subset tuple_f1")
    for subset_name in sorted(subset_scores.keys()):
        raw_values = subset_scores[subset_name]["raw"]
        verified_values = subset_scores[subset_name]["verified"]
        raw_value = sum(raw_values) / len(raw_values) if raw_values else 0.0
        verified_value = sum(verified_values) / len(verified_values) if verified_values else 0.0
        summary_lines.append(
            f"{subset_name}: n={len(raw_values)} "
            f"raw={raw_value:.3f} verified={verified_value:.3f}"
        )
    summary_lines.append(
        format_event_count_diagnostic_summary(
            raw_event_count_diagnostics,
            title="--- raw event count diagnostics (before exact dedup)",
        )
    )
    summary_lines.append(
        format_event_count_diagnostic_summary(event_count_diagnostics)
    )
    summary_lines.append(
        "--- event boundary control\n"
        f"mode={normalize_event_boundary_control(getattr(training_args, 'event_boundary_control', 'off'))} "
        f"applications={boundary_control_applications} "
        f"opportunities={boundary_opportunities} "
        f"direct_gates={direct_boundary_gate_applications} "
        f"completed_gates={completed_boundary_gate_applications} "
        f"continue={boundary_continue_decisions} "
        f"close={boundary_close_decisions} "
        f"noncommit={boundary_noncommit_decisions} "
        f"commit_rate={(boundary_continue_decisions + boundary_close_decisions) / max(boundary_control_applications, 1):.3f} "
        f"handle_advances={decoder_handle_advances} "
        f"decisions_per_raw_event={(boundary_continue_decisions + boundary_close_decisions) / max(generated_event_count_before_dedup, 1):.3f} "
        f"direct_continue_candidates={direct_continue_candidates} "
        f"direct_stop_candidates={direct_stop_candidates} "
        f"hard_cap_samples={hard_event_cap_samples} "
        f"events_before_dedup={generated_event_count_before_dedup} "
        f"events_after_dedup={verified_event_count} "
        f"duplicate_events={duplicate_event_count} "
        f"duplicate_event_samples={repeated_event_samples} "
        f"handle_expected_mae={mean_or_zero(handle_expected_count_errors):.3f} "
        f"qwen_continue_mean={mean_or_zero(boundary_qwen_continue_means):.3f} "
        f"handle_continue_mean={mean_or_zero(boundary_handle_continue_means):.3f} "
        f"blended_continue_mean={mean_or_zero(boundary_blended_continue_means):.3f}"
    )
    summary_lines.append(format_relaxed_bucket_summary(event_count_relaxed_buckets))
    summary_lines.append("***")
    print("\n".join(summary_lines))

    report_timestamp = beijing_time_for_filename()
    if bool(getattr(training_args, "write_eval_report", True)):
        report_dir = str(getattr(training_args, "eval_report_dir", "") or "").strip()
        if not report_dir:
            report_dir = os.path.join(training_args.output_dir, "eval_reports")
        report_dir = os.path.abspath(report_dir)
        os.makedirs(report_dir, exist_ok=True)
        report_name = (
            f"{safe_filename_part(training_args.expt_name)}_"
            f"{safe_filename_part(data_args.data_name)}_"
            f"{safe_filename_part(eval_split)}_"
            f"{report_timestamp}.txt"
        )
        report_path = os.path.join(report_dir, report_name)
        tmp_report_path = report_path + ".tmp"
        good_sample_reports.sort(key=lambda item: item[0], reverse=True)
        max_good_samples = max(0, int(getattr(training_args, "max_good_sample_reports", 8)))
        selected_good_reports = good_sample_reports[:max_good_samples]
        report_lines = [
            "*** APLER Emergency Evaluation Report ***",
            f"report_beijing_time: {summary_beijing_time}",
            f"data_name: {data_args.data_name}",
            f"split: {eval_split}",
            f"expt_name: {training_args.expt_name}",
            f"model_name_or_path: {model_args.model_name_or_path}",
            f"ckpt_dir: {model_args.ckpt_dir}",
            "",
            *summary_lines,
            "",
            "*** Good Samples ***",
            f"selection_rule: relaxed diagnostic F1 min >= {good_sample_threshold:.3f} and max > {good_sample_peak_threshold:.3f}",
            f"selected={len(selected_good_reports)} available={len(good_sample_reports)} max_saved={max_good_samples}",
        ]
        if selected_good_reports:
            for rank, (_rank_score, sample_report) in enumerate(selected_good_reports, start=1):
                report_lines.extend(["", f"### Good Sample Rank {rank}", sample_report])
        else:
            report_lines.append("(no samples met the good-sample threshold)")
        with open(tmp_report_path, "w", encoding="utf-8", newline="\n") as report_file:
            report_file.write("\n".join(report_lines).rstrip() + "\n")
        os.replace(tmp_report_path, report_path)
        print(f"Evaluation report saved to: {report_path}")

    if bool(getattr(training_args, "write_eval_json_report", True)):
        explicit_json_report_path = str(
            getattr(training_args, "eval_json_report_path", "") or ""
        ).strip()
        json_report_dir = str(
            getattr(training_args, "eval_json_report_dir", "") or ""
        ).strip()
        if not json_report_dir:
            json_report_dir = str(
                getattr(training_args, "eval_report_dir", "") or ""
            ).strip()
        if not json_report_dir:
            json_report_dir = os.path.join(
                training_args.output_dir,
                "eval_reports",
            )
        event_count_total = event_count_diagnostics["all"]
        event_count_mae = (
            event_count_total["absolute_error_sum"]
            / max(1, int(event_count_total["n"]))
        )
        json_payload = build_machine_readable_evaluation_payload(
            generated_at=summary_beijing_time,
            data_name=data_args.data_name,
            eval_split=eval_split,
            expt_name=training_args.expt_name,
            model_name_or_path=model_args.model_name_or_path,
            ckpt_dir=model_args.ckpt_dir,
            raw_tuple_summary=raw_tuple_summary,
            verified_tuple_summary=verified_tuple_summary,
            raw_tuple_docavg_f1=avg_raw_f1,
            verified_tuple_docavg_f1=avg_verified_f1,
            verified_standard_summary=verified_standard_summary,
            verified_event_summary=verified_event_summary,
            parse_score=avg_parse_score,
            empty_output_count=empty_output_count,
            incomplete_json_count=incomplete_json_count,
            valid_event_rate=detail_summary.get("schema", 0.0),
            event_count_mae=event_count_mae,
            duplicate_event_count=duplicate_event_count,
            repeated_event_samples=repeated_event_samples,
            single_event_docavg_f1=mean_or_zero(
                subset_scores["single_event"]["verified"]
            ),
            multi_event_docavg_f1=mean_or_zero(
                subset_scores["multi_event"]["verified"]
            ),
            documents=document_metric_records,
        )
        json_report_name = (
            f"{safe_filename_part(training_args.expt_name)}_"
            f"{safe_filename_part(data_args.data_name)}_"
            f"{safe_filename_part(eval_split)}_"
            f"{report_timestamp}.metrics.json"
        )
        json_report_path = os.path.abspath(
            explicit_json_report_path
            if explicit_json_report_path
            else os.path.join(json_report_dir, json_report_name)
        )
        write_json_atomically(json_payload, json_report_path)
        print(f"Machine-readable evaluation report saved to: {json_report_path}")
    return 100 * avg_verified_f1


def evaluation(model_args, data_args, training_args):
    data_args.data_name = require_data_name(data_args.data_name)
    is_event_task = resolve_event_task(data_args.data_name, data_args.task_type)
    model_args.train = False
    if not is_event_task:
        raise ValueError(
            "This slimmed evaluation entrypoint only supports emergency event extraction. "
            "Use --task_type emergency and an emergency DATA_NAME."
        )
    if not training_args.fix_attn_mask:
        logging.warning("Enabling fix_attn_mask for emergency event-handle evaluation.")
        training_args.fix_attn_mask = True
    return emergency_evaluation(model_args, data_args, training_args)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.data_name = require_data_name(data_args.data_name)

    accu = evaluation(model_args, data_args, training_args)
    print(f"Emergency verified tuple F1: {accu:.3f}")
