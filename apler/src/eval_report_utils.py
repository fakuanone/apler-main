import hashlib
import json
import os


def machine_metric_record(metric):
    metric = metric or {}
    return {
        "hit": int(metric.get("hit", 0)),
        "pred": int(metric.get("pred", 0)),
        "gold": int(metric.get("gold", 0)),
        "precision": float(metric.get("precision", 0.0)),
        "recall": float(metric.get("recall", 0.0)),
        "f1": float(metric.get("f1", 0.0)),
    }


def is_incomplete_json_generation(stop_reason):
    return str(stop_reason or "") in {
        "max_tokens",
        "eos_incomplete",
        "semantic_incomplete_json",
    }


def canonical_record_sha256(record):
    canonical_text = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def build_machine_readable_evaluation_payload(
    *,
    generated_at,
    data_name,
    eval_split,
    expt_name,
    model_name_or_path,
    ckpt_dir,
    raw_tuple_summary,
    verified_tuple_summary,
    raw_tuple_docavg_f1,
    verified_tuple_docavg_f1,
    verified_standard_summary,
    verified_event_summary,
    parse_score,
    empty_output_count,
    incomplete_json_count,
    valid_event_rate,
    event_count_mae,
    duplicate_event_count,
    repeated_event_samples,
    single_event_docavg_f1,
    multi_event_docavg_f1,
    documents,
):
    verified_arg_c = machine_metric_record(
        (verified_standard_summary or {}).get("arg_c", {})
    )
    verified_event_level = machine_metric_record(verified_event_summary)
    raw_tuple_micro = machine_metric_record(raw_tuple_summary)
    verified_tuple_micro = machine_metric_record(verified_tuple_summary)
    return {
        "schema_version": 1,
        "generated_at": str(generated_at),
        "data": {
            "data_name": str(data_name),
            "split": str(eval_split),
            "sample_count": len(documents),
        },
        "checkpoint": {
            "experiment_name": str(expt_name),
            "model_name_or_path": str(model_name_or_path),
            "path": os.path.abspath(str(ckpt_dir)),
        },
        "metrics": {
            "raw_tuple_micro": raw_tuple_micro,
            "verified_tuple_micro": verified_tuple_micro,
            "raw_tuple_micro_f1": raw_tuple_micro["f1"],
            "verified_tuple_micro_f1": verified_tuple_micro["f1"],
            "raw_tuple_docavg_f1": float(raw_tuple_docavg_f1),
            "verified_tuple_docavg_f1": float(verified_tuple_docavg_f1),
            "arg_c": verified_arg_c,
            "arg_c_f1": verified_arg_c["f1"],
            "event_level": verified_event_level,
            "event_level_f1": verified_event_level["f1"],
            "parse_score": float(parse_score),
            "empty_output_count": int(empty_output_count),
            "incomplete_json_count": int(incomplete_json_count),
            "valid_event_rate": float(valid_event_rate),
            "event_count_mae": float(event_count_mae),
            "duplicate_event_count": int(duplicate_event_count),
            "duplicate_event_sample_count": int(repeated_event_samples),
            "single_event_tuple_docavg_f1": float(single_event_docavg_f1),
            "multi_event_tuple_docavg_f1": float(multi_event_docavg_f1),
        },
        "metric_definitions": {
            "tuple_micro": "Set-based tuple counts pooled across documents.",
            "tuple_docavg_f1": "Arithmetic mean of per-document set-based tuple F1.",
            "arg_c": "Verified strict argument classification pooled counts.",
            "event_level": "Verified event_type and trigger pooled counts.",
            "empty_output_count": "Documents with no extractable predicted events.",
            "incomplete_json_count": "Documents stopped by max_tokens, eos_incomplete, or semantic_incomplete_json.",
            "event_count_mae": "Mean absolute event-count error after exact duplicate removal.",
            "duplicate_event_count": "Canonical events removed because all event fields exactly matched an earlier event.",
            "single_multi_event_f1": "Verified per-document tuple F1 averaged within the gold event-count subset.",
        },
        "documents": list(documents),
    }


def write_json_atomically(payload, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output_path = output_path + ".tmp"
    with open(tmp_output_path, "w", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(tmp_output_path, output_path)
