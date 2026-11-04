"""Paper section 3.3: training-derived structural budget supervision."""
import math
import re
import unicodedata

MIN_LATENT_BUDGET = 1
MAX_LATENT_BUDGET = 9


def normalized_trigger(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def derive_budget(example, statistics, max_steps=9):
    if not statistics or not {"sentence_p75", "evidence_p75"} <= statistics.keys():
        raise ValueError("Budget labels require persisted training-only statistics.")
    events = example.get("events", [])
    types = [e["event_type"] for e in events]
    triggers = [normalized_trigger(e.get("trigger", "")) for e in events]
    evidence = {s for e in events for s in e.get("evidence", [])}
    scale = min(6, (max(1, len(events)) - 1).bit_length())
    context = int(len(example["sentences"]) >= statistics["sentence_p75"]
                  or len(evidence) >= statistics["evidence_p75"])
    ambiguous = int(len(types) != len(set(types)) or len(triggers) != len(set(triggers)))
    heterogeneous = int(len(set(types)) > 1)
    return max(1, min(max_steps, 1 + scale + context + ambiguous + heterogeneous))


def gold_budget_probability(progress):
    return max(0.0, min(1.0, 1.0 - float(progress) / 0.30))


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute statistics from an empty training partition")
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def training_statistics(rows):
    return {
        "sentence_p75": percentile([len(r["sentences"]) for r in rows], .75),
        "evidence_p75": percentile([len({s for e in r["events"] for s in e.get("evidence", [])}) for r in rows], .75),
        "instance_capacity": max(1, min(32, math.ceil(percentile([len(r["events"]) for r in rows], .99)))),
        "training_documents": len(rows),
        "quantile_method": "linear interpolation; capacity ceil; all training rows before token checks",
        "kappa": 6,
    }
