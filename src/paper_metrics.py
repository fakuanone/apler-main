"""Paper 4.3: source spans, one-to-one event assignment, micro counts."""
import json
from src.set_utils import minimum_cost_bipartite_assignment
from src.span_grounding import grounded_events
from src.latent_budget import normalized_trigger


def parse_prediction(text):
    try:
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {"events"} or not isinstance(value["events"], list):
            return None
        for event in value["events"]:
            if not isinstance(event, dict) or not isinstance(event.get("event_type"), str) or not isinstance(event.get("trigger"), str):
                return None
            if set(event) != {"event_type", "trigger", "evidence", "arguments"} or not event["event_type"].strip() or not event["trigger"].strip():
                return None
            if not isinstance(event["evidence"], list) or any(not isinstance(s, str) for s in event["evidence"]):
                return None
            if not isinstance(event.get("arguments"), dict):
                return None
            if any(not isinstance(v, list) or any(not isinstance(x, str) for x in v) for v in event["arguments"].values()):
                return None
        return value
    except (ValueError, TypeError):
        return None


def metric(hit, pred, gold):
    p, r = (hit / pred if pred else 0.0), (hit / gold if gold else 0.0)
    return {"hit": hit, "pred": pred, "gold": gold, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def overlap_count(gold, pred, role_sensitive):
    if not gold or not pred:
        return 0
    costs = [[-int(bool(gs & ps) and (not role_sensitive or gr == pr))
              for gr, gs in gold] for pr, ps in pred]
    return -sum(costs[i][j] for i, j in minimum_cost_bipartite_assignment(costs))


def text_events(events):
    """Explicit fallback for the supplied string-only annotations; never invent offsets."""
    return [{"type": e.get("event_type", ""), "trigger": frozenset([normalized_trigger(e.get("trigger", ""))]),
             "arguments": [(role, frozenset([normalized_trigger(value)]))
                           for role, values in e.get("arguments", {}).items() for value in values]}
            for e in events]


def score_document(example, payload, mode="paper_span"):
    if mode == "paper_span":
        gold = grounded_events(example, example["events"], gold=True)
        pred = grounded_events(example, (payload or {}).get("events", []))
    elif mode == "dataset_text":
        gold = text_events(example["events"])
        pred = text_events((payload or {}).get("events", []))
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    pairs = {}
    scale = 1 + sum(len(e["arguments"]) for e in gold + pred)
    costs = []
    for i, pe in enumerate(pred):
        row = []
        for j, ge in enumerate(gold):
            compatible = pe["type"] == ge["type"] and bool(pe["trigger"] & ge["trigger"])
            ai = overlap_count(ge["arguments"], pe["arguments"], False) if compatible else 0
            ac = overlap_count(ge["arguments"], pe["arguments"], True) if compatible else 0
            pairs[i, j] = (compatible, ai, ac)
            # Tiny final tie-break rewards compatible zero-argument matches.
            row.append(-(ai * scale + ac + 1 / (scale * (1 + len(pred) + len(gold)))) if compatible else 1e12)
        costs.append(row)
    matched = [(i, j) for i, j in minimum_cost_bipartite_assignment(costs) if pairs[i, j][0]] if gold and pred else []
    trigger_cost = [[-int(pairs[i, j][0]) for j in range(len(gold))] for i in range(len(pred))]
    trig = -sum(trigger_cost[i][j] for i, j in minimum_cost_bipartite_assignment(trigger_cost)) if gold and pred else 0
    ai, ac, similarity = 0, 0, 0.0
    for i, j in matched:
        _, hi, hc = pairs[i, j]
        ai += hi; ac += hc
        den = len(pred[i]["arguments"]) + len(gold[j]["arguments"])
        similarity += 2 * hc / den if den else 1.0
    np, ng = sum(len(e["arguments"]) for e in pred), sum(len(e["arguments"]) for e in gold)
    return {"trigger": metric(trig, len(pred), len(gold)), "arg_i": metric(ai, np, ng),
            "arg_c": metric(ac, np, ng), "event": metric(similarity, len(pred), len(gold)),
            "count_correct": int(len(pred) == len(gold)), "parse_valid": int(payload is not None)}


def aggregate(records):
    result = {}
    for key in ("trigger", "arg_i", "arg_c", "event"):
        result[key] = metric(*(sum(r[key][field] for r in records) for field in ("hit", "pred", "gold")))
    result["count_accuracy"] = sum(r["count_correct"] for r in records) / max(1, len(records))
    result["parse_rate"] = sum(r["parse_valid"] for r in records) / max(1, len(records))
    result["documents"] = len(records)
    return result
