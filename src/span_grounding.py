"""Exact text grounding with reversible normalization and explicit gold audits."""
import re
import unicodedata


class GoldSpanError(ValueError):
    pass


def normalize_with_offsets(text):
    output, offsets = [], []
    # Keep combining sequences together so NFC contraction retains source offsets.
    for match in re.finditer(r"\s+|[^\s]", text):
        start, end = match.span()
        char = match.group()
        if char.isspace():
            output.append(" "); offsets.append((start, end))
        elif unicodedata.combining(char) and output:
            previous = output.pop(); old_start, _ = offsets.pop()
            normalized = unicodedata.normalize("NFC", previous + char)
            output.extend(normalized); offsets.extend([(old_start, end)] * len(normalized))
        else:
            normalized = unicodedata.normalize("NFC", char)
            output.extend(normalized); offsets.extend([(start, end)] * len(normalized))
    return "".join(output), offsets


def document_text(example):
    return "\n".join(str(s) for s in example["sentences"])


def sentence_ranges(example):
    result, cursor = [], 0
    for sentence in example["sentences"]:
        result.append((cursor, cursor + len(sentence)))
        cursor += len(sentence) + 1
    return result


def locate(text, value):
    haystack, offsets = normalize_with_offsets(text)
    needle = normalize_with_offsets(str(value))[0].strip()
    if not needle:
        return []
    spans, start = [], 0
    while (index := haystack.find(needle, start)) >= 0:
        spans.append((offsets[index][0], offsets[index + len(needle) - 1][1]))
        start = index + 1
    return list(dict.fromkeys(spans))


def candidates(example, value, explicit=None, evidence=None):
    text = document_text(example)
    found = locate(text, value)
    if explicit is not None:
        span = tuple(explicit) if isinstance(explicit, (list, tuple)) else ()
        return [span] if span in found else []
    if evidence:
        bounds = sentence_ranges(example)
        indices = [int(s[1:]) for s in evidence if re.fullmatch(r"S\d+", str(s))]
        supported = [span for span in found if any(
            i < len(bounds) and bounds[i][0] <= span[0] < span[1] <= bounds[i][1] for i in indices)]
        if supported:
            return supported
    return found


def grounded_events(example, events, *, gold=False):
    result = []
    for index, event in enumerate(events):
        evidence = event.get("evidence", [])
        trigger = candidates(example, event.get("trigger", ""), event.get("trigger_span"), evidence)
        arguments = []
        for role, raw_values in event.get("arguments", {}).items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            explicit = event.get("argument_spans", {}).get(role, [None] * len(values))
            if len(explicit) != len(values):
                raise ValueError(f"{example.get('doc_id')}: argument_spans length mismatch")
            for value, span in zip(values, explicit):
                spans = candidates(example, value, span, evidence)
                if gold and len(spans) != 1:
                    raise GoldSpanError(f"{example.get('doc_id')} event {index} role {role}: {len(spans)} candidate spans for {value!r}")
                arguments.append((role, frozenset(spans)))
        if gold and len(trigger) != 1:
            raise GoldSpanError(f"{example.get('doc_id')} event {index}: {len(trigger)} trigger spans for {event.get('trigger')!r}")
        result.append({"type": event.get("event_type", ""), "trigger": frozenset(trigger), "arguments": arguments})
    return result


def audit_gold(rows):
    issues = []
    for row in rows:
        try:
            grounded_events(row, row["events"], gold=True)
        except (ValueError, TypeError, KeyError) as exc:
            issues.append({"doc_id": row.get("doc_id"), "error": str(exc)})
    return {"documents": len(rows), "valid_documents": len(rows) - len(issues), "issues": issues}
