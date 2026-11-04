"""Deterministic four-stage teacher and label-free student tokenization."""
import json
from src.latent_budget import derive_budget
from src.span_grounding import locate, document_text, sentence_ranges

STAGES = ("EVIDENCE", "EVENT_SEMANTICS", "INSTANCE_ORGANIZATION", "ARGUMENT_OWNERSHIP")


def anchor_index(event, example):
    indices = [int(s[1:]) for s in event.get("evidence", []) if str(s).startswith("S") and str(s)[1:].isdigit()]
    indices = sorted({i for i in indices if i < len(example["sentences"])})
    if not indices:
        raise ValueError(f"{example['doc_id']}: event has no valid evidence sentence")
    supported = [i for i in indices if locate(example["sentences"][i], event["trigger"])]
    return supported[0] if supported else indices[0]


def ordered_events(example):
    def first_trigger(e):
        if e.get("trigger_span") is not None:
            return e["trigger_span"][0]
        spans = locate(document_text(example), e["trigger"])
        bounds = sentence_ranges(example)
        evidence = [bounds[int(s[1:])] for s in e.get("evidence", []) if str(s)[1:].isdigit() and int(s[1:]) < len(bounds)]
        supported = [s for s in spans if any(a <= s[0] < s[1] <= b for a, b in evidence)]
        return min((s[0] for s in supported or spans), default=10**12)
    return sorted(example.get("events", []), key=lambda e: (first_trigger(e), e["event_type"], json.dumps(e["arguments"], ensure_ascii=False, sort_keys=True)))


def event_semantic_text(event, example):
    support = [f"{s}: {example['sentences'][int(s[1:])]}" for s in event.get("evidence", []) if str(s)[1:].isdigit() and int(s[1:]) < len(example["sentences"])]
    return json.dumps({"type": event["event_type"], "trigger": event["trigger"], "support": support, "arguments": event["arguments"]}, ensure_ascii=False, sort_keys=True)


def teacher_sections(example, schema=None):
    """Build stage-specific supervision from annotations, without resolving ambiguous spans.

    Stage 3 partitions each same-type group by its observed features instead of
    repeating every event pair. Stage 4 records ownership independently of text
    identity: a shared value may legitimately belong to several annotated events.
    """
    events = ordered_events(example)
    if not events:
        return ["No annotated evidence supports any event.", "No event category is supported.",
                "The event instance set is empty.", "No role arguments belong to an event."]
    text, bounds, locations = document_text(example), sentence_ranges(example), {}

    def evidence_ids(event):
        return sorted({int(s[1:]) for s in event.get("evidence", [])
                       if str(s).startswith("S") and str(s)[1:].isdigit()
                       and int(s[1:]) < len(bounds)})

    def grounding(value, evidence, explicit=None):
        if value not in locations:
            locations[value] = locate(text, value)
        found = locations[value]
        supported = [span for span in found if any(
            bounds[i][0] <= span[0] < span[1] <= bounds[i][1] for i in evidence)]
        if explicit is not None:
            span = tuple(explicit) if isinstance(explicit, (list, tuple)) else ()
            spans = [span] if span in found else []
        else:
            spans = supported or found
        return {"candidates": spans,
                "status": "unlocated" if not spans else "unique" if len(spans) == 1 else "ambiguous",
                "basis": "annotated_span" if explicit is not None else "text_match",
                "within_evidence": bool(spans) and all(span in supported for span in spans)}

    ids = [f"E{i}" for i in range(len(events))]
    evidence = [evidence_ids(event) for event in events]
    triggers = [grounding(e["trigger"], support, e.get("trigger_span")) for e, support in zip(events, evidence)]
    sections = [{"offsets": "Half-open character offsets in newline-joined document; candidates are not inferred gold spans.",
                 "sentences": {f"S{i}": example["sentences"][i] for i in sorted({s for ev in evidence for s in ev})},
                 "triggers": []},
                {"confirmations": []},
                {"order": ids, "order_basis": "Document trigger position; ambiguous positions use the first candidate, with deterministic label tie-breaks. This is not inferred temporal order.",
                 "same_type_groups": []},
                {"rule": "Bind each role/value only to its listed annotated owners. Preserve the complete source wording, including negation. Shared values do not merge instances.",
                 "bindings": [], "events_without_arguments": []}]
    groups, ownership = {}, {}
    argument_mentions = [{} for _ in events]
    for i, event in enumerate(events):
        support = [f"S{s}" for s in evidence[i]]
        sections[0]["triggers"].append({"event": ids[i], "text": event["trigger"], "evidence": support, **triggers[i]})
        semantic = {"event": ids[i], "event_type": event["event_type"], "trigger": event["trigger"],
                    "evidence": support, "annotated_roles": sorted(role for role, values in event["arguments"].items() if values)}
        description = (schema or {}).get(event["event_type"], {}).get("description")
        if description:
            semantic["type_meaning"] = description
        sections[1]["confirmations"].append(semantic)
        groups.setdefault(event["event_type"], []).append(i)
        has_arguments = False
        for role, values in sorted(event["arguments"].items()):
            explicit = event.get("argument_spans", {}).get(role, [None] * len(values))
            if len(explicit) != len(values):
                raise ValueError(f"{example.get('doc_id')}: argument_spans length mismatch")
            for value, span in sorted(zip(values, explicit), key=lambda pair: (pair[0], str(pair[1]))):
                has_arguments = True
                source = grounding(value, evidence[i], span)
                ownership.setdefault((role, value), []).append({"event": ids[i], **source})
                argument_mentions[i].setdefault(role, []).append({"value": value, "candidates": source["candidates"]})
        if not has_arguments:
            sections[3]["events_without_arguments"].append(ids[i])

    for event_type, members in groups.items():
        if len(members) < 2:
            continue
        roles = sorted({role for i in members for role in events[i]["arguments"]})
        features = {}
        for i in members:
            features[i] = {"trigger": events[i]["trigger"], "trigger_candidates": triggers[i]["candidates"],
                           "evidence": [f"S{s}" for s in evidence[i]],
                           **{f"arguments.{role}": sorted(set(events[i]["arguments"].get(role, []))) for role in roles}}
            # Equal strings at different annotated positions can distinguish instances.
            for role in roles:
                if any(role in events[j].get("argument_spans", {}) for j in members):
                    features[i][f"argument_candidates.{role}"] = argument_mentions[i].get(role, [])
        shared, contrasts, indistinguishable = [], {}, {}
        for field in features[members[0]]:
            partitions = {}
            for i in members:
                value = features[i][field]
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                partitions.setdefault(key, {"value": value, "events": []})["events"].append(ids[i])
            if len(partitions) == 1:
                shared.append(field)
            else:
                contrasts[field] = list(partitions.values())
        for i in members:
            signature = json.dumps(features[i], ensure_ascii=False, sort_keys=True)
            indistinguishable.setdefault(signature, []).append(ids[i])
        sections[2]["same_type_groups"].append(
            {"event_type": event_type, "members": [ids[i] for i in members], "shared_fields": shared,
             "contrasts": contrasts, "unresolved_members": [v for v in indistinguishable.values() if len(v) > 1]})
    sections[2]["boundary_rule"] = "Use concrete contrasts to keep instances separate. Unresolved members retain annotation multiplicity; do not invent a distinguishing fact."
    sections[3]["bindings"] = [{"role": role, "value": value, "owners": owners}
                               for (role, value), owners in sorted(ownership.items())]
    return [json.dumps(section, ensure_ascii=False, separators=(",", ":")) for section in sections]


def build_prompt(example, schema, language="zh"):
    instruction = ("抽取文档中有证据支持的所有独立事件。输出严格 JSON，仅含 events 数组；每个事件含 event_type、trigger、evidence、arguments。trigger 和 arguments 必须使用原文连续文本；没有事件时输出空数组。不要输出推理过程。\n"
                   if language == "zh" else "Extract all independently supported events as strict JSON with an events array. Each event has event_type, trigger, evidence, arguments. Copy contiguous source spans for triggers and arguments. Return an empty array when no event is supported. Do not output reasoning.\n")
    instruction += json.dumps(schema, ensure_ascii=False) + "\nDOCUMENT\n"
    text, body_spans, marker_spans = instruction, [], []
    for i, sentence in enumerate(example["sentences"]):
        text += f"S{i}: "
        start = len(text); text += sentence; body_spans.append((start, len(text)))
        text += " "; start = len(text); text += f"[END_S{i}]"; marker_spans.append((start, len(text)))
        text += "\n"
    return text, body_spans, marker_spans


def tokenize_example(example, tokenizer, schema, config, statistics, *, training=False):
    prompt, body, markers = build_prompt(example, schema, config["language"])
    if getattr(tokenizer, "chat_template", None):
        wrapped = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        shift = wrapped.find(prompt)
        if shift < 0:
            raise ValueError("Chat template must preserve document text and sentence markers")
        body = [(a + shift, b + shift) for a, b in body]; markers = [(a + shift, b + shift) for a, b in markers]
        prompt = wrapped
    def encode(text):
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    encoded = encode(prompt); offsets = encoded["offset_mapping"]
    def intersect(span, positions=offsets):
        a, b = span
        return [i for i, (x, y) in enumerate(positions) if x < b and y > a and y > x]
    if len(encoded["input_ids"]) > config["input_window"]:
        raise ValueError(f"{example['doc_id']}: input exceeds {config['input_window']} tokens; no silent truncation")
    sentence_positions = []
    for span in markers:
        hits = intersect(span)
        if not hits:
            raise ValueError("Sentence end marker could not be token-aligned")
        sentence_positions.append(hits[-1])
    result = {"input_ids": encoded["input_ids"], "sentence_positions": sentence_positions,
              "document_positions": sorted({i for span in body for i in intersect(span)})}
    if not training:
        return result
    events = ordered_events(example)
    target = {"events": [{k: e[k] for k in ("event_type", "trigger", "evidence", "arguments")} for e in events]}
    ids = tokenizer.encode(json.dumps(target, ensure_ascii=False, separators=(",", ":")), add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    ids.append(tokenizer.eos_token_id)
    if len(ids) > config["max_output_tokens"]:
        raise ValueError(f"{example['doc_id']}: complete target exceeds output token limit; labels were not truncated")
    teacher, spans = prompt, []
    for name, content in zip(STAGES, teacher_sections(example, schema)):
        teacher += f"\n<{name}>\n"; start = len(teacher); teacher += content
        spans.append((start, len(teacher))); teacher += f"\n</{name}>\n"
    reference = encode(teacher)
    ranges = []
    for span in spans:
        hits = intersect(span, reference["offset_mapping"])
        if not hits: raise ValueError("Empty teacher stage token range")
        ranges.append((hits[0], hits[-1] + 1))
    result.update(target_ids=ids, target_steps=derive_budget(example, statistics), teacher_ids=reference["input_ids"],
                  stage_ranges=ranges, gold_anchors=[anchor_index(e, example) for e in events],
                  event_reference_ids=[tokenizer.encode(event_semantic_text(e, example), add_special_tokens=False) for e in events])
    return result
