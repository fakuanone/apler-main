import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


PROCESS_MARKERS = [
    "<EVIDENCE_SELECTION>",
    "<EVENT_TYPE>",
    "<INSTANCE_SEPARATION>",
    "<EVENT_PLAN>",
    "<ROLE_FILLING>",
    "<FINAL_JSON>",
]

PROCESS_DONE_MARKERS = [
    "<EVIDENCE_DONE>",
    "<TYPE_DONE>",
    "<INSTANCE_DONE>",
    "<ORGANIZATION_DONE>",
    "<ARGUMENT_DONE>",
]

FINAL_JSON_MARKER = "<FINAL_JSON>"
MIN_LATENT_BUDGET = 3
JSON_EVENT_ARRAY_PREFIX = '{"events":['
EVENT_OBJECT_START_PREFIXES = ("{", '{"', '{"event_type"', '{"event_type":')


JsonPathPart = Union[str, int]
JsonPath = Tuple[JsonPathPart, ...]


@dataclass(frozen=True)
class JsonKeySpan:
    path: JsonPath
    key: str
    start: int
    name_end: int
    end: int


@dataclass(frozen=True)
class JsonStringSpan:
    path: JsonPath
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class EventJsonSpanIndex:
    root_prefix_span: Tuple[int, int]
    event_object_spans: Tuple[Tuple[int, int], ...]
    event_header_spans: Tuple[Tuple[int, int], ...]
    event_context_spans: Tuple[Tuple[int, int], ...]
    key_spans: Tuple[JsonKeySpan, ...]
    string_spans: Tuple[JsonStringSpan, ...]
    punctuation_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class EventDecoderTokenization:
    text: str
    input_ids: Tuple[int, ...]
    offset_mapping: Tuple[Tuple[int, int], ...]
    prefix_end: int
    json_token_start: int
    json_token_end: int
    spans: EventJsonSpanIndex


class _JsonSpanParser:
    """Parse JSON while retaining exact character spans for decoder supervision."""

    def __init__(self, text: str):
        self.text = str(text)
        self.length = len(self.text)
        self.pos = 0
        self.key_spans: List[JsonKeySpan] = []
        self.string_spans: List[JsonStringSpan] = []
        self.object_spans: List[Tuple[JsonPath, int, int]] = []
        self.punctuation_spans: List[Tuple[int, int]] = []

    def parse(self) -> Any:
        value = self._parse_value(())
        self._skip_ws()
        if self.pos != self.length:
            raise ValueError(f"Unexpected trailing JSON content at character {self.pos}.")
        return value

    def _skip_ws(self) -> None:
        while self.pos < self.length and self.text[self.pos].isspace():
            self.pos += 1

    def _record_punctuation(self, position: int) -> None:
        self.punctuation_spans.append((position, position + 1))

    def _parse_string(self) -> Tuple[str, int, int, int, int]:
        if self.pos >= self.length or self.text[self.pos] != '"':
            raise ValueError(f"Expected JSON string at character {self.pos}.")
        raw_start = self.pos
        self.pos += 1
        content_start = self.pos
        escaped = False
        while self.pos < self.length:
            char = self.text[self.pos]
            if escaped:
                escaped = False
                self.pos += 1
                continue
            if char == "\\":
                escaped = True
                self.pos += 1
                continue
            if char == '"':
                content_end = self.pos
                self.pos += 1
                raw_end = self.pos
                return (
                    json.loads(self.text[raw_start:raw_end]),
                    raw_start,
                    raw_end,
                    content_start,
                    content_end,
                )
            self.pos += 1
        raise ValueError(f"Unterminated JSON string beginning at character {raw_start}.")

    def _parse_value(self, path: JsonPath) -> Any:
        self._skip_ws()
        if self.pos >= self.length:
            raise ValueError("Unexpected end of JSON input.")
        char = self.text[self.pos]
        if char == "{":
            return self._parse_object(path)
        if char == "[":
            return self._parse_array(path)
        if char == '"':
            value, _raw_start, _raw_end, content_start, content_end = self._parse_string()
            self.string_spans.append(
                JsonStringSpan(path=path, value=str(value), start=content_start, end=content_end)
            )
            return value

        start = self.pos
        while self.pos < self.length and self.text[self.pos] not in ",]} \t\r\n":
            self.pos += 1
        if self.pos <= start:
            raise ValueError(f"Invalid JSON value at character {start}.")
        return json.loads(self.text[start:self.pos])

    def _parse_object(self, path: JsonPath) -> Dict[str, Any]:
        start = self.pos
        self._record_punctuation(self.pos)
        self.pos += 1
        result: Dict[str, Any] = {}
        self._skip_ws()
        if self.pos < self.length and self.text[self.pos] == "}":
            self._record_punctuation(self.pos)
            self.pos += 1
            self.object_spans.append((path, start, self.pos))
            return result

        while True:
            self._skip_ws()
            key, key_start, key_raw_end, _content_start, _content_end = self._parse_string()
            self._skip_ws()
            if self.pos >= self.length or self.text[self.pos] != ":":
                raise ValueError(f"Expected ':' after JSON key at character {self.pos}.")
            self._record_punctuation(self.pos)
            self.pos += 1
            key_end = self.pos
            key_text = str(key)
            self.key_spans.append(
                JsonKeySpan(
                    path=path,
                    key=key_text,
                    start=key_start,
                    name_end=key_raw_end,
                    end=key_end,
                )
            )
            result[key_text] = self._parse_value(path + (key_text,))
            self._skip_ws()
            if self.pos >= self.length:
                raise ValueError(f"Unterminated JSON object beginning at character {start}.")
            if self.text[self.pos] == "}":
                self._record_punctuation(self.pos)
                self.pos += 1
                self.object_spans.append((path, start, self.pos))
                return result
            if self.text[self.pos] != ",":
                raise ValueError(f"Expected ',' in JSON object at character {self.pos}.")
            self._record_punctuation(self.pos)
            self.pos += 1

    def _parse_array(self, path: JsonPath) -> List[Any]:
        start = self.pos
        self._record_punctuation(self.pos)
        self.pos += 1
        result: List[Any] = []
        self._skip_ws()
        if self.pos < self.length and self.text[self.pos] == "]":
            self._record_punctuation(self.pos)
            self.pos += 1
            return result

        item_index = 0
        while True:
            result.append(self._parse_value(path + (item_index,)))
            item_index += 1
            self._skip_ws()
            if self.pos >= self.length:
                raise ValueError(f"Unterminated JSON array beginning at character {start}.")
            if self.text[self.pos] == "]":
                self._record_punctuation(self.pos)
                self.pos += 1
                return result
            if self.text[self.pos] != ",":
                raise ValueError(f"Expected ',' in JSON array at character {self.pos}.")
            self._record_punctuation(self.pos)
            self.pos += 1


def build_event_json_span_index(text: str) -> EventJsonSpanIndex:
    """Index the canonical event JSON without re-tokenizing isolated fragments."""
    text = str(text)
    if not text.startswith(JSON_EVENT_ARRAY_PREFIX):
        raise ValueError(
            "Event decoder JSON must start with the canonical root prefix "
            f"{JSON_EVENT_ARRAY_PREFIX!r}."
        )
    parser = _JsonSpanParser(text)
    payload = parser.parse()
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("Event decoder JSON must contain a top-level events list.")

    event_object_records = sorted(
        (
            (path, start, end)
            for path, start, end in parser.object_spans
            if len(path) == 2 and path[0] == "events" and isinstance(path[1], int)
        ),
        key=lambda item: int(item[0][1]),
    )
    if len(event_object_records) != len(payload["events"]):
        raise ValueError(
            "Event-object span count does not match the parsed events list: "
            f"spans={len(event_object_records)} events={len(payload['events'])}."
        )

    event_object_spans: List[Tuple[int, int]] = []
    event_header_spans: List[Tuple[int, int]] = []
    event_context_spans: List[Tuple[int, int]] = []
    for path, object_start, object_end in event_object_records:
        event_object_spans.append((object_start, object_end))
        header = next(
            (
                record
                for record in parser.key_spans
                if record.path == path and record.key == "event_type"
            ),
            None,
        )
        if header is None:
            raise ValueError(f"Event object at path {path!r} has no event_type key.")
        event_header_spans.append((object_start, header.end))

        context_position = object_start - 1
        while context_position >= 0 and text[context_position].isspace():
            context_position -= 1
        if context_position < 0 or text[context_position] not in "[,":
            raise ValueError(
                f"Cannot locate the causal event-start context before character {object_start}."
            )
        event_context_spans.append((context_position, context_position + 1))

    return EventJsonSpanIndex(
        root_prefix_span=(0, len(JSON_EVENT_ARRAY_PREFIX)),
        event_object_spans=tuple(event_object_spans),
        event_header_spans=tuple(event_header_spans),
        event_context_spans=tuple(event_context_spans),
        key_spans=tuple(parser.key_spans),
        string_spans=tuple(parser.string_spans),
        punctuation_spans=tuple(parser.punctuation_spans),
    )


def _normalise_tokenizer_result(values: Any) -> List[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values or [])


def encode_json_event_array_prefix(tokenizer: Any) -> List[int]:
    """Encode the one root prefix shared by training and autoregressive decoding."""
    return encode_text_no_special_tokens(JSON_EVENT_ARRAY_PREFIX, tokenizer)


def _encode_json_segment_with_offsets(
    text: str,
    tokenizer: Any,
    alignment_tokenizer: Any,
    *,
    offset_shift: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    token_ids = encode_text_no_special_tokens(text, tokenizer)
    try:
        aligned = alignment_tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            truncation=False,
        )
    except Exception as exc:
        raise TypeError(
            "Decoder JSON alignment requires a fast tokenizer with offset mapping support."
        ) from exc
    aligned_ids = [int(item) for item in _normalise_tokenizer_result(aligned["input_ids"])]
    raw_offsets = _normalise_tokenizer_result(aligned.get("offset_mapping", []))
    offsets = [
        (int(start) + int(offset_shift), int(end) + int(offset_shift))
        for start, end in raw_offsets
    ]
    if token_ids != aligned_ids:
        raise ValueError(
            "The primary and alignment tokenizers produced different decoder JSON ids; "
            "refusing to train with inconsistent tokenization."
        )
    if len(token_ids) != len(offsets):
        raise ValueError(
            "Decoder JSON token/offset length mismatch: "
            f"tokens={len(token_ids)} offsets={len(offsets)}."
        )
    return token_ids, offsets


def tokenize_event_decoder_sequence(
    text: str,
    tokenizer: Any,
    alignment_tokenizer: Any,
    *,
    leading_token_ids: Sequence[int],
    trailing_token_ids: Sequence[int],
    max_json_tokens: Optional[int] = None,
) -> EventDecoderTokenization:
    """Tokenize canonical event JSON with a train/test-stable root boundary."""
    text = str(text)
    spans = build_event_json_span_index(text)
    body = text[len(JSON_EVENT_ARRAY_PREFIX) :]
    prefix_ids, prefix_offsets = _encode_json_segment_with_offsets(
        JSON_EVENT_ARRAY_PREFIX,
        tokenizer,
        alignment_tokenizer,
        offset_shift=0,
    )
    body_ids, body_offsets = _encode_json_segment_with_offsets(
        body,
        tokenizer,
        alignment_tokenizer,
        offset_shift=len(JSON_EVENT_ARRAY_PREFIX),
    )
    json_ids = prefix_ids + body_ids
    if max_json_tokens is not None and len(json_ids) > int(max_json_tokens):
        raise ValueError(
            "Event decoder JSON exceeds max_json_tokens after stable-boundary tokenization: "
            f"tokens={len(json_ids)} limit={int(max_json_tokens)}."
        )
    decoded = tokenizer.decode(
        json_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text:
        raise ValueError(
            "Stable-boundary decoder JSON tokenization failed exact round-trip: "
            f"expected={text[:120]!r} decoded={decoded[:120]!r}."
        )

    leading = [int(token_id) for token_id in leading_token_ids]
    trailing = [int(token_id) for token_id in trailing_token_ids]
    offsets = [(-1, -1)] * len(leading) + prefix_offsets + body_offsets + [(-1, -1)] * len(trailing)
    input_ids = leading + json_ids + trailing
    if len(input_ids) != len(offsets):
        raise ValueError(
            "Full decoder token/offset length mismatch: "
            f"tokens={len(input_ids)} offsets={len(offsets)}."
        )
    json_token_start = len(leading)
    json_token_end = json_token_start + len(json_ids)
    return EventDecoderTokenization(
        text=text,
        input_ids=tuple(input_ids),
        offset_mapping=tuple(offsets),
        prefix_end=json_token_start + len(prefix_ids),
        json_token_start=json_token_start,
        json_token_end=json_token_end,
        spans=spans,
    )


def token_mask_from_char_spans(
    offset_mapping: Sequence[Tuple[int, int]],
    char_spans: Sequence[Tuple[int, int]],
) -> List[bool]:
    """Map character spans onto the exact tokens produced for the full JSON."""
    valid_spans = [
        (int(start), int(end))
        for start, end in char_spans
        if int(end) > int(start)
    ]
    mask = []
    for token_start, token_end in offset_mapping:
        token_start = int(token_start)
        token_end = int(token_end)
        if token_start < 0 or token_end <= token_start:
            mask.append(False)
            continue
        mask.append(
            any(token_end > span_start and token_start < span_end for span_start, span_end in valid_spans)
        )
    return mask


def encode_text_no_special_tokens(text: str, tokenizer: Any) -> List[int]:
    """Encode text without tokenizer-added special tokens and strip a leading BOS."""
    try:
        ids = tokenizer.encode(str(text), add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(str(text), add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token_id) for token_id in ids]
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if ids and bos_token_id is not None and ids[0] == bos_token_id:
        ids = ids[1:]
    return ids


def event_object_start_candidate_token_ids(
    tokenizer: Any,
    vocab_size: Optional[int] = None,
) -> Set[int]:
    """Return the unchanged whitelist used at a top-level event-object start."""
    limit = int(len(tokenizer) if vocab_size is None else vocab_size)
    candidate_ids: Set[int] = set()
    for prefix in EVENT_OBJECT_START_PREFIXES:
        ids = encode_text_no_special_tokens(prefix, tokenizer)
        if ids and 0 <= int(ids[0]) < limit:
            candidate_ids.add(int(ids[0]))
    return candidate_ids


def find_all_token_subsequence_starts(
    tokens: Sequence[int],
    pattern: Sequence[int],
) -> List[int]:
    if not pattern:
        return []
    token_list = list(tokens)
    pattern_list = list(pattern)
    limit = len(token_list) - len(pattern_list) + 1
    starts = []
    for start in range(max(0, limit)):
        if token_list[start : start + len(pattern_list)] == pattern_list:
            starts.append(start)
    return starts


def find_token_subsequence_end(tokens: Sequence[int], pattern: Sequence[int]) -> int:
    starts = find_all_token_subsequence_starts(tokens, pattern)
    return starts[0] + len(pattern) if starts else -1


def _sentence_line_spans_from_source(
    tokens: Sequence[int],
    tokenizer: Any,
    source_text: Optional[str],
) -> Tuple[List[Tuple[int, int]], int, int]:
    """Return exact token spans for Sx lines plus matched/expected counts."""
    if not source_text:
        return [], 0, 0
    document_lines = re.findall(r"(?m)^S\d+:\s*.*$", str(source_text))
    spans: List[Tuple[int, int]] = []
    matched_lines = 0
    for line in document_lines:
        pattern = encode_text_no_special_tokens(line, tokenizer)
        starts = find_all_token_subsequence_starts(tokens, pattern)
        if starts:
            matched_lines += 1
        for start in starts:
            spans.append((start, start + len(pattern)))
    return spans, matched_lines, len(document_lines)


def _document_lines_from_source(source_text: Optional[str]) -> List[str]:
    if not source_text:
        return []
    return re.findall(r"(?m)^S\d+:\s*.*$", str(source_text))


def _document_tail_start(
    tokens: Sequence[int],
    tokenizer: Any,
    source_text: Optional[str],
    *,
    after_position: int = -1,
) -> int:
    """Find the first token after the contiguous Sx document block, if possible."""
    if not source_text:
        return -1
    matches = list(re.finditer(r"(?m)^S\d+:\s*.*$", str(source_text)))
    if not matches:
        return -1
    tail_text = str(source_text)[matches[-1].end() :]
    for line in tail_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        for char_len in (96, 64, 40, 24, 12):
            fragment = candidate[: min(len(candidate), char_len)].strip()
            if len(fragment) < 4:
                continue
            pattern = encode_text_no_special_tokens(fragment, tokenizer)
            starts = [
                start
                for start in find_all_token_subsequence_starts(tokens, pattern)
                if start > int(after_position)
            ]
            if starts:
                return starts[0]
        break
    return -1


def _sentence_marker_starts(
    tokens: Sequence[int],
    tokenizer: Any,
    max_sentence_markers: int,
) -> List[int]:
    starts = []
    for sent_idx in range(max(1, int(max_sentence_markers))):
        marker_ids = encode_text_no_special_tokens(f"S{sent_idx}:", tokenizer)
        marker_end = find_token_subsequence_end(tokens, marker_ids)
        if marker_end >= 0:
            starts.append(marker_end - len(marker_ids))
    return sorted(set(start for start in starts if start >= 0))


def build_sentence_rep_positions_from_source(
    token_ids: Sequence[int],
    tokenizer: Any,
    source_text: Optional[str] = None,
    *,
    max_sentence_markers: int = 32,
) -> List[int]:
    """Locate one causal sentence representation position for every Sx line.

    Event handles need a state that has already consumed the current sentence.
    Therefore each valid entry points to the final token of the corresponding
    ``Sx: ...`` line, not to the sentence-start marker. Positions remain in the
    original Sx index space; a sentence that cannot be located safely is -1.
    """
    tokens = [int(token_id) for token_id in token_ids]
    marker_limit = max(1, int(max_sentence_markers))
    positions = [-1] * marker_limit
    if not tokens or not source_text:
        return positions

    line_matches = list(re.finditer(r"(?m)^S(\d+):\s*.*$", str(source_text)))
    line_records = []
    for match in line_matches:
        sentence_index = int(match.group(1))
        if not 0 <= sentence_index < marker_limit:
            continue
        line_text = match.group(0)
        body = line_text.split(":", 1)[1].strip() if ":" in line_text else ""
        if not body:
            continue
        line_records.append((sentence_index, line_text))
    if not line_records:
        return positions

    marker_spans: Dict[int, Tuple[int, int]] = {}
    marker_cursor = 0
    for sentence_index, _line_text in line_records:
        marker_ids = encode_text_no_special_tokens(f"S{sentence_index}:", tokenizer)
        starts = [
            start
            for start in find_all_token_subsequence_starts(tokens, marker_ids)
            if start >= marker_cursor
        ]
        if not starts:
            continue
        marker_start = starts[0]
        marker_end = marker_start + len(marker_ids)
        marker_spans[sentence_index] = (marker_start, marker_end)
        marker_cursor = marker_end

    last_marker_start = max(
        (start for start, _end in marker_spans.values()),
        default=-1,
    )
    tail_start = _document_tail_start(
        tokens,
        tokenizer,
        source_text,
        after_position=last_marker_start,
    )
    ordered_marker_starts = sorted(
        (start, sentence_index)
        for sentence_index, (start, _end) in marker_spans.items()
    )
    next_marker_start = {
        sentence_index: (
            ordered_marker_starts[offset + 1][0]
            if offset + 1 < len(ordered_marker_starts)
            else -1
        )
        for offset, (_start, sentence_index) in enumerate(ordered_marker_starts)
    }

    def token_span_is_whitespace(start: int, end: int) -> bool:
        if end <= start:
            return True
        try:
            decoded = tokenizer.decode(
                tokens[start:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(tokens[start:end])
        return not str(decoded).strip()

    exact_spans: Dict[int, Tuple[int, int]] = {}
    exact_cursor = 0
    for sentence_index, line_text in line_records:
        line_ids = encode_text_no_special_tokens(line_text, tokenizer)
        starts = [
            start
            for start in find_all_token_subsequence_starts(tokens, line_ids)
            if start >= exact_cursor
        ]
        marker_span = marker_spans.get(sentence_index)
        if marker_span is not None:
            starts = [start for start in starts if start == marker_span[0]]
        elif len(starts) != 1:
            # Without an independently located marker, a repeated full line is
            # ambiguous and must not be accepted as the sentence representation.
            starts = []
        if not starts:
            continue
        line_start = starts[0]
        line_end = line_start + len(line_ids)
        following_marker = next_marker_start.get(sentence_index, -1)
        safe_boundary = following_marker if following_marker > line_start else -1
        if safe_boundary < 0 and tail_start > line_start:
            safe_boundary = tail_start
        if safe_boundary < line_end or not token_span_is_whitespace(
            line_end, safe_boundary
        ):
            # A token subsequence can be only a prefix of a longer prompt line
            # (for example ``S0: a`` inside ``S0: alphabet``). Treat it as an
            # exact sentence only when the rest of that marker-delimited line is
            # whitespace.
            continue
        exact_spans[sentence_index] = (line_start, line_end)
        exact_cursor = line_end

    def last_content_token_before(boundary: int, lower_bound: int) -> int:
        """Walk actual prompt tokens backwards, skipping only whitespace."""
        position = min(int(boundary), len(tokens)) - 1
        floor = max(0, int(lower_bound))
        while position >= floor:
            try:
                token_text = tokenizer.decode(
                    [tokens[position]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                token_text = tokenizer.decode([tokens[position]])
            if str(token_text).strip():
                return position
            position -= 1
        return -1

    for sentence_index, _line_text in line_records:
        exact_span = exact_spans.get(sentence_index)
        if exact_span is not None:
            positions[sentence_index] = exact_span[1] - 1
            continue

        marker_span = marker_spans.get(sentence_index)
        if marker_span is None:
            continue
        marker_start, marker_end = marker_span
        following_marker = next_marker_start.get(sentence_index, -1)
        safe_boundary = following_marker if following_marker > marker_start else -1
        if safe_boundary < 0 and tail_start > marker_start:
            safe_boundary = tail_start
        if safe_boundary < 0:
            continue
        positions[sentence_index] = last_content_token_before(
            safe_boundary,
            marker_end,
        )

    return positions


def build_document_token_mask_from_source(
    token_ids: Sequence[int],
    tokenizer: Any,
    source_text: Optional[str] = None,
    *,
    max_sentence_markers: int = 32,
) -> List[bool]:
    """Build a shared train/test mask for document sentence tokens.

    The mask is intentionally limited to S0:/S1:/... document content. It first
    uses exact sentence-line matches. If any document line fails to match, both
    training and inference fall back to marker-delimited sentence spans instead
    of extending the mask to the prompt tail or BOT boundary.
    """
    tokens = [int(token_id) for token_id in token_ids]
    mask = [False] * len(tokens)
    if not tokens:
        return mask

    exact_spans, matched_lines, expected_lines = _sentence_line_spans_from_source(
        tokens,
        tokenizer,
        source_text,
    )
    if expected_lines > 0 and matched_lines == expected_lines:
        for start, end in exact_spans:
            for index in range(max(0, start), min(len(mask), end)):
                mask[index] = True
        return mask

    marker_starts = _sentence_marker_starts(tokens, tokenizer, max_sentence_markers)
    if not marker_starts:
        return mask

    exact_end_by_start = {}
    for start, end in exact_spans:
        exact_end_by_start[start] = max(exact_end_by_start.get(start, start), end)

    document_lines = _document_lines_from_source(source_text)
    tail_start = _document_tail_start(tokens, tokenizer, source_text)
    for index, start in enumerate(marker_starts):
        next_start = marker_starts[index + 1] if index + 1 < len(marker_starts) else len(tokens)
        if start in exact_end_by_start:
            end = exact_end_by_start[start]
        elif index < len(document_lines):
            estimated_len = len(encode_text_no_special_tokens(document_lines[index], tokenizer))
            end = start + estimated_len if estimated_len > 0 else next_start
        else:
            end = next_start
        if index + 1 < len(marker_starts):
            end = min(end, next_start)
        elif tail_start > start:
            end = min(end, tail_start)
        end = min(end, len(tokens))
        if end <= start:
            continue
        for position in range(start, end):
            mask[position] = True
    return mask

EVENT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "TrafficAccident": {
        "description": "Traffic accidents such as crashes, rollovers, or road incidents.",
        "roles": ["time", "location", "casualty", "vehicle", "cause", "rescue_org"],
        "core_roles": ["location"],
        "required_roles": ["location"],
        "single_value_roles": ["time", "location", "cause"],
    },
    "NaturalDisaster": {
        "description": "Natural disasters such as earthquakes, floods, landslides, and severe weather.",
        "roles": ["time", "location", "casualty", "affected_area", "cause", "rescue_org"],
        "core_roles": ["location"],
        "required_roles": ["location"],
        "single_value_roles": ["time", "location", "cause"],
    },
    "FireExplosion": {
        "description": "Fire and explosion events.",
        "roles": ["time", "location", "casualty", "facility", "cause", "rescue_org"],
        "core_roles": ["location"],
        "required_roles": ["location"],
        "single_value_roles": ["time", "location", "facility", "cause"],
    },
    "PublicHealth": {
        "description": "Public health emergencies such as poisoning, outbreaks, and disease clusters.",
        "roles": ["time", "location", "casualty", "facility", "cause", "response_org"],
        "core_roles": ["location"],
        "required_roles": ["location"],
        "single_value_roles": ["time", "location", "facility", "cause"],
    },
    "Rescue": {
        "description": "Emergency rescue, evacuation, transfer, and response operations.",
        "roles": ["time", "location", "rescued_people", "rescue_org", "method", "related_event"],
        "core_roles": ["rescue_org"],
        "required_roles": ["rescue_org"],
        "single_value_roles": ["time", "location", "method", "related_event"],
    },
}

EVENT_TYPES = list(EVENT_SCHEMA.keys())
ROLE_TYPES = sorted({role for spec in EVENT_SCHEMA.values() for role in spec["roles"]})

EVENT_TRIGGER_HINTS: Dict[str, List[str]] = {
    "TrafficAccident": ["accident", "crash", "collision", "rollover"],
    "NaturalDisaster": ["earthquake", "flood", "landslide", "storm", "warning"],
    "FireExplosion": ["fire", "explosion", "blast", "burn"],
    "PublicHealth": ["infection", "outbreak", "poisoning", "epidemic", "cluster"],
    "Rescue": ["rescue", "evacuation", "transfer", "treatment", "response"],
}

TIME_RE = re.compile(r"\d|year|month|day|hour|minute|today|yesterday|morning|afternoon|night", re.IGNORECASE)
LOCATION_RE = re.compile(r".{2,}")
CASUALTY_RE = re.compile(r"\d|dead|death|injured|missing|poisoned|casualty", re.IGNORECASE)
VEHICLE_RE = re.compile(r"vehicle|car|truck|bus|ship|train|bike", re.IGNORECASE)
ORG_RE = re.compile(r"department|bureau|team|hospital|police|fire|rescue|center|authority", re.IGNORECASE)
FACILITY_RE = re.compile(r"facility|factory|warehouse|school|hospital|center|building", re.IGNORECASE)
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def require_data_name(data_name: Optional[str]) -> str:
    value = str(data_name or "").strip()
    if not value:
        raise ValueError("--data_name is required.")
    return value


def resolve_event_task(data_name: Optional[str], task_type: str = "auto") -> bool:
    mode = str(task_type or "auto").strip().lower()
    if mode in {"emergency", "event", "event_extraction"}:
        return True
    if mode != "auto":
        raise ValueError("task_type must be one of: auto or emergency.")
    return "emergency" in require_data_name(data_name).lower()


def limit_dataset_for_debug(dataset: Any, enabled: bool, limit: int = 32) -> Any:
    """Return a small in-memory view without modifying the source dataset."""
    if not enabled:
        return dataset
    limit = max(1, int(limit))
    size = min(limit, len(dataset))
    select = getattr(dataset, "select", None)
    if callable(select):
        return select(range(size))
    return dataset[:size]


def filter_positive_event_examples(dataset: Any) -> Tuple[Any, int]:
    """Remove zero-event rows without mutating the source dataset."""
    keep_indices = []
    for index in range(len(dataset)):
        example = dataset[index]
        events = example.get("events", []) if isinstance(example, dict) else []
        if isinstance(events, list) and any(isinstance(event, dict) for event in events):
            keep_indices.append(index)
    skipped = len(dataset) - len(keep_indices)
    if skipped == 0:
        return dataset, 0
    select = getattr(dataset, "select", None)
    if callable(select):
        return select(keep_indices), skipped
    return [dataset[index] for index in keep_indices], skipped


def validate_event_configuration(
    event_type_count: int,
    role_type_count: int,
    max_event_count: int,
    max_sentence_markers: int,
) -> None:
    if int(max_event_count) < 1:
        raise ValueError("max_event_count must be at least 1.")
    if int(event_type_count) != len(EVENT_TYPES):
        raise ValueError(
            "event_type_count must match the active event schema: "
            f"expected {len(EVENT_TYPES)}, got {event_type_count}."
        )
    if int(role_type_count) != len(ROLE_TYPES):
        raise ValueError(
            "role_type_count must match the active event schema: "
            f"expected {len(ROLE_TYPES)}, got {role_type_count}."
        )
    if int(max_sentence_markers) < 1:
        raise ValueError("max_sentence_markers must be at least 1.")


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_label_json(data: Any) -> str:
    """Serialize decoder labels without reordering event fields.

    ``compact_json`` intentionally keeps sorted keys for stable sort keys and
    comparisons.  Training labels are different: field order is part of the
    decoder trajectory and must stay aligned with the latent-to-JSON workflow.
    """
    return json.dumps(data, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def resolve_emergency_data_path(data_name: str, split: str = "train") -> Path:
    candidate = Path(data_name)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        split_path = candidate / f"{split}.jsonl"
        if split_path.is_file():
            return split_path
        raise FileNotFoundError(
            f"Cannot find split {split!r} in emergency dataset directory {str(candidate)!r}."
        )

    base = project_root() / "data" / data_name
    split_path = base / f"{split}.jsonl"
    if split_path.exists():
        return split_path

    train_path = base / "train.jsonl"
    if split == "train" and train_path.exists():
        return train_path

    raise FileNotFoundError(f"Cannot find emergency dataset for {data_name!r} and split {split!r}.")


def load_emergency_examples(data_name: str, split: str = "train") -> List[Dict[str, Any]]:
    return load_jsonl(resolve_emergency_data_path(data_name, split))


def format_schema(schema: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    schema = schema or EVENT_SCHEMA
    lines = ["事件类型与角色约束:"]
    for event_type, spec in schema.items():
        roles = list(spec["roles"])
        core_roles = list(spec.get("core_roles") or spec.get("required_roles") or [])
        optional_roles = [role for role in roles if role not in set(core_roles)]
        lines.append(
            f"- {event_type}: {spec['description']}; "
            f"allowed_roles=[{', '.join(roles)}]; "
            f"core_roles=[{', '.join(core_roles)}]; "
            f"optional_roles=[{', '.join(optional_roles)}]"
        )
    return "\n".join(lines)


def format_document(example: Dict[str, Any]) -> str:
    lines = [f"文档ID: {example.get('doc_id', 'unknown')}", "文档句子:"]
    for idx, sentence in enumerate(example["sentences"]):
        lines.append(f"S{idx}: {str(sentence).replace(chr(0xfeff), '').strip()}")
    return "\n".join(lines)


def format_prompt_for_model(prompt: str, tokenizer: Any, use_chat_template: bool = False) -> str:
    """Wrap event prompts in the backbone's native assistant envelope when requested."""
    if not use_chat_template:
        return prompt
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template) or not getattr(tokenizer, "chat_template", None):
        raise ValueError("use_chat_template=True requires a tokenizer with a configured chat template.")
    return apply_template(
        [{"role": "user", "content": str(prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_event_type_rules() -> str:
    lines = ["事件类型与实例区分规则:"]
    for event_type, hints in EVENT_TRIGGER_HINTS.items():
        lines.append(f"- {event_type}: 触发词或语境提示={', '.join(hints)}")
    lines.append(
        "实例规则: 对同类型或相同触发词事件，应依据证据句、时间、地点、参与对象和角色论元区分事件实例。"
    )
    lines.append(
        "角色规则: 只有当文档提供明确支持时才填充角色，不能为了补齐类型-角色空间而编造论元。"
    )
    lines.append(
        "跨事件规则: 论元必须由当前事件的证据支持，不能从其他事件实例中串扰复制。"
    )
    return "\n".join(lines)


def canonical_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalised = []
    for event in events:
        if not isinstance(event, dict):
            continue
        arguments = {}
        raw_arguments = event.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raw_arguments = {}
        for role, value in raw_arguments.items():
            if value is None:
                continue
            if isinstance(value, list):
                values = [str(v) for v in value if str(v)]
            else:
                values = [str(value)] if str(value) else []
            if values:
                arguments[role] = values

        evidence = event.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [evidence]

        normalised.append(
            {
                "event_type": str(event.get("event_type", "")),
                "trigger": str(event.get("trigger", "")),
                "evidence": [str(e) for e in evidence],
                "arguments": arguments,
            }
        )

    def first_evidence_index(item: Dict[str, Any]) -> int:
        indices = []
        for evidence_id in item.get("evidence", []):
            evidence_text = str(evidence_id)
            if evidence_text.startswith("S") and evidence_text[1:].isdigit():
                indices.append(int(evidence_text[1:]))
        return min(indices) if indices else 10**9

    return sorted(
        normalised,
        key=lambda item: (
            first_evidence_index(item),
            item["event_type"],
            item["trigger"],
            compact_json(item["arguments"]),
            ",".join(item["evidence"]),
        ),
    )


def deduplicate_exact_events(
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the first occurrence of each exactly equal canonical event."""

    unique_events = []
    seen_fingerprints = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        fingerprint = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        unique_events.append(event)
    return unique_events


def _valid_evidence_indices(
    event: Dict[str, Any],
    sentence_count: int,
    max_sentence_markers: int,
) -> List[int]:
    upper_bound = min(max(0, int(sentence_count)), max(1, int(max_sentence_markers)))
    indices = []
    for evidence_id in event.get("evidence", []):
        match = re.fullmatch(r"S(\d+)", str(evidence_id).strip())
        if match is None:
            continue
        index = int(match.group(1))
        if 0 <= index < upper_bound:
            indices.append(index)
    return sorted(set(indices))


def resolve_event_identity_anchor(
    event: Dict[str, Any],
    sentences: Sequence[Any],
    max_sentence_markers: int = 32,
) -> int:
    """Resolve one evidence-backed identity anchor shared by all model paths.

    Preference is given to trigger support.  Without trigger support, the
    evidence sentence supporting the largest number of annotated argument
    values is used.  Remaining ties fall back to the earliest valid evidence.
    """
    if not isinstance(event, dict):
        return -100
    sentence_texts = [str(sentence) for sentence in sentences]
    evidence_indices = _valid_evidence_indices(
        event,
        sentence_count=len(sentence_texts),
        max_sentence_markers=max_sentence_markers,
    )
    if not evidence_indices:
        return -100

    trigger = str(event.get("trigger", "")).strip()
    if trigger:
        for index in evidence_indices:
            if trigger in sentence_texts[index]:
                return index

    argument_values = []
    arguments = event.get("arguments", {})
    if isinstance(arguments, dict):
        for raw_values in arguments.values():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            argument_values.extend(
                str(value).strip() for value in values if str(value).strip()
            )
    if argument_values:
        support_counts = {
            index: sum(value in sentence_texts[index] for value in argument_values)
            for index in evidence_indices
        }
        best_count = max(support_counts.values())
        if best_count > 0:
            return min(index for index, count in support_counts.items() if count == best_count)
    return evidence_indices[0]


def _nonempty_scalar(value: Any) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def _nonempty_sequence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(_nonempty_scalar(item) for item in value)
    return _nonempty_scalar(value)


def has_nonempty_argument_value(arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    for values in arguments.values():
        if not isinstance(values, list):
            values = [values]
        if any(_nonempty_scalar(value) for value in values):
            return True
    return False


def event_content_flags(event: Any) -> Dict[str, bool]:
    if not isinstance(event, dict):
        return {
            "event_type": False,
            "trigger": False,
            "evidence": False,
            "arguments": False,
        }
    return {
        "event_type": _nonempty_scalar(event.get("event_type", "")),
        "trigger": _nonempty_scalar(event.get("trigger", "")),
        "evidence": _nonempty_sequence(event.get("evidence", [])),
        "arguments": has_nonempty_argument_value(event.get("arguments", {})),
    }


def is_empty_event(event: Any) -> bool:
    flags = event_content_flags(event)
    return not any(flags.values())


def is_semantically_valid_event(event: Any) -> bool:
    flags = event_content_flags(event)
    content_kinds = int(flags["trigger"]) + int(flags["evidence"]) + int(flags["arguments"])
    return bool(flags["event_type"] and content_kinds >= 2)


def has_semantically_valid_event_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return False
    events = payload.get("events", [])
    return any(is_semantically_valid_event(event) for event in events)


def has_semantically_complete_event_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return False
    events = payload.get("events", [])
    if not events:
        return False
    return all(is_semantically_valid_event(event) for event in events)


def has_empty_event_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return False
    return any(is_empty_event(event) for event in payload.get("events", []))


def is_extractable_event_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return False
    events = payload.get("events", [])
    if not events:
        return False
    if any(is_empty_event(event) for event in events):
        return False
    return any(any(event_content_flags(event).values()) for event in events)


def event_payload_quality_metrics(
    payload: Any,
    sentences: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Measure the generated payload without repairing or rewriting its content."""

    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return {"schema": 0.0, "citation_validity": 0.0, "claim_support": 0.0}
    events = payload.get("events", [])
    if not events:
        return {"schema": 0.0, "citation_validity": 0.0, "claim_support": 0.0}

    sentence_texts = [re.sub(r"\s+", "", str(sentence)) for sentence in (sentences or [])]
    canonical_event_keys = {"event_type", "trigger", "evidence", "arguments"}
    schema_valid_events = 0
    evidence_score_sum = 0.0
    support_hits = 0
    support_total = 0

    def nonempty_values(value: Any) -> List[str]:
        raw_values = value if isinstance(value, list) else [value]
        return [
            str(item).strip()
            for item in raw_values
            if item is not None and str(item).strip()
        ]

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        trigger = event.get("trigger")
        evidence = event.get("evidence")
        arguments = event.get("arguments")
        schema_spec = EVENT_SCHEMA.get(event_type) if isinstance(event_type, str) else None

        evidence_shape_valid = (
            isinstance(evidence, list)
            and bool(evidence)
            and all(
                isinstance(evidence_id, str)
                and re.fullmatch(r"S\d+", evidence_id.strip()) is not None
                for evidence_id in evidence
            )
        )
        arguments_shape_valid = isinstance(arguments, dict)
        if arguments_shape_valid and schema_spec is not None:
            allowed_roles = set(schema_spec.get("roles", []))
            arguments_shape_valid = all(
                isinstance(role, str)
                and role in allowed_roles
                and isinstance(values, list)
                and bool(nonempty_values(values))
                and len(nonempty_values(values)) == len(values)
                for role, values in arguments.items()
            )
        schema_valid = (
            set(event.keys()) == canonical_event_keys
            and schema_spec is not None
            and isinstance(trigger, str)
            and bool(trigger.strip())
            and evidence_shape_valid
            and arguments_shape_valid
        )
        schema_valid_events += int(schema_valid)

        raw_evidence = evidence if isinstance(evidence, list) else []
        valid_evidence_indices = []
        valid_evidence_count = 0
        for evidence_id in raw_evidence:
            match = re.fullmatch(r"S(\d+)", str(evidence_id).strip())
            if match is None:
                continue
            sentence_index = int(match.group(1))
            if 0 <= sentence_index < len(sentence_texts):
                valid_evidence_count += 1
                valid_evidence_indices.append(sentence_index)
        evidence_score_sum += (
            valid_evidence_count / len(raw_evidence) if raw_evidence else 0.0
        )

        claims = nonempty_values(trigger)
        if isinstance(arguments, dict):
            for values in arguments.values():
                claims.extend(nonempty_values(values))
        cited_texts = [sentence_texts[index] for index in valid_evidence_indices]
        for claim in claims:
            normalized_claim = re.sub(r"\s+", "", claim)
            support_total += 1
            if normalized_claim and any(
                normalized_claim in sentence_text for sentence_text in cited_texts
            ):
                support_hits += 1

    event_count = len(events)
    return {
        "schema": schema_valid_events / event_count,
        "citation_validity": evidence_score_sum / event_count,
        "claim_support": support_hits / support_total if support_total else 0.0,
    }


def build_event_prompt(example: Dict[str, Any]) -> str:
    source_report_count = sum(
        1
        for sentence in example.get("sentences", [])
        if str(sentence).strip().startswith("[Source report ")
    )
    source_hint = ""
    if source_report_count:
        source_hint = (
            f"本文档包含 {source_report_count} 个来源片段。"
            "每个 [Source report N] 片段都可能包含独立事件。"
            "请扫描所有片段，而不是只处理第一个片段。"
            "在可能的情况下，按照最早支持证据句的顺序排列事件。\n"
        )
    return (
        "你是一个文档级突发事件抽取系统。请抽取文档中所有有证据支持的事件实例，"
        "并输出严格 JSON。JSON 顶层字段为 events，每个事件包含 event_type、trigger、"
        "evidence 和 arguments。\n\n"
        f"{format_schema()}\n\n"
        f"{format_event_type_rules()}\n\n"
        f"{source_hint}"
        f"{format_document(example)}\n\n"
        "请在内部完成证据选择、事件类型判断、事件实例区分和角色填充。"
        "内部决策请遵循五条原则：support，事件和论元必须由原文证据支持；"
        "separate，同类型或相似触发事件必须按证据、时间、地点、对象和论元区分实例；"
        "bind，每个论元只绑定到支持它的事件；"
        "order，事件按最早核心证据句顺序排列；"
        "no-borrow，不能从其他事件借用未被当前证据支持的论元。"
        "对于多事件文档，必须覆盖全文中的所有事件，尤其注意同类型事件和跨片段事件。"
        "输出时只输出一个严格 JSON 对象；不要输出任何标记、额外解释或推理过程。"
    )


def build_teacher_process(
    example: Dict[str, Any],
    max_sentence_markers: int = 32,
) -> str:
    events = canonical_events(example["events"])
    sentences = {
        f"S{index}": str(sentence)
        for index, sentence in enumerate(example.get("sentences", []))
    }

    def evidence_ids(event: Dict[str, Any]) -> List[str]:
        return [str(evidence_id) for evidence_id in event.get("evidence", [])]

    def evidence_support_order(event_evidence: Sequence[str]) -> List[str]:
        def sort_key(evidence_id: str) -> Tuple[int, str]:
            match = re.fullmatch(r"S(\d+)", str(evidence_id).strip())
            return (int(match.group(1)), "") if match else (10**9, str(evidence_id))

        return sorted((str(evidence_id) for evidence_id in event_evidence), key=sort_key)

    def sentence_for(evidence_id: str) -> str:
        return sentences.get(str(evidence_id), "")

    def locate(value: str, event_evidence: Sequence[str]) -> str:
        text_value = str(value).strip()
        for evidence_id in evidence_support_order(event_evidence):
            if text_value and text_value in sentence_for(str(evidence_id)):
                return f"{text_value}@{evidence_id}"
        return f"{text_value}@unaligned" if text_value else "NA"

    def support_sentence(value: str, event_evidence: Sequence[str]) -> str:
        text_value = str(value).strip()
        for evidence_id in evidence_support_order(event_evidence):
            if text_value and text_value in sentence_for(str(evidence_id)):
                return str(evidence_id)
        return "unaligned" if text_value else "NA"

    def role_values(event: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        values = []
        for role, raw_values in event.get("arguments", {}).items():
            for value in _normalise_values(raw_values):
                values.append((str(role), value, support_sentence(value, evidence_ids(event))))
        return values

    def identity_anchor(event: Dict[str, Any]) -> str:
        anchor_index = resolve_event_identity_anchor(
            event,
            example.get("sentences", []),
            max_sentence_markers=max_sentence_markers,
        )
        return f"S{anchor_index}" if anchor_index >= 0 else "NA"

    def event_differences(left_event: Dict[str, Any], right_event: Dict[str, Any]) -> List[str]:
        differences = []
        if identity_anchor(left_event) != identity_anchor(right_event):
            differences.append("identity_anchor")
        if evidence_ids(left_event) != evidence_ids(right_event):
            differences.append("evidence")
        if str(left_event.get("trigger", "")) != str(right_event.get("trigger", "")):
            differences.append("trigger")
        for role in ("time", "location", "casualty", "affected_area"):
            left_values = _normalise_values(left_event.get("arguments", {}).get(role))
            right_values = _normalise_values(right_event.get("arguments", {}).get(role))
            if left_values != right_values:
                differences.append(role)
        return differences

    def summarize_argument_support(event: Dict[str, Any], limit: int = 6) -> str:
        supports = role_values(event)
        if not supports:
            return "no explicit argument span is annotated; keep optional roles unsupported."
        clauses = [
            f"bind {role} {value} to this event because it is supported by {sentence_id}"
            for role, value, sentence_id in supports[:limit]
        ]
        if len(supports) > limit:
            clauses.append(f"{len(supports) - limit} additional arguments follow the same evidence-bound binding rule")
        return "; ".join(clauses) + "."

    source_report_count = sum(
        1
        for sentence in example.get("sentences", [])
        if str(sentence).strip().startswith("[Source report ")
    )
    evidence_lines = [
        PROCESS_MARKERS[0],
        f"process goal: make evidence-support decisions before generating fields; gold_event_count={len(events)}; source_report_count={source_report_count}.",
    ]
    type_lines = [
        PROCESS_MARKERS[1],
        "type decisions assign event categories from trigger and supporting evidence, not from output-field templates.",
    ]
    instance_lines = [
        PROCESS_MARKERS[2],
        "instance decisions keep same-type events separate when their anchor evidence or supported arguments differ.",
    ]
    plan_lines = [
        PROCESS_MARKERS[3],
        "organization decisions preserve canonical earliest-evidence order; identity anchors distinguish instances without redefining output order.",
    ]
    role_lines = [
        PROCESS_MARKERS[4],
        "argument decisions bind each value to the event whose evidence supports it; no-borrow prevents copying arguments across events.",
    ]

    anchors = [identity_anchor(event) for event in events]
    for idx, event in enumerate(events, start=1):
        ev_ids = evidence_ids(event)
        evidence = ",".join(ev_ids) if ev_ids else "NA"
        trigger = str(event.get("trigger", "")).strip() or "NA"
        trigger_sentence = support_sentence(trigger, ev_ids)
        anchor = anchors[idx - 1]
        evidence_lines.append(
            f"event{idx} evidence decision: use evidence {evidence}; identity anchor is {anchor}; "
            f"trigger {trigger} is supported by {trigger_sentence}; arguments must be checked inside this evidence set."
        )
        type_lines.append(
            f"event{idx} type decision: assign {event.get('event_type', 'NA')} because trigger {trigger} "
            f"and anchor evidence {anchor} support this event type."
        )

        same_type_notes = []
        for peer_index, peer_event in enumerate(events, start=1):
            if peer_index == idx or peer_event.get("event_type") != event.get("event_type"):
                continue
            differences = event_differences(event, peer_event)
            if differences:
                same_type_notes.append(
                    f"separate from event{peer_index} because {','.join(differences[:4])} differ"
                )
            else:
                same_type_notes.append(
                    f"keep separate from event{peer_index} by the gold instance boundary even though surface cues are close"
                )
        if same_type_notes:
            instance_reason = "; ".join(same_type_notes) + "."
        else:
            instance_reason = "no same-type competing event shares this identity anchor."
        instance_lines.append(
            f"event{idx} instance decision: keep one independent event around anchor {anchor}; {instance_reason}"
        )
        plan_lines.append(
            f"event{idx} organization decision: keep canonical earliest-evidence order while using identity anchor {anchor} for separation; "
            f"keep evidence {evidence} attached to this event."
        )
        role_lines.append(
            f"event{idx} argument decision: {summarize_argument_support(event)} no-borrow: do not copy unsupported time, location, casualty, or organization arguments from another event."
        )

    sections = (
        evidence_lines + [PROCESS_DONE_MARKERS[0]]
        + type_lines + [PROCESS_DONE_MARKERS[1]]
        + instance_lines + [PROCESS_DONE_MARKERS[2]]
        + plan_lines + [PROCESS_DONE_MARKERS[3]]
        + role_lines + [PROCESS_DONE_MARKERS[4]]
    )
    return "\n".join(sections) + "\n"

def build_final_answer(example: Dict[str, Any]) -> str:
    payload = {"events": canonical_events(example["events"])}
    body = compact_label_json(payload)
    if payload["events"] and not body.startswith(JSON_EVENT_ARRAY_PREFIX + '{"event_type"'):
        raise ValueError(
            "Final-answer label order is inconsistent with JSON start training: "
            "expected first event to begin with event_type."
        )
    return f"{FINAL_JSON_MARKER}\n{body}"


def derive_latent_budget(example: Dict[str, Any], max_steps: int) -> int:
    """Derive the shared structural target for adaptive latent reasoning."""
    events = [event for event in example.get("events", []) if isinstance(event, dict)]
    event_types = [
        str(event.get("event_type", "")).strip()
        for event in events
        if str(event.get("event_type", "")).strip()
    ]
    triggers = [
        str(event.get("trigger", "")).strip()
        for event in events
        if str(event.get("trigger", "")).strip()
    ]
    evidence_ids = {
        str(evidence_id)
        for event in events
        for evidence_id in event.get("evidence", [])
    }
    event_count = max(1, len(events))
    event_component = min(3, (event_count - 1).bit_length())
    context_component = int(
        len(example.get("sentences", [])) >= 10 or len(evidence_ids) >= 8
    )
    instance_component = int(
        len(event_types) != len(set(event_types))
        or len(triggers) != len(set(triggers))
    )
    type_component = int(len(set(event_types)) > 1)
    steps = (
        MIN_LATENT_BUDGET
        + event_component
        + context_component
        + instance_component
        + type_component
    )
    upper_bound = max(1, int(max_steps))
    lower_bound = min(MIN_LATENT_BUDGET, upper_bound)
    return max(lower_bound, min(upper_bound, steps))


def example_complexity(example: Dict[str, Any], max_steps: int) -> int:
    """Backward-compatible name for the shared latent-budget target."""
    return derive_latent_budget(example, max_steps)


def event_decoder_context_length(
    source_length: int,
    answer_length: int,
    num_latent: int,
    max_event_count: int,
) -> int:
    """Worst-case decoder length: source + handles + latent + bridge + answer."""
    return (
        int(source_length)
        + int(answer_length)
        + int(num_latent)
        + 2 * int(max_event_count)
    )


def event_generation_prefix_reserve(
    num_latent: int,
    max_event_count: int,
    fixed_suffix_tokens: int = 2,
) -> int:
    """Reserve handles, latent states, bridge tokens, and fixed decoder suffix."""
    return int(num_latent) + 2 * int(max_event_count) + int(fixed_suffix_tokens)


def strip_bos(ids: Sequence[int], bos_token_id: Optional[int]) -> List[int]:
    ids = list(ids)
    if ids and bos_token_id is not None and ids[0] == bos_token_id:
        return ids[1:]
    return ids


def find_subsequence_position(tokens: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    token_list = list(tokens)
    pattern_list = list(pattern)
    for start in range(0, len(token_list) - len(pattern_list) + 1):
        if token_list[start : start + len(pattern_list)] == pattern_list:
            return start + len(pattern_list)
    return -1


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if FINAL_JSON_MARKER in text:
        text = text.split(FINAL_JSON_MARKER, 1)[1]
    payloads = candidate_json_substrings(text)
    for payload in payloads:
        for candidate in (payload, repair_decoded_json(payload)):
            try:
                parsed = normalize_json_strings(json.loads(candidate))
                parsed = unwrap_event_payload(parsed)
                if is_extractable_event_payload(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def candidate_json_substrings(text: str) -> List[str]:
    candidates = []
    stripped = text.strip()

    def add_candidate(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for match in re.finditer(r'(?:"?events"?\s*:\s*\[)', stripped):
        fragment = stripped[match.start() :]
        if fragment.startswith('"events"'):
            add_candidate(balance_json_text("{" + fragment))
        elif fragment.startswith("events"):
            add_candidate(balance_json_text('{"' + fragment))

    event_object_match = re.search(r'\{\s*"(?:arguments|event_type|trigger|evidence)"', stripped)
    if event_object_match:
        fragment = stripped[event_object_match.start() :]
        add_candidate(balance_json_text('{"events":[' + fragment + "]}"))

    for match in re.finditer(r"\{", text):
        start = match.start()
        end = find_first_complete_json_end(text, start)
        if end >= start:
            add_candidate(text[start : end + 1])

    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            add_candidate(text[start : end + 1])

    candidates.sort(key=lambda item: (0 if '"events"' in item or '" events "' in item else 1, -len(item)))
    return candidates


def unwrap_event_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("events"), dict):
        nested = payload["events"]
        if isinstance(nested.get("events"), list):
            return nested
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        events = payload["events"]
        if len(events) == 1 and isinstance(events[0], dict) and isinstance(events[0].get("events"), list):
            return {"events": events[0]["events"]}
    if isinstance(payload, dict) and {"event_type", "trigger"}.issubset(payload.keys()):
        return {"events": [payload]}
    return payload


def find_first_complete_json_end(text: str, start: int = 0) -> int:
    if start < 0:
        return -1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def repair_decoded_json(text: str) -> str:
    repaired = re.sub(r'"\s*([A-Za-z_][A-Za-z0-9_]*)\s*"\s*:', r'"\1":', text)
    repaired = re.sub(r':\s*"\s*([A-Za-z_][A-Za-z0-9_]*)\s*"', r':"\1"', repaired)
    repaired = re.sub(r'\[\s*"\s*(S\d+)\s*"', r'["\1"', repaired)
    repaired = re.sub(r',\s*"\s*(S\d+)\s*"', r',"\1"', repaired)
    return balance_json_text(repaired)


def balance_json_text(text: str) -> str:
    stack = []
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
    suffix = ""
    if in_string:
        # A max-token stop often lands inside the final value. Close that value
        # before closing its containers so earlier complete events remain usable.
        if escape:
            suffix += "\\"
        suffix += '"'
    return text + suffix + "".join(reversed(stack))


def normalize_json_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).strip(): normalize_json_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_strings(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def event_tuples(payload: Optional[Dict[str, Any]]) -> Set[Tuple[str, str, str, str]]:
    if not payload or not isinstance(payload.get("events"), list):
        return set()
    tuples: Set[Tuple[str, str, str, str]] = set()
    for event in payload["events"]:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", ""))
        trigger = str(event.get("trigger", ""))
        arguments = event.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        for role, values in arguments.items():
            if not isinstance(values, list):
                values = [values]
            for value in values:
                tuples.add((event_type, trigger, str(role), str(value)))
    return tuples


def payload_event_count(payload: Optional[Dict[str, Any]]) -> int:
    if not payload or not isinstance(payload.get("events"), list):
        return 0
    return sum(1 for event in payload["events"] if isinstance(event, dict))


def payload_argument_count(payload: Optional[Dict[str, Any]]) -> int:
    if not payload or not isinstance(payload.get("events"), list):
        return 0
    total = 0
    for event in payload["events"]:
        if not isinstance(event, dict):
            continue
        arguments = event.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        for values in arguments.values():
            total += len(_normalise_values(values))
    return total


def tuple_f1(gold_payload: Dict[str, Any], pred_payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    gold = event_tuples(gold_payload)
    pred = event_tuples(pred_payload)
    if not gold and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(gold & pred)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_metric_text(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip())
    return text.strip(
        " \t\r\n,;:!?\"'[](){}<>"
        "\u3000\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f"
        "\uff08\uff09\u3010\u3011\u201c\u201d\u2018\u2019"
    )


_CONNECTOR_RE = re.compile(r"(?:\u3001|,|\uff0c|;|\uff1b|/|\\\\|\u53ca|\u4e0e|\u548c)")
_ADMIN_PREFIX_RE = re.compile(
    r"^([\u4e00-\u9fff]{1,10}(?:\u7701|\u5e02|\u533a|\u53bf|\u9547|\u4e61|\u8857\u9053))(.{2,})$"
)
_LOCATION_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:"
    r"\u9ad8\u901f\u516c\u8def|\u9ad8\u901f|\u8def\u6bb5|\u5927\u6865|\u6865|"
    r"\u96a7\u9053|\u6279\u53d1\u5e02\u573a|\u5e02\u573a|\u5b66\u6821|\u4e2d\u5b66|\u5c0f\u5b66|"
    r"\u7164\u77ff|\u77ff\u4e95|\u77ff\u533a|\u5c0f\u533a|\u4ed3\u5e93|\u5de5\u5382|\u56ed\u533a|"
    r"\u516c\u56ed|\u666f\u533a|\u6cb3\u6bb5|\u7801\u5934|\u8f66\u7ad9|\u5ba2\u8fd0\u7ad9|"
    r"\u6751|\u9547|\u53bf|\u533a|\u5e02|\u8def|\u8857|\u697c|\u5e97|\u9662|\u5382|\u7ad9)"
)
_ORG_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:"
    r"\u5c40|\u961f|\u90e8\u95e8|\u4e2d\u5fc3|\u533b\u9662|\u6d3e\u51fa\u6240|"
    r"\u6d88\u9632|\u6551\u63f4\u7ad9|\u8857\u9053\u529e|\u59d4\u5458\u4f1a)"
)
_VEHICLE_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{1,}(?:"
    r"\u7f51\u7ea6\u8f66|\u7535\u52a8\u8f66|\u6469\u6258\u8f66|\u9762\u5305\u8f66|"
    r"\u53a2\u5f0f\u8d27\u8f66|\u5927\u5ba2\u8f66|\u516c\u4ea4\u8f66|\u8d27\u8f66|\u5ba2\u8f66|"
    r"\u8f7f\u8f66|\u6821\u8f66|\u6e21\u8239|\u4f5c\u4e1a\u8239|\u8239|\u8247|\u8f66)"
)
_QUANTITY_RE = re.compile(
    r"(?:\u81f3\u5c11|\u7ea6|\u8fd1|\u8d85|\u8d85\u8fc7|\u903e|\u5171|\u5df2\u6709)?"
    r"(?:\d+(?:\.\d+)?|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u4ebf\u4e24\u51e0\u6570]+)"
    r"(?:\u4f59|\u591a)?(?:\u4eba|\u540d|\u4f4d|\u6237|\u5bb6|\u8f86|\u8258|\u5904|\u4e2a|\u6240|\u4f8b|\u5b97|\u8d77|\u5428|\u95f4|\u680b)"
)
_TIME_RE = re.compile(
    r"(?:\d{2,4}\u5e74)?\d{1,2}\u6708\d{1,2}\u65e5(?:\d{1,2}\u65f6(?:\d{1,2}\u5206)?)?"
    r"|\d{1,2}\u65e5\d{1,2}\u65f6(?:\d{1,2}\u5206)?"
    r"|\d{1,2}\u65f6(?:\d{1,2}\u5206)?"
    r"|\d{1,2}:\d{2}"
    r"|\u51cc\u6668|\u6e05\u6668|\u65e9\u4e0a|\u4e0a\u5348|\u4e2d\u5348|\u4e0b\u5348|\u665a\u4e0a|\u6628\u665a"
    r"|\u4eca\u5929|\u6628\u5929|\u6628\u65e5|\u6b21\u65e5|\u5f53\u65e5|\u5f53\u5730\u65f6\u95f4"
    r"|\u5468[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u65e5\u5929]"
)


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        value = normalize_metric_text(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _strip_admin_prefix(text: str) -> str:
    head = text
    while True:
        match = _ADMIN_PREFIX_RE.match(head)
        if not match:
            return head
        candidate = match.group(2)
        if len(candidate) < 2:
            return head
        head = candidate


def _regex_heads(pattern: re.Pattern, text: str) -> List[str]:
    return _dedupe_keep_order(match.group(0) for match in pattern.finditer(text))


def argument_heads(value: Any, role: str = "") -> List[str]:
    text = normalize_metric_text(value)
    if not text:
        return []
    role = normalize_metric_text(role)
    segments = [segment for segment in _CONNECTOR_RE.split(text) if segment]

    if role == "time":
        matches = _regex_heads(_TIME_RE, text)
        return matches or [text[-12:] if len(text) > 12 else text]

    if role in {"casualty", "rescued_people", "affected_area"}:
        matches = _regex_heads(_QUANTITY_RE, text)
        if matches:
            return _dedupe_keep_order(
                re.sub(
                    r"^(?:\u81f3\u5c11|\u7ea6|\u8fd1|\u8d85|\u8d85\u8fc7|\u903e|\u5171|\u5df2\u6709)",
                    "",
                    match,
                )
                for match in matches
            )

    if role == "vehicle":
        matches = []
        for segment in segments:
            matches.extend(_regex_heads(_VEHICLE_SUFFIX_RE, segment))
        if matches:
            return _dedupe_keep_order(matches)

    if role in {"location", "facility"}:
        heads = []
        for segment in segments:
            segment = re.sub(r"(\u9644\u8fd1|\u5468\u8fb9|\u4e00\u5e26|\u5883\u5185|\u5904)$", "", segment)
            matches = _regex_heads(_LOCATION_SUFFIX_RE, segment)
            head = matches[-1] if matches else segment
            if role == "location":
                head = _strip_admin_prefix(head)
            heads.append(head)
        return _dedupe_keep_order(heads or [text])

    if role in {"rescue_org", "response_org"}:
        heads = []
        for segment in segments:
            matches = _regex_heads(_ORG_SUFFIX_RE, segment)
            heads.append(matches[-1] if matches else segment)
        return _dedupe_keep_order(heads or [text])

    if role in {"trigger", "related_event"}:
        return [text]

    tail = segments[-1] if segments else text
    return [tail[-10:] if len(tail) > 10 else tail]


def argument_head(value: Any, role: str = "") -> str:
    heads = argument_heads(value, role)
    return heads[-1] if heads else ""


def _metric_f1_from_counts(hit: int, pred_total: int, gold_total: int) -> Dict[str, float]:
    if gold_total == 0 and pred_total == 0:
        precision = recall = f1 = 1.0
    elif pred_total == 0 or gold_total == 0:
        precision = recall = f1 = 0.0
    else:
        precision = hit / pred_total
        recall = hit / gold_total
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hit": float(hit),
        "pred": float(pred_total),
        "gold": float(gold_total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metric_f1_from_counters(gold: Counter, pred: Counter) -> Dict[str, float]:
    overlap = gold & pred
    return _metric_f1_from_counts(
        int(sum(overlap.values())),
        int(sum(pred.values())),
        int(sum(gold.values())),
    )


def standard_metric_counters(payload: Optional[Dict[str, Any]]) -> Dict[str, Counter]:
    counters = {
        "trigger": Counter(),
        "arg_i": Counter(),
        "arg_c": Counter(),
        "span": Counter(),
        "head": Counter(),
    }
    if not payload or not isinstance(payload.get("events"), list):
        return counters
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        event_type = normalize_metric_text(event.get("event_type", ""))
        trigger = normalize_metric_text(event.get("trigger", ""))
        if event_type and trigger:
            counters["trigger"][(event_type, trigger)] += 1
        arguments = event.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        for role, values in arguments.items():
            if not isinstance(values, list):
                values = [values]
            role_text = normalize_metric_text(role)
            for value in values:
                span = normalize_metric_text(value)
                if not span:
                    continue
                counters["span"][(span,)] += 1
                for head in argument_heads(span, role_text):
                    counters["head"][(head,)] += 1
                if event_type and trigger:
                    counters["arg_i"][(event_type, trigger, span)] += 1
                    if role_text:
                        counters["arg_c"][(event_type, trigger, role_text, span)] += 1
    return counters


def standard_event_metrics(
    gold_payload: Dict[str, Any],
    pred_payload: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    gold_counters = standard_metric_counters(gold_payload)
    pred_counters = standard_metric_counters(pred_payload)
    return {
        key: _metric_f1_from_counters(gold_counters[key], pred_counters[key])
        for key in ("trigger", "arg_i", "arg_c", "span", "head")
    }


RELAXED_METRIC_KEYS = (
    "soft_trigger",
    "relaxed_span",
    "normalized_arg_i",
    "normalized_arg_c",
    "partial_argument",
)

RELAXED_MISSING_GOLD_FLOOR = 0.833

_RELAXED_TRIGGER_CLUSTERS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "trigger:rain",
        (
            "\u964d\u96e8",
            "\u66b4\u96e8",
            "\u5927\u66b4\u96e8",
            "\u7279\u5927\u66b4\u96e8",
            "\u5f3a\u964d\u96e8",
            "\u5f3a\u964d\u6c34",
            "\u6301\u7eed\u964d\u96e8",
            "\u6301\u7eed\u5f3a\u964d\u96e8",
            "\u964d\u6c34",
        ),
    ),
    (
        "trigger:earthquake",
        ("\u5730\u9707", "\u5f3a\u9707", "\u4f59\u9707", "\u9707\u707e", "\u9707\u52a8", "\u5730\u9707\u707e\u5bb3"),
    ),
    (
        "trigger:flood",
        ("\u6d2a\u6c34", "\u6d2a\u707e", "\u6d2a\u6d9d", "\u5185\u6d9d", "\u5c71\u6d2a", "\u6c5b\u60c5", "\u6d2a\u5cf0", "\u6da8\u6c34"),
    ),
    ("trigger:drought", ("\u5e72\u65f1", "\u65f1\u60c5", "\u65f1\u707e", "\u7f3a\u6c34")),
    (
        "trigger:storm",
        ("\u53f0\u98ce", "\u98ce\u66b4", "\u66b4\u98ce\u96e8", "\u98ce\u707e", "\u5f3a\u98ce", "\u98d3\u98ce"),
    ),
    (
        "trigger:landslide",
        ("\u6ed1\u5761", "\u5c71\u4f53\u6ed1\u5761", "\u6ce5\u77f3\u6d41", "\u584c\u65b9", "\u5d29\u584c"),
    ),
    ("trigger:fire", ("\u706b\u707e", "\u8d77\u706b", "\u7740\u706b", "\u5931\u706b", "\u71c3\u70e7", "\u706b\u60c5", "\u5927\u706b")),
    (
        "trigger:explosion",
        (
            "\u7206\u70b8",
            "\u7206\u70b8\u6848",
            "\u7206\u70b8\u88ad\u51fb",
            "\u7206\u71c3",
            "\u70b8\u5f39\u88ad\u51fb",
            "\u7206\u70b8\u4e8b\u4ef6",
            "\u7206\u70b8\u4e8b\u6545",
        ),
    ),
    (
        "trigger:traffic_accident",
        (
            "\u4e8b\u6545",
            "\u4ea4\u901a\u4e8b\u6545",
            "\u8f66\u7978",
            "\u60e8\u5267",
            "\u76f8\u649e",
            "\u78b0\u649e",
            "\u649e\u8f66",
            "\u522e\u649e",
            "\u8ffd\u5c3e",
            "\u4fa7\u7ffb",
            "\u7ffb\u8f66",
            "\u503e\u8986",
            "\u5760\u8f66",
            "\u51b2\u51fa\u9053\u8def",
            "\u5760\u6bc1",
            "\u5760\u843d",
            "\u5760\u5165",
            "\u6389\u5165",
            "\u4ea4\u901a\u60e8\u5267",
            "\u81f4\u547d\u4e8b\u6545",
        ),
    ),
    (
        "trigger:public_health",
        (
            "\u75ab\u60c5",
            "\u4f20\u67d3\u75c5",
            "\u611f\u67d3",
            "\u786e\u8bca",
            "\u75c5\u4f8b",
            "\u53d1\u75c5",
            "\u6d41\u884c\u75c5",
            "\u66b4\u53d1",
            "\u7206\u53d1",
            "\u75ab\u60c5\u66b4\u53d1",
        ),
    ),
    ("trigger:poisoning", ("\u4e2d\u6bd2", "\u98df\u7269\u4e2d\u6bd2", "\u6c14\u4f53\u4e2d\u6bd2", "\u6bd2\u6c14\u6cc4\u6f0f")),
    (
        "trigger:rescue",
        ("\u6551\u63f4", "\u8425\u6551", "\u62a2\u6551", "\u641c\u6551", "\u6551\u52a9", "\u8f6c\u79fb", "\u758f\u6563", "\u64a4\u79bb", "\u5b89\u7f6e"),
    ),
)

_PAREN_CONTENT_RE = re.compile(r"(?:\([^)]*\)|\uff08[^\uff09]*\uff09)")
_LOCATION_GENERIC_RE = re.compile(
    r"(?:\u5927\u90e8\u5206\u5730\u533a|\u90e8\u5206\u5730\u533a|\u5730\u533a|\u533a\u57df|"
    r"\u6d77\u57df|\u5c71\u533a|\u707e\u533a|\u73b0\u573a|\u8f96\u533a|\u57ce\u533a|\u5e02\u533a|\u53bf\u57ce|"
    r"\u5883\u5185|\u9644\u8fd1|\u4e00\u5e26|\u5468\u8fb9|\u5f53\u5730|"
    r"\u4e1c\u5317\u90e8|\u897f\u5317\u90e8|\u4e1c\u5357\u90e8|\u897f\u5357\u90e8|"
    r"\u4e1c\u90e8|\u897f\u90e8|\u5357\u90e8|\u5317\u90e8)+$"
)
_RELAXED_QUANTITY_STATUS_RE = re.compile(
    r"(?:\u81f3\u5c11|\u7ea6|\u8fd1|\u8d85|\u8d85\u8fc7|\u903e|\u5171|\u5df2\u6709|\u7d2f\u8ba1)?"
    r"(?:\d+(?:\.\d+)?|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u4ebf\u4e24\u51e0\u6570\u4e0a]+)"
    r"(?:\u4f59|\u591a)?(?:\u4eba|\u540d|\u4f4d|\u4f8b|\u6237|\u5bb6)"
    r"(?:\u6b7b\u4ea1|\u9047\u96be|\u4e27\u751f|\u8eab\u4ea1|\u7f79\u96be|\u53d7\u4f24|\u53d7\u521b|\u91cd\u4f24|\u8f7b\u4f24|"
    r"\u5931\u8054|\u5931\u8e2a|\u4e0b\u843d\u4e0d\u660e|\u611f\u67d3|\u786e\u8bca|\u53d1\u75c5|\u4e2d\u6bd2|"
    r"\u53d7\u707e|\u88ab\u56f0|\u83b7\u6551|\u8f6c\u79fb|\u758f\u6563|\u5b89\u7f6e)?"
)
_CASUALTY_STATUS_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("death", ("\u6b7b\u4ea1", "\u9047\u96be", "\u4e27\u751f", "\u8eab\u4ea1", "\u7f79\u96be")),
    ("injured", ("\u53d7\u4f24", "\u53d7\u521b", "\u4f24\u8005", "\u91cd\u4f24", "\u8f7b\u4f24")),
    ("missing", ("\u5931\u8e2a", "\u5931\u8054", "\u4e0b\u843d\u4e0d\u660e")),
    ("infected", ("\u611f\u67d3", "\u786e\u8bca", "\u53d1\u75c5")),
    ("poisoned", ("\u4e2d\u6bd2",)),
    ("affected", ("\u53d7\u707e",)),
    ("trapped", ("\u88ab\u56f0",)),
    ("rescued", ("\u83b7\u6551", "\u6551\u51fa")),
    ("evacuated", ("\u8f6c\u79fb", "\u758f\u6563", "\u64a4\u79bb")),
    ("settled", ("\u5b89\u7f6e",)),
)
_CAUSE_SHELL_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^\u53d7(.+?)(?:\u5f71\u54cd|\u88ad\u51fb|\u4fb5\u88ad)$"), r"\1"),
    (re.compile(r"^\u7531(.+?)(?:\u5f15\u53d1|\u5bfc\u81f4|\u9020\u6210|\u5f15\u8d77|\u6240\u81f4|\u8bf1\u53d1)$"), r"\1"),
    (re.compile(r"^(?:\u56e0|\u7531\u4e8e)(.+)$"), r"\1"),
    (re.compile(r"(.+?)(?:\u5bfc\u81f4|\u9020\u6210|\u5f15\u53d1|\u5f15\u8d77|\u6240\u81f4|\u8bf1\u53d1).*$"), r"\1"),
)
_WEAK_TIME_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("time:recent", ("\u8fd1\u65e5", "\u8fd1\u671f", "\u6700\u8fd1", "\u8fde\u65e5\u6765", "\u8fd1\u671f\u4ee5\u6765")),
    ("time:same_day", ("\u5f53\u5929", "\u5f53\u65e5", "\u5f53\u665a", "\u5f53\u5929\u4e0a\u5348", "\u5f53\u5929\u665a\u4e0a", "\u4eca\u5929")),
    ("time:yesterday", ("\u6628\u65e5", "\u6628\u5929")),
    ("time:current_status", ("\u76ee\u524d", "\u622a\u81f3\u76ee\u524d", "\u622a\u81f3\u5f53\u65e5")),
    ("time:incident_relative", ("\u4e8b\u53d1\u65f6", "\u4e8b\u53d1\u540e")),
)
_RELAXED_ENTITY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "org": (
        "\u8b66\u65b9",
        "\u516c\u5b89",
        "\u6d88\u9632",
        "\u653f\u5e9c",
        "\u5e94\u6025\u90e8\u95e8",
        "\u6551\u63f4\u961f",
        "\u533b\u9662",
        "\u536b\u751f\u90e8\u95e8",
        "\u75be\u63a7\u4e2d\u5fc3",
        "\u7ea2\u5341\u5b57\u4f1a",
    ),
    "facility": (
        "\u7164\u77ff",
        "\u77ff\u4e95",
        "\u5de5\u5382",
        "\u4ed3\u5e93",
        "\u5b66\u6821",
        "\u533b\u9662",
        "\u6e05\u771f\u5bfa",
        "\u5e02\u573a",
        "\u4f53\u80b2\u4e2d\u5fc3",
        "\u9152\u5e97",
        "\u8f66\u7ad9",
        "\u6865\u6881",
        "\u9053\u8def",
        "\u96a7\u9053",
    ),
    "vehicle": (
        "\u5ba2\u8f66",
        "\u8d27\u8f66",
        "\u516c\u4ea4\u8f66",
        "\u9762\u5305\u8f66",
        "\u8f7f\u8f66",
        "\u6469\u6258\u8f66",
        "\u7535\u52a8\u8f66",
        "\u5217\u8f66",
        "\u98de\u673a",
        "\u8239",
        "\u6e21\u8239",
    ),
}


def _strip_relaxed_modifiers(text: str, role: str = "") -> str:
    text = normalize_metric_text(text)
    if not text:
        return ""
    prefixes = (
        "\u81f3\u5c11",
        "\u7ea6\u6709",
        "\u7ea6",
        "\u8fd1",
        "\u8d85\u8fc7",
        "\u8d85",
        "\u903e",
        "\u5171",
        "\u5df2\u6709",
        "\u7d2f\u8ba1",
        "\u9020\u6210",
        "\u5bfc\u81f4",
    )
    for prefix in prefixes:
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix) :]
            break
    if role in {"trigger", "related_event"}:
        for suffix in ("\u4e8b\u6545", "\u4e8b\u4ef6", "\u95ee\u9898"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
    if role in {"cause", "casualty", "rescued_people"}:
        for suffix in ("\u75c5\u6bd2", "\u4eba\u5458"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
    return normalize_metric_text(text)


def _strip_parenthetical(text: Any) -> str:
    return normalize_metric_text(_PAREN_CONTENT_RE.sub("", str(text or "")))


def _strip_location_generics(text: str) -> str:
    current = normalize_metric_text(text)
    while current:
        stripped = _LOCATION_GENERIC_RE.sub("", current)
        if stripped == current:
            break
        current = stripped
    return normalize_metric_text(current)


def _strip_cause_shells(text: str) -> str:
    current = normalize_metric_text(text)
    for pattern, replacement in _CAUSE_SHELL_PATTERNS:
        stripped = normalize_metric_text(pattern.sub(replacement, current))
        if stripped and stripped != current:
            return stripped
    return current


def _strip_cause_modifiers(text: str) -> str:
    current = normalize_metric_text(text)
    for prefix in ("\u6301\u7eed", "\u8fde\u7eed", "\u5f3a\u70c8", "\u4e25\u91cd", "\u7a81\u53d1"):
        if current.startswith(prefix) and len(current) > len(prefix):
            return normalize_metric_text(current[len(prefix) :])
    return current


def _add_relaxed_candidate(candidates: Set[str], value: Any) -> None:
    candidate = normalize_metric_text(value)
    if candidate:
        candidates.add(candidate)


def _add_entity_keyword_candidates(candidates: Set[str], text: str, group: str) -> None:
    for keyword in _RELAXED_ENTITY_KEYWORDS.get(group, ()):
        if keyword in text:
            candidates.add(f"{group}:{keyword}")


def _add_quantity_status_candidates(candidates: Set[str], text: str) -> None:
    for match in _RELAXED_QUANTITY_STATUS_RE.finditer(text):
        match_text = normalize_metric_text(match.group(0))
        if not match_text:
            continue
        _add_relaxed_candidate(candidates, _strip_relaxed_modifiers(match_text, "casualty"))
        quantities = [
            _strip_relaxed_modifiers(quantity.group(0), "casualty")
            for quantity in _QUANTITY_RE.finditer(match_text)
        ]
        for status_name, terms in _CASUALTY_STATUS_GROUPS:
            if any(term in match_text for term in terms):
                for quantity in quantities:
                    _add_relaxed_candidate(candidates, f"casualty:{status_name}:{quantity}")


def normalize_relaxed_candidates(value: Any, role: str = "") -> Set[str]:
    role = normalize_metric_text(role)
    raw_text = str(value or "")
    base = normalize_metric_text(raw_text)
    candidates: Set[str] = set()
    _add_relaxed_candidate(candidates, base)
    _add_relaxed_candidate(candidates, _strip_parenthetical(raw_text))
    _add_relaxed_candidate(candidates, _strip_relaxed_modifiers(base, role))

    if role in {"trigger", "related_event"}:
        for cluster_name, terms in _RELAXED_TRIGGER_CLUSTERS:
            if any(term in base for term in terms):
                candidates.add(cluster_name)

    if role in {"location", "affected_area"}:
        for candidate in list(candidates) or [base]:
            _add_relaxed_candidate(candidates, _strip_location_generics(candidate))
            _add_relaxed_candidate(candidates, _strip_admin_prefix(_strip_location_generics(candidate)))
        for head in argument_heads(base, "location"):
            _add_relaxed_candidate(candidates, head)
            _add_relaxed_candidate(candidates, _strip_location_generics(head))

    if role == "cause":
        for candidate in list(candidates) or [base]:
            shell_stripped = _strip_cause_shells(candidate)
            _add_relaxed_candidate(candidates, shell_stripped)
            _add_relaxed_candidate(candidates, _strip_cause_modifiers(shell_stripped))
            _add_relaxed_candidate(candidates, _strip_cause_modifiers(candidate))

    if role in {"casualty", "rescued_people"}:
        no_paren = _strip_parenthetical(raw_text)
        _add_quantity_status_candidates(candidates, no_paren or base)
        for head in argument_heads(no_paren or base, role):
            _add_relaxed_candidate(candidates, head)

    if role == "time":
        for group_name, terms in _WEAK_TIME_GROUPS:
            if any(term in base for term in terms):
                candidates.add(group_name)
        for head in argument_heads(base, role):
            _add_relaxed_candidate(candidates, head)

    if role in {"rescue_org", "response_org", "facility", "vehicle"}:
        for head in argument_heads(base, role):
            _add_relaxed_candidate(candidates, head)
        if role in {"rescue_org", "response_org"}:
            _add_entity_keyword_candidates(candidates, base, "org")
        elif role == "facility":
            _add_entity_keyword_candidates(candidates, base, "facility")
        elif role == "vehicle":
            _add_entity_keyword_candidates(candidates, base, "vehicle")

    return {candidate for candidate in candidates if candidate}


def _char_overlap_f1(a: str, b: str) -> float:
    a = normalize_metric_text(a)
    b = normalize_metric_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a_counter = Counter(a)
    b_counter = Counter(b)
    overlap = sum((a_counter & b_counter).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(b), 1)
    recall = overlap / max(len(a), 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def relaxed_value_score(gold_value: Any, pred_value: Any, role: str = "") -> float:
    role = normalize_metric_text(role)
    gold = normalize_metric_text(gold_value)
    pred = normalize_metric_text(pred_value)
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    if gold == pred:
        return 1.0

    gold_relaxed = _strip_relaxed_modifiers(gold, role)
    pred_relaxed = _strip_relaxed_modifiers(pred, role)
    if gold_relaxed and pred_relaxed and gold_relaxed == pred_relaxed:
        return 0.95

    gold_candidates = normalize_relaxed_candidates(gold, role)
    pred_candidates = normalize_relaxed_candidates(pred, role)
    if gold_candidates and pred_candidates and gold_candidates & pred_candidates:
        return 0.95

    for gold_candidate in gold_candidates:
        for pred_candidate in pred_candidates:
            shorter, longer = sorted((gold_candidate, pred_candidate), key=len)
            if len(shorter) >= 2 and shorter in longer:
                return 0.95

    gold_heads = set(argument_heads(gold, role) or [])
    pred_heads = set(argument_heads(pred, role) or [])
    if gold_heads and pred_heads and gold_heads & pred_heads:
        return 0.90

    shorter, longer = sorted((gold_relaxed or gold, pred_relaxed or pred), key=len)
    if len(shorter) >= 2 and shorter in longer:
        if role in {"time", "location", "facility", "cause", "trigger", "related_event"}:
            return 0.85
        if len(shorter) >= 3:
            return 0.75

    char_f1 = _char_overlap_f1(gold_relaxed or gold, pred_relaxed or pred)
    return char_f1 if char_f1 >= 0.70 else 0.0


def _relaxed_f1_from_items(
    gold_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    matcher,
) -> Dict[str, float]:
    used_gold: Set[int] = set()
    hit = 0
    for pred_item in pred_items:
        best_idx = -1
        best_score = 0.0
        for idx, gold_item in enumerate(gold_items):
            if idx in used_gold:
                continue
            score = float(matcher(gold_item, pred_item))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score > 0:
            used_gold.add(best_idx)
            hit += 1
    return _relaxed_metric_from_counts(hit, len(pred_items), len(gold_items))


def _relaxed_metric_from_counts(hit: int, pred_total: int, gold_total: int) -> Dict[str, float]:
    metric = _metric_f1_from_counts(hit, pred_total, gold_total)
    if gold_total == 0 and pred_total > 0:
        metric["precision"] = max(metric["precision"], RELAXED_MISSING_GOLD_FLOOR)
        metric["recall"] = max(metric["recall"], RELAXED_MISSING_GOLD_FLOOR)
        metric["f1"] = max(metric["f1"], RELAXED_MISSING_GOLD_FLOOR)
        metric["missing_gold_adjusted"] = 1.0
    else:
        metric["missing_gold_adjusted"] = 0.0
    return metric


def _event_items_for_relaxed_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    items = {
        "triggers": [],
        "args": [],
        "spans": [],
    }
    if not payload or not isinstance(payload.get("events"), list):
        return items
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        event_type = normalize_metric_text(event.get("event_type", ""))
        trigger = normalize_metric_text(event.get("trigger", ""))
        if event_type and trigger:
            items["triggers"].append({"event_type": event_type, "trigger": trigger})
        arguments = event.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        for role, values in arguments.items():
            role_text = normalize_metric_text(role)
            if not isinstance(values, list):
                values = [values]
            for value in values:
                span = normalize_metric_text(value)
                if not span:
                    continue
                item = {
                    "event_type": event_type,
                    "trigger": trigger,
                    "role": role_text,
                    "value": span,
                }
                items["args"].append(item)
                items["spans"].append({"role": role_text, "value": span})
    return items


def relaxed_event_metrics(
    gold_payload: Dict[str, Any],
    pred_payload: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    gold_items = _event_items_for_relaxed_metrics(gold_payload)
    pred_items = _event_items_for_relaxed_metrics(pred_payload)

    def trigger_match(gold_item, pred_item):
        if gold_item.get("event_type") != pred_item.get("event_type"):
            return 0.0
        return relaxed_value_score(gold_item.get("trigger"), pred_item.get("trigger"), "trigger")

    def span_match(gold_item, pred_item):
        role = gold_item.get("role") if gold_item.get("role") == pred_item.get("role") else gold_item.get("role", "")
        return relaxed_value_score(gold_item.get("value"), pred_item.get("value"), role)

    def arg_i_match(gold_item, pred_item):
        if gold_item.get("event_type") != pred_item.get("event_type"):
            return 0.0
        return relaxed_value_score(gold_item.get("value"), pred_item.get("value"), gold_item.get("role", ""))

    def arg_c_match(gold_item, pred_item):
        if gold_item.get("event_type") != pred_item.get("event_type"):
            return 0.0
        if gold_item.get("role") != pred_item.get("role"):
            return 0.0
        return relaxed_value_score(gold_item.get("value"), pred_item.get("value"), gold_item.get("role", ""))

    def partial_arg_match(gold_item, pred_item):
        if gold_item.get("event_type") != pred_item.get("event_type"):
            return 0.0
        if gold_item.get("role") != pred_item.get("role"):
            return 0.0
        if relaxed_value_score(gold_item.get("value"), pred_item.get("value"), gold_item.get("role", "")) > 0:
            return 1.0
        return 1.0 if _char_overlap_f1(gold_item.get("value", ""), pred_item.get("value", "")) >= 0.50 else 0.0

    return {
        "soft_trigger": _relaxed_f1_from_items(
            gold_items["triggers"],
            pred_items["triggers"],
            trigger_match,
        ),
        "relaxed_span": _relaxed_f1_from_items(
            gold_items["spans"],
            pred_items["spans"],
            span_match,
        ),
        "normalized_arg_i": _relaxed_f1_from_items(
            gold_items["args"],
            pred_items["args"],
            arg_i_match,
        ),
        "normalized_arg_c": _relaxed_f1_from_items(
            gold_items["args"],
            pred_items["args"],
            arg_c_match,
        ),
        "partial_argument": _relaxed_f1_from_items(
            gold_items["args"],
            pred_items["args"],
            partial_arg_match,
        ),
    }


def empty_relaxed_metric_counts() -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "hit": 0.0,
            "pred": 0.0,
            "gold": 0.0,
            "normal_n": 0.0,
            "special_precision_sum": 0.0,
            "special_recall_sum": 0.0,
            "special_f1_sum": 0.0,
            "special_n": 0.0,
        }
        for key in RELAXED_METRIC_KEYS
    }


def add_relaxed_metric_counts(
    totals: Dict[str, Dict[str, float]],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    for key in RELAXED_METRIC_KEYS:
        item = metrics.get(key, {})
        pred_total = int(item.get("pred", 0))
        gold_total = int(item.get("gold", 0))
        if gold_total == 0 and pred_total > 0:
            totals[key]["special_precision_sum"] += max(
                float(item.get("precision", 0.0)),
                RELAXED_MISSING_GOLD_FLOOR,
            )
            totals[key]["special_recall_sum"] += max(
                float(item.get("recall", 0.0)),
                RELAXED_MISSING_GOLD_FLOOR,
            )
            totals[key]["special_f1_sum"] += max(
                float(item.get("f1", 0.0)),
                RELAXED_MISSING_GOLD_FLOOR,
            )
            totals[key]["special_n"] += 1.0
            continue
        totals[key]["hit"] += int(item.get("hit", 0))
        totals[key]["pred"] += pred_total
        totals[key]["gold"] += gold_total
        totals[key]["normal_n"] += 1.0


def finalize_relaxed_metric_counts(
    totals: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    finalized = {}
    for key, item in totals.items():
        normal_metric = _metric_f1_from_counts(
            int(item.get("hit", 0.0)),
            int(item.get("pred", 0.0)),
            int(item.get("gold", 0.0)),
        )
        normal_n = int(item.get("normal_n", 0.0))
        special_n = int(item.get("special_n", 0.0))
        result = dict(normal_metric)
        result["normal_n"] = float(normal_n)
        result["special_n"] = float(special_n)
        if special_n > 0:
            special_precision = float(item.get("special_precision_sum", 0.0)) / special_n
            special_recall = float(item.get("special_recall_sum", 0.0)) / special_n
            special_f1 = float(item.get("special_f1_sum", 0.0)) / special_n
            if normal_n > 0:
                total_n = normal_n + special_n
                result["precision"] = (normal_metric["precision"] * normal_n + item["special_precision_sum"]) / total_n
                result["recall"] = (normal_metric["recall"] * normal_n + item["special_recall_sum"]) / total_n
                result["f1"] = (normal_metric["f1"] * normal_n + item["special_f1_sum"]) / total_n
            else:
                result["precision"] = special_precision
                result["recall"] = special_recall
                result["f1"] = special_f1
        finalized[key] = result
    return finalized


def empty_standard_metric_counts() -> Dict[str, Dict[str, int]]:
    return {
        key: {"hit": 0, "pred": 0, "gold": 0}
        for key in ("trigger", "arg_i", "arg_c", "span", "head")
    }


def add_standard_metric_counts(
    totals: Dict[str, Dict[str, int]],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    for key in ("trigger", "arg_i", "arg_c", "span", "head"):
        item = metrics.get(key, {})
        totals[key]["hit"] += int(item.get("hit", 0))
        totals[key]["pred"] += int(item.get("pred", 0))
        totals[key]["gold"] += int(item.get("gold", 0))


def finalize_standard_metric_counts(
    totals: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    return {
        key: _metric_f1_from_counts(
            int(item.get("hit", 0)),
            int(item.get("pred", 0)),
            int(item.get("gold", 0)),
        )
        for key, item in totals.items()
    }


def _sentence_lookup(example: Dict[str, Any]) -> Dict[str, str]:
    return {f"S{i}": sentence for i, sentence in enumerate(example.get("sentences", []))}


def _normalise_values(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    if value is None or value == "":
        return []
    return [str(value)]
