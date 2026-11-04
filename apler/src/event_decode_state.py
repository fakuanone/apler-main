"""Incremental JSON event-array state tracking for autoregressive decoding."""

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


EVENT_BOUNDARY_CONTROL_MODES = {"off", "handle_soft", "handle_hierarchical"}
_EVENT_BOUNDARY_TOKEN_CACHE: Dict[Tuple[int, int], Dict[str, Tuple[Any, ...]]] = {}


@dataclass(frozen=True)
class EventDecodeState:
    """Structural position of an autoregressive decoder inside the events array."""

    current_event_index: int = 0
    completed_event_count: int = 0
    events_array_open: bool = False
    event_object_open: bool = False
    arguments_object_seen: bool = False
    arguments_object_open: bool = False
    present_roles: Tuple[str, ...] = ()


def _events_array_start(text: str) -> int:
    match = re.search(r'(?<!\\)"events"\s*:\s*\[', str(text))
    return match.end() - 1 if match else -1


def _scan_event_objects(text: str, array_start: int) -> Tuple[int, Optional[int], bool]:
    stack: List[str] = ["["]
    completed = 0
    current_event_start: Optional[int] = None
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
            if char == "}" and stack == ["["] and current_event_start is not None:
                completed += 1
                current_event_start = None
            if char == "]" and not stack:
                return completed, None, False

    return completed, current_event_start, bool(stack)


def _arguments_state(event_text: str) -> Tuple[bool, bool, Tuple[str, ...]]:
    matches = list(re.finditer(r'(?<!\\)"arguments"\s*:\s*\{', event_text))
    if not matches:
        return False, False, ()
    object_start = matches[-1].end() - 1
    depth = 1
    in_string = False
    escaped = False
    string_start = -1
    roles: List[str] = []
    expect_key = True

    index = object_start + 1
    while index < len(event_text):
        char = event_text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and expect_key:
                    cursor = index + 1
                    while cursor < len(event_text) and event_text[cursor].isspace():
                        cursor += 1
                    if cursor < len(event_text) and event_text[cursor] == ":":
                        role = event_text[string_start + 1 : index]
                        if role and role not in roles:
                            roles.append(role)
                        expect_key = False
            index += 1
            continue
        if char == '"':
            in_string = True
            string_start = index
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return True, False, tuple(roles)
        elif char == "," and depth == 1:
            expect_key = True
        index += 1
    return True, True, tuple(roles)


def inspect_event_decode_state(decoded_text: str) -> EventDecodeState:
    """Track events by completed top-level event objects, not emitted field names."""

    text = str(decoded_text)
    array_start = _events_array_start(text)
    if array_start < 0:
        return EventDecodeState()
    completed, event_start, array_open = _scan_event_objects(text, array_start)
    arguments_seen = False
    arguments_open = False
    present_roles: Tuple[str, ...] = ()
    if event_start is not None:
        arguments_seen, arguments_open, present_roles = _arguments_state(
            text[event_start:]
        )
    return EventDecodeState(
        current_event_index=completed,
        completed_event_count=completed,
        events_array_open=array_open,
        event_object_open=event_start is not None,
        arguments_object_seen=arguments_seen,
        arguments_object_open=arguments_open,
        present_roles=present_roles,
    )


def classify_event_token_transition(
    decoded_text: str,
    token_piece: str,
) -> Optional[str]:
    """Classify structural effects caused by appending one decoded token piece."""

    before = inspect_event_decode_state(decoded_text)
    if not before.events_array_open or not before.event_object_open:
        return None

    after = inspect_event_decode_state(f"{decoded_text}{token_piece}")
    if after.completed_event_count <= before.completed_event_count:
        return None
    if not after.events_array_open:
        return "stop"
    if after.event_object_open:
        return "continue"

    if str(f"{decoded_text}{token_piece}").rstrip().endswith(","):
        return "continue"
    return "complete"


def event_count_distribution(
    existence_probabilities: Sequence[float],
) -> Tuple[float, ...]:
    """Combine independent handle-existence probabilities into P(event_count)."""

    probabilities = [
        min(1.0, max(0.0, float(probability)))
        for probability in existence_probabilities
    ]
    distribution = [1.0] + [0.0] * len(probabilities)
    observed_handles = 0
    for probability in probabilities:
        observed_handles += 1
        for event_count in range(observed_handles, -1, -1):
            absent_mass = distribution[event_count] * (1.0 - probability)
            present_mass = (
                distribution[event_count - 1] * probability
                if event_count > 0
                else 0.0
            )
            distribution[event_count] = absent_mass + present_mass
    total_mass = sum(distribution)
    if total_mass <= 0.0:
        distribution = [1.0] + [0.0] * len(probabilities)
    else:
        distribution = [
            probability / total_mass for probability in distribution
        ]
    return tuple(distribution)


def expected_event_count(count_distribution: Sequence[float]) -> float:
    return sum(
        event_count * float(probability)
        for event_count, probability in enumerate(count_distribution)
    )


def event_count_reliability(count_distribution: Sequence[float]) -> float:
    """Return normalized certainty in [0, 1]; diffuse count beliefs approach zero."""

    distribution = [max(0.0, float(probability)) for probability in count_distribution]
    if len(distribution) <= 1:
        return 1.0
    entropy = -sum(
        probability * math.log(probability)
        for probability in distribution
        if probability > 0.0
    )
    max_entropy = math.log(float(len(distribution)))
    if max_entropy <= 0.0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - entropy / max_entropy))


def conditional_event_continue_probability(
    count_distribution: Sequence[float],
    completed_event_count: int,
) -> float:
    """Return P(N > k | N >= k) after k completed event objects."""

    distribution = [max(0.0, float(probability)) for probability in count_distribution]
    completed = max(0, int(completed_event_count))
    if completed >= len(distribution) - 1:
        return 0.0
    reached_mass = sum(distribution[completed:])
    if reached_mass <= 0.0:
        return 0.0
    continue_mass = sum(distribution[completed + 1 :])
    return min(1.0, max(0.0, continue_mass / reached_mass))


def normalize_event_boundary_control(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    if mode not in EVENT_BOUNDARY_CONTROL_MODES:
        raise ValueError(
            "event_boundary_control must be one of "
            f"{sorted(EVENT_BOUNDARY_CONTROL_MODES)}, got {value!r}."
        )
    return mode


def _decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return ""


def _event_boundary_piece_kind(piece: str) -> Optional[str]:
    piece = str(piece or "")
    stripped = piece.lstrip()
    if not stripped:
        return "whitespace" if piece else None
    if stripped.startswith(","):
        remainder = stripped[1:].lstrip()
        canonical_event_start = '{"event_type":'
        if (
            not remainder
            or canonical_event_start.startswith(remainder)
            or remainder.startswith(canonical_event_start)
        ):
            return "continue"
    if stripped.startswith("]"):
        remainder = stripped[1:].lstrip()
        if not remainder or (
            remainder.startswith("}") and not remainder[1:].strip()
        ):
            return "stop"
    if stripped.startswith("}") and not stripped[1:].strip():
        return "root_close"
    return None


def event_boundary_token_groups(tokenizer, vocab_size: int) -> Dict[str, Tuple[Any, ...]]:
    """Find native tokenizer pieces that can cross an event-array boundary."""

    vocab_size = int(vocab_size)
    cache_key = (id(tokenizer), vocab_size)
    cached = _EVENT_BOUNDARY_TOKEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    token_ids = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            token_ids = sorted(
                {
                    int(token_id)
                    for token_id in get_vocab().values()
                    if 0 <= int(token_id) < vocab_size
                }
            )
        except Exception:
            token_ids = None
    if not token_ids:
        token_ids = list(range(vocab_size))

    groups: Dict[str, List[Any]] = {
        "continue": [],
        "stop": [],
        "root_close": [],
        "whitespace": [],
        "event_tail": [],
        "multi_close": [],
        "event_close": [],
    }
    batch_decode = getattr(tokenizer, "batch_decode", None)
    for start in range(0, len(token_ids), 4096):
        token_id_chunk = token_ids[start : start + 4096]
        pieces = None
        if callable(batch_decode):
            try:
                pieces = batch_decode(
                    [[token_id] for token_id in token_id_chunk],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                pieces = None
        if pieces is None:
            pieces = [
                _decode_single_token(tokenizer, token_id)
                for token_id in token_id_chunk
            ]
        for token_id, piece in zip(token_id_chunk, pieces):
            kind = _event_boundary_piece_kind(piece)
            if kind is not None:
                groups[kind].append(int(token_id))
            decoded_piece = str(piece or "")
            if decoded_piece.lstrip().startswith("}"):
                groups["event_tail"].append((int(token_id), decoded_piece))
            if decoded_piece.count("}") >= 2:
                groups["multi_close"].append((int(token_id), decoded_piece))
            if "}" in decoded_piece:
                groups["event_close"].append((int(token_id), decoded_piece))

    result = {name: tuple(values) for name, values in groups.items()}
    if not result["continue"] or not result["stop"] or not result["root_close"]:
        raise RuntimeError(
            "Cannot locate tokenizer tokens for the top-level event-array boundary: "
            f"continue={len(result['continue'])}, stop={len(result['stop'])}, "
            f"root_close={len(result['root_close'])}."
        )
    _EVENT_BOUNDARY_TOKEN_CACHE[cache_key] = result
    return result


def at_completed_event_boundary(decoded_text: str) -> bool:
    state = inspect_event_decode_state(decoded_text)
    return (
        state.events_array_open
        and not state.event_object_open
        and state.completed_event_count >= 1
        and str(decoded_text or "").rstrip().endswith("}")
    )


def at_event_object_tail(decoded_text: str) -> bool:
    state = inspect_event_decode_state(decoded_text)
    return (
        state.events_array_open
        and state.event_object_open
        and state.arguments_object_seen
        and not state.arguments_object_open
        and str(decoded_text or "").rstrip().endswith("}")
    )


def direct_event_boundary_token_groups(
    tokenizer,
    vocab_size: int,
    decoded_text: str,
    state: Optional[EventDecodeState] = None,
) -> Dict[str, Tuple[int, ...]]:
    """Group native tokenizer pieces that directly commit continue or stop."""

    groups = event_boundary_token_groups(tokenizer, vocab_size)
    state = state or inspect_event_decode_state(decoded_text)
    transitions: Dict[str, List[int]] = {"continue": [], "stop": []}
    if not state.event_object_open:
        return {name: tuple(values) for name, values in transitions.items()}

    if state.arguments_object_open:
        candidates = groups["multi_close"]
    elif at_event_object_tail(decoded_text):
        candidates = groups["event_tail"]
    else:
        candidates = groups["event_close"]

    for token_id, piece in candidates:
        kind = classify_event_token_transition(decoded_text, piece)
        if kind in transitions:
            transitions[kind].append(int(token_id))
    return {
        name: tuple(dict.fromkeys(values))
        for name, values in transitions.items()
    }
