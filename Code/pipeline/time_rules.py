from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

from . import normalization as common

VERSION = "event-reference-temporal-rules-2"
TIME_FIELD = "Error time and Error date"
CARDINAL_WORD = rf"(?:{common.NUMBER_WORD}|hundred|thousand)"
CARDINAL = rf"{CARDINAL_WORD}(?:[ -]+(?:and[ -]+)?{CARDINAL_WORD})*"
QUANTITY = rf"(?:(?:\d+|{CARDINAL})\s+and\s+a\s+half|\d+(?:\.\d+)?|{CARDINAL}|an?|half|a quarter)"
UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
PART = rf"{QUANTITY}\s+{UNIT}"
DURATION = rf"(?:{PART}(?:\s+(?:and\s+)?{PART})*|(?:half|a quarter)\s+(?:of\s+)?an?\s+hour)"
DIRECTION = r"(?:prior\s+to|subsequent\s+to|later|earlier|afterwards?|beforehand|after|before|following|into|since)"
RELATIVE = re.compile(rf"(?<![\w.])(?P<duration>{DURATION})\s+(?P<direction>{DIRECTION})\b", re.I)
PART_RE = re.compile(rf"(?P<n>{QUANTITY})\s+(?P<unit>{UNIT})\b", re.I)
APPROXIMATE = re.compile(
    r"\b(?:about|around|roughly|approximately|nearly|almost|over|under|at least|at most|"
    r"more than|less than|between)\s*(?:\d+\s*(?:to|and|[-–])\s*)?$",
    re.I,
)
RANGE_PREFIX = re.compile(rf"\b{QUANTITY}\s*(?:to|and|[-–])\s*$", re.I)
INCIDENT = re.compile(
    r"\b(?:incident|fault|error|failure|failed|malfunction\w*|breakdown|alarm\w*|"
    r"problem|issue|stoppage|stopped|jammed|stalled|seized|shutdown)\b",
    re.I,
)
NONFACTUAL = re.compile(
    r"\b(?:if|would|could|might|should|will|planned|scheduled|expected|"
    r"no|not|never|without)\b",
    re.I,
)
START = re.compile(
    r"\b(?:start(?:ed|ing)?|began|begin(?:ning)?|commenc(?:ed|ement)|kick(?:ed)?\s+off)\b", re.I
)
END = re.compile(r"\b(?:end(?:ed|ing)?|finish(?:ed)?|complet(?:ed|ion)|wrap(?:ped)?\s+up)\b", re.I)
ROUTINE = {
    "check",
    "inspection",
    "repair",
    "maintenance",
    "restart",
    "test",
    "handover",
    "review",
    "record",
    "report",
    "break",
    "measurement",
}
ALIASES = {
    "inspected": "inspection",
    "inspect": "inspection",
    "inspections": "inspection",
    "checked": "check",
    "checks": "check",
    "checking": "check",
    "tested": "test",
    "tests": "test",
    "repaired": "repair",
    "repairs": "repair",
    "restarted": "restart",
    "reviewed": "review",
    "recorded": "record",
    "reported": "report",
}
STOP = set(
    (
        "a an the this that these those then it its we i he she they my our your their his her "
        "at on in of to from for by with as and or but when while was were is are had has have "
        "been being be did do done took take place occurred occurring happened happening "
        "exactly later earlier after before prior following into since minutes minute hours hour "
        "seconds second days day weeks week elapsed first last final previous next same "
        "start started starting began begin beginning commenced commencement end ended ending "
        "finish finished complete completed completion kicked kick off wrapped wrap up "
        "routine normal normally"
    ).split()
)
OFFSET_CLOCK = re.compile(
    rf"\b(?P<n>{common.NUMBER})\s+(?:minutes?\s+)?(?P<direction>past|to)\s+"
    rf"(?P<hour>{common.NUMBER}|noon|midnight)(?:\s+(?P<suffix>{common.MERIDIEM}))?\b",
    re.I,
)
SPOKEN_DIGIT_CLOCK = re.compile(
    rf"\bat\s+{common.DIGIT_WORD}(?:[ -]+{common.DIGIT_WORD}){{3}}\b", re.I
)
SECOND_CLOCK = re.compile(
    rf"(?<!\w)\d{{1,2}}:\d{{2}}:\d{{2}}(?:\.\d+)?(?:\s*{common.MERIDIEM})?(?!\w)", re.I
)


@dataclass
class Event:
    index: int
    clause: int
    start: int
    end: int
    description: str
    kind: str
    names: set[str]
    phase: str | None
    explicit_target: bool
    routine: bool
    factual: bool
    value: datetime | None = None
    relation: str | None = None
    reference: str = ""
    reference_clock: int | None = None
    seconds: Decimal | None = None
    dependencies: list[int] = field(default_factory=list)
    reason: str = ""


def _descriptor(text):
    words = [ALIASES.get(w, w) for w in re.findall(r"[a-z]+", text.lower())]
    names = set(words) - STOP - set(common.NUMBERS)
    phase = "end" if END.search(text) else "start" if START.search(text) else None
    explicit = bool(INCIDENT.search(text))
    routine = bool(names & ROUTINE) or (
        phase == "start" and bool(names & {"run", "work", "cycle", "operation", "production"})
    )
    return names, phase, explicit, routine, not bool(NONFACTUAL.search(text))


def _cardinal(value):
    value = value.replace("-", " ").strip()
    for scale, size, upper in [("thousand", 1000, 999), ("hundred", 100, 9)]:
        if re.search(rf"\b{scale}\b", value):
            parts = re.split(rf"\b{scale}\b", value)
            if len(parts) != 2:
                return None
            head = _cardinal(parts[0].strip())
            tail = re.sub(r"^and\s+", "", parts[1].strip())
            rest = _cardinal(tail) if tail else 0
            if head is None or not 1 <= head <= upper or rest is None or not 0 <= rest < size:
                return None
            return head * size + rest
    return common.number(value)


def _seconds(text):
    text = re.sub(r"\bhalf\s+(?:of\s+)?an?\s+hour\b", "0.5 hour", text, flags=re.I)
    text = re.sub(r"\ba quarter\s+(?:of\s+)?an?\s+hour\b", "0.25 hour", text, flags=re.I)
    total = Decimal(0)
    for part in PART_RE.finditer(text):
        value = part["n"].lower()
        if value in {"a", "an", "half", "a quarter"}:
            n = {
                "a": Decimal(1),
                "an": Decimal(1),
                "half": Decimal(".5"),
                "a quarter": Decimal(".25"),
            }[value]
        elif value.endswith(" and a half"):
            parsed = _cardinal(value.removesuffix(" and a half"))
            if parsed is None:
                return None
            n = Decimal(parsed) + Decimal(".5")
        else:
            try:
                n = Decimal(value)
            except InvalidOperation:
                parsed = _cardinal(value)
                if parsed is None:
                    return None
                n = Decimal(parsed)
        unit = part["unit"].lower()
        scale = (
            1
            if unit.startswith("s")
            else 60
            if unit.startswith("m")
            else 3600
            if unit.startswith("h")
            else 86400
            if unit.startswith("d")
            else 604800
        )
        total += n * scale
    return total


def _clock_spans(text, base):
    spans = base + [
        common.ner_utils.Span(m.start(), m.end(), "TIME")
        for pattern in (OFFSET_CLOCK, SPOKEN_DIGIT_CLOCK, SECOND_CLOCK)
        for m in pattern.finditer(text)
    ]
    result = []
    for span in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if not any(span.start < other.end and other.start < span.end for other in result):
            result.append(span)
    return sorted(result, key=lambda s: s.start)


def _clock_value(text):
    offset = OFFSET_CLOCK.fullmatch(text)
    if offset:
        amount = common.number(offset["n"])
        name = offset["hour"].lower()
        hour = 0 if name == "midnight" else 12 if name == "noon" else common.number(name)
        if amount is None or not 0 <= amount < 60 or hour is None:
            return None
        suffix = offset["suffix"] or ""
        base = common.normalize_time(f"{hour}:00 {suffix}".strip())
        if base is None:
            return None
        minute = (
            base[0] * 60 + amount * (1 if offset["direction"].lower() == "past" else -1)
        ) % 1440
        return minute // 60, minute % 60, 0
    if SPOKEN_DIGIT_CLOCK.fullmatch(text):
        digits = "".join(common.DIGITS[w] for w in re.findall(r"[a-z]+", text.lower())[1:])
        hour, minute = int(digits[:2]), int(digits[2:])
        return (hour, minute, 0) if hour < 24 and minute < 60 else None
    if SECOND_CLOCK.fullmatch(text):
        match = re.fullmatch(r"(\d{1,2}:\d{2}):(\d{2})(.*)", text)
        if match[3].startswith("."):
            return None
        hm = common.normalize_time(match[1] + match[3])
        second = int(match[2])
        return (*hm, second) if hm and second < 60 else None
    hm = common.normalize_time(text)
    return (*hm, 0) if hm else None


def _clauses(text, protected):
    masked = list(text)
    for a, b in protected:
        masked[a:b] = " " * (b - a)
    clean = "".join(masked)
    cuts = [(m.start(), m.end()) for m in re.finditer(r"[,;.!?\n]+", clean)]
    for m in re.finditer(r"\b(?:and|but|while|whereas|with)\s+", clean, re.I):
        left = max([b for _, b in cuts if b <= m.start()] or [0])
        right = min([a for a, _ in cuts if a >= m.end()] or [len(text)])
        following = [a for a, _ in protected if m.end() <= a < right]
        subject = text[m.end() : min(following)].strip() if following else ""
        if subject and any(left <= a < m.start() for a, _ in protected) and following:
            cuts.append((m.start(), m.end()))
    parts, start = [], 0
    for a, b in sorted(cuts):
        if a < start:
            continue
        if text[start:a].strip():
            parts.append((start, a))
        start = b
    if text[start:].strip():
        parts.append((start, len(text)))
    return parts


def _reference_and_event(tail, has_prefix):
    if has_prefix:
        return tail.strip(), ""
    verb = re.search(r"\b(?:began|started|ended|finished|completed|wrapped up)\b", tail, re.I)
    boundaries = []
    if verb and tail[verb.end() :].strip():
        boundaries.append(verb.end())
    for m in re.finditer(r"\b(?:the|a|an|my|our|their|it|we|I)\b", tail, re.I):
        before = tail[: m.start()].strip()
        if (
            before
            and not re.search(r"\b(?:of|by|at|in|on|from|to|with)$", before, re.I)
            and _descriptor(before)[0]
        ):
            boundaries.append(m.start())
    if boundaries:
        cut = min(boundaries)
        return tail[:cut].strip(), tail[cut:].strip()
    return tail.strip(), ""


def _build_events(text, relations):
    surfaces = common.candidates(text)
    clocks, dates = _clock_spans(text, surfaces["TIME"]), surfaces["DATE"]
    protected = [(s.start, s.end) for s in clocks + dates] + [
        (m.start(), m.end()) for m in relations
    ]
    clauses = _clauses(text, protected)
    events = []

    def add(clause, start, end, description, kind, **kwargs):
        names, phase, explicit, routine, factual = _descriptor(description)
        event = Event(
            len(events),
            clause,
            start,
            end,
            description.strip(),
            kind,
            names,
            phase,
            explicit,
            routine,
            factual,
            **kwargs,
        )
        events.append(event)
        return event

    known_dates = {common.normalize_date(text[s.start : s.end]) for s in dates}
    clock_indices = {}
    for clock in clocks:
        ci = next(i for i, (a, b) in enumerate(clauses) if a <= clock.start < b)
        a, b = clauses[ci]
        local_dates = {
            common.normalize_date(text[s.start : s.end]) for s in dates if a <= s.start < b
        }
        choices = local_dates or known_dates
        date = next(iter(choices)) if len(choices) == 1 and None not in choices else None
        hm = _clock_value(text[clock.start : clock.end])
        prior = [m for m in relations if a <= m.start() < clock.start]
        desc_start = prior[-1].end() if prior else a
        description = text[desc_start : clock.start]
        for s in dates:
            description = description.replace(text[s.start : s.end], " ")
        if not prior and not _descriptor(description)[0] and not _descriptor(description)[1]:
            end = min([m.start() for m in relations if clock.end <= m.start() < b] + [b])
            suffix = text[clock.end : end]
            for s in dates:
                suffix = suffix.replace(text[s.start : s.end], " ")
            description += " " + suffix
        value = (
            datetime.combine(date, datetime.min.time()).replace(
                hour=hm[0], minute=hm[1], second=hm[2]
            )
            if date and hm
            else None
        )
        event = add(ci, clock.start, clock.end, description, "clock", value=value)
        if not date:
            event.reason = "ambiguous_or_missing_full_date"
        elif not hm:
            event.reason = "unparsed_clock"
        if re.match(r":\d", text[clock.end :]):
            event.value, event.reason = None, "unsupported_clock_precision"
        if not event.factual:
            event.value, event.reason = None, "nonfactual_event"
        clock_indices[(clock.start, clock.end)] = event.index

    for match in relations:
        ci = next(i for i, (a, b) in enumerate(clauses) if a <= match.start() < b)
        a, b = clauses[ci]
        prefix = text[a : match.start()].strip()
        prefix = re.sub(r"^(?:then|and|but|so|exactly)\b[ ,]*", "", prefix, flags=re.I)
        prefix = re.sub(r"\bexactly\s*$", "", prefix, flags=re.I).strip()
        same_clocks = [s for s in clocks if match.end() <= s.start < b]
        direction = re.sub(r"\s+", " ", match["direction"].lower())
        ref_clock = None
        if direction in {"later", "earlier", "afterward", "afterwards", "beforehand"}:
            reference, suffix = "", text[match.end() : b]
        elif same_clocks:
            reference = text[match.end() : same_clocks[0].start]
            suffix = ""
            if len(same_clocks) == 1:
                ref_clock = clock_indices[(same_clocks[0].start, same_clocks[0].end)]
        else:
            reference, suffix = _reference_and_event(text[match.end() : b], bool(prefix))
        description = (prefix + " " + suffix).strip()
        if not description and ci + 1 < len(clauses):
            na, nb = clauses[ci + 1]
            gap = text[b:na]
            if (
                "," in gap
                and not re.search(r"[.;!?\n]", gap)
                and not any(na <= m.start() < nb for m in relations)
                and not any(na <= s.start < nb for s in clocks)
            ):
                description = text[na:nb]
        seconds = _seconds(match["duration"])
        if direction in {"before", "earlier", "prior to", "beforehand"} and seconds is not None:
            seconds = -seconds
        event = add(
            ci,
            match.start(),
            match.end(),
            description,
            "relative",
            relation=match.group(),
            reference=reference,
            reference_clock=ref_clock,
            seconds=seconds,
        )
        if direction == "into":
            event.reference = "start of " + reference
        if APPROXIMATE.search(text[a : match.start()]) or RANGE_PREFIX.search(
            text[a : match.start()]
        ):
            event.reason = "inexact_duration"
        elif seconds is None:
            event.reason = "unparsed_duration"
        elif len(same_clocks) > 1:
            event.reason = "ambiguous_reference"
        if re.search(r"\b(?:ago|now|today|yesterday|tomorrow)\b", reference, re.I):
            event.reason = "unsupported_deictic_reference"
    for ci, (a, b) in enumerate(clauses):
        if any(e.clause == ci for e in events):
            continue
        description = text[a:b]
        names, phase, _, routine, factual = _descriptor(description)
        if factual and (routine or phase):
            add(ci, a, b, description, "untimed", reason="missing_clock_anchor")
    return events


def _reference_candidates(event, events):
    if event.reference_clock is not None:
        return [event.reference_clock]
    ref = event.reference.strip()
    names, phase, _, _, _ = _descriptor(ref)
    candidates = [e for e in events if e.index != event.index and e.factual]
    if names or phase:
        if names:
            candidates = [e for e in candidates if names & e.names]
        if phase:
            candidates = [e for e in candidates if e.phase == phase]
        if not candidates:
            return []
        if re.search(r"\b(?:that|this|previous|last|most recent)\b", ref, re.I):
            preceding = [e for e in candidates if e.start < event.start]
            if preceding:
                group = max(e.clause for e in preceding)
                candidates = [e for e in preceding if e.clause == group]
        elif re.search(r"\bfirst\b", ref, re.I):
            group = min(e.clause for e in candidates)
            candidates = [e for e in candidates if e.clause == group]
    else:
        candidates = [e for e in candidates if e.start < event.start]
        if not candidates:
            return []
        group = max(e.clause for e in candidates)
        candidates = [e for e in candidates if e.clause == group]
        derived = [e for e in candidates if e.kind == "relative"]
        if derived:
            candidates = derived
    return [e.index for e in candidates]


def resolve(text, prediction):
    result = dict(prediction)
    relations = list(RELATIVE.finditer(text))
    trace = {"version": VERSION, "status": "no_relative_expression", "steps": [], "events": []}
    if not relations:
        return result, trace
    events = _build_events(text, relations)
    for event in events:
        if event.kind == "relative" and not event.reason:
            event.dependencies = _reference_candidates(event, events)
            if not event.dependencies:
                event.reason = "unresolved_reference"

    def evaluate(index, visiting):
        event = events[index]
        if event.value is not None or event.reason:
            return event.value
        if index in visiting:
            event.reason = "cyclic_reference"
            return None
        if not event.factual:
            event.reason = "nonfactual_event"
            return None
        if not event.dependencies:
            event.reason = "missing_clock_anchor"
            return None
        values = [evaluate(i, visiting | {index}) for i in event.dependencies]
        if any(v is None for v in values):
            event.reason = "unresolved_reference"
            return None
        if len(set(values)) != 1:
            event.reason = "ambiguous_reference"
            return None
        micros = event.seconds * 1_000_000
        if micros != micros.to_integral_value():
            event.reason = "unsupported_duration_precision"
            return None
        try:
            event.value = values[0] + timedelta(microseconds=int(micros))
        except (OverflowError, ValueError):
            event.reason = "out_of_range_time"
        return event.value

    for event in events:
        evaluate(event.index, set())
    used = {i for event in events for i in event.dependencies}
    targets = [e for e in events if e.explicit_target and e.factual]
    selection = "explicit_event"
    if not targets:
        targets = [
            e
            for e in events
            if e.kind == "relative"
            and e.index not in used
            and e.description
            and not e.routine
            and e.factual
        ]
        selection = "unique_terminal_event"
    for event in events:
        record = {
            "id": event.index,
            "kind": event.kind,
            "description": event.description,
            "reference": event.reference,
            "dependencies": event.dependencies,
            "value": event.value.isoformat(sep=" ") if event.value is not None else None,
            "status": event.reason or "resolved",
        }
        trace["events"].append(record)
        if event.kind == "relative" and event.value is not None:
            trace["steps"].append(
                {
                    "expression": event.relation,
                    "reference": event.reference,
                    "dependencies": event.dependencies,
                    "value": event.value.isoformat(sep=" "),
                }
            )
    trace["selection"] = selection
    trace["target_events"] = [e.index for e in targets]
    result[TIME_FIELD] = common.MISSING
    if not targets:
        trace["status"] = "no_identifiable_target_event"
    elif any(e.value is None for e in targets):
        trace["status"] = next(e.reason or "unresolved_target" for e in targets if e.value is None)
    elif len({e.value for e in targets}) != 1:
        trace["status"] = "ambiguous_target_time"
    elif any(e.value.second or e.value.microsecond for e in targets):
        trace["status"] = "subminute_precision"
    else:
        result[TIME_FIELD] = targets[0].value.strftime("%Y/%m/%d %H:%M")
        trace["status"] = "resolved"
    return result, trace
