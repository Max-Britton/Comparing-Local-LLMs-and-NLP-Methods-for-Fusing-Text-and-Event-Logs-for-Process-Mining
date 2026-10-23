from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from . import ner_utils

VERSION = "shared-extraction-surface-2"
CATALOG = (
    "Turning & Milling",
    "Round Grinding",
    "Laser Marking",
    "Lapping",
    "Wire Cut",
    "Flat Grinding",
)
LABELS = ner_utils.ENTITY_LABELS
MISSING = ner_utils.NOT_SPECIFIED
DIGITS = {
    word: str(i)
    for i, word in enumerate("zero one two three four five six seven eight nine".split())
}
DIGITS["oh"] = "0"
NUMBERS = {v: k for k, v in ner_utils.CARDINALS.items()} | {
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORD = "(?:" + "|".join(sorted(NUMBERS, key=len, reverse=True)) + ")"
NUMBER = rf"(?:\d{{1,3}}|{NUMBER_WORD}(?:[ -]+{NUMBER_WORD})?)"
ORDINALS = {v: k for k, v in ner_utils.ORDINALS.items()} | {
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
ORDINALS.update({"twenty-" + ner_utils.ORDINALS[i]: 20 + i for i in range(1, 10)})
ORDINALS["thirty-first"] = 31
ORDINAL = (
    "(?:"
    + "|".join(re.escape(x).replace(r"\-", "[ -]") for x in sorted(ORDINALS, key=len, reverse=True))
    + ")"
)
DIGIT_WORD = "(?:" + "|".join(DIGITS) + ")"
WORKER_VALUE = rf"(?:\d(?:[ -]?\d){{2,5}}|{DIGIT_WORD}(?:[ -]+{DIGIT_WORD}){{2,5}})"
MACHINE_PATTERN = (
    rf"\b(?:(?:machine|unit|station)\s*(?:number\s*|no\.?\s*|#\s*)?(?:{ORDINAL}|{NUMBER})|"
    rf"{ORDINAL}\s+(?:machine|unit|station))\b"
)
WORKER_PATTERN = (
    rf"\b(?:(?:worker|operator|employee|technician|staff)(?:\s+with)?"
    rf"(?:\s+(?:personnel|badge))?(?:\s+(?:id|number|code))?\s*[:#-]?\s*"
    rf"|id\s*[:#-]?\s*){WORKER_VALUE}\b"
)
MONTH = "(?:" + "|".join(sorted(ner_utils.MONTHS, key=len, reverse=True)) + ")"
SMALL_WORDS = rf"{NUMBER_WORD}(?:[ -]+{NUMBER_WORD})?"
HUNDREDS = rf"{SMALL_WORDS}\s+hundred(?:\s+(?:and\s+)?{SMALL_WORDS})?"
THOUSANDS = rf"{NUMBER_WORD}\s+thousand(?:\s+(?:and\s+)?(?:{HUNDREDS}|{SMALL_WORDS}))?"
YEAR = rf"(?:\d{{4}}|{THOUSANDS}|{HUNDREDS}|{NUMBER_WORD}\s+(?:oh\s+)?{SMALL_WORDS})"
DAY = rf"(?:{ORDINAL}|\d{{1,2}}(?:st|nd|rd|th)?|{SMALL_WORDS})"
YEAR_SEPARATOR = r"(?:,\s*|\s+(?:in\s+the\s+year\s+)?)"
DATE_DMY = (
    rf"(?:the\s+)?(?P<day>{DAY})\s+(?:of\s+)?(?P<month>{MONTH}){YEAR_SEPARATOR}(?P<year>{YEAR})"
)
DATE_MDY = rf"(?P<month>{MONTH})\s+(?:the\s+)?(?P<day>{DAY}){YEAR_SEPARATOR}(?P<year>{YEAR})"
DATE_PATTERN = (
    r"\b(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    + re.sub(r"\(\?P<\w+>", "(?:", DATE_DMY + "|" + DATE_MDY)
    + r")\b"
)
MERIDIEM = r"(?:a\.?m\.?|p\.?m\.?|in the morning|in the afternoon|in the evening|at night)"
HOUR_WORDS = rf"(?:twenty[ -]+(?:one|two|three)|(?:zero|oh)[ -]+{DIGIT_WORD}|{NUMBER_WORD})"
MINUTE_WORDS = rf"(?:(?:zero|oh)[ -]+{DIGIT_WORD}|{SMALL_WORDS})"
TIME_PATTERN = (
    rf"(?<!\w)(?:\d{{1,2}}:\d{{2}}(?:\s*{MERIDIEM})?|"
    rf"\d{{1,2}}\s*{MERIDIEM}|(?:noon|midnight)|"
    rf"(?:quarter\s+(?:past|to)|half\s+past)\s+{NUMBER_WORD}(?:\s+{MERIDIEM})?|"
    rf"{HOUR_WORDS}(?:[ -]+{MINUTE_WORDS})?(?:\s+o.clock)?\s+{MERIDIEM}|"
    rf"at\s+{HOUR_WORDS}[ -]+{MINUTE_WORDS}(?:\s+{MERIDIEM})?)(?!\w)"
)


def number(value):
    tokens = re.findall(r"\d+|[a-z]+", value.lower())
    if len(tokens) == 1:
        return int(tokens[0]) if tokens[0].isdigit() else NUMBERS.get(tokens[0])
    if (
        len(tokens) == 2
        and NUMBERS.get(tokens[0], 0) >= 20
        and 0 < NUMBERS.get(tokens[1], 100) < 10
    ):
        return NUMBERS[tokens[0]] + NUMBERS[tokens[1]]
    return None


def normalize_worker(value):
    if not value:
        return MISSING
    cleaned = re.sub(
        r"\b(?:worker|operator|employee|technician|staff|with|personnel|badge|number|code)\b",
        "",
        value.lower(),
    )
    cleaned = re.sub(r"\bid\s*", "", cleaned).strip(" :#-")
    if re.fullmatch(r"\d(?:[ -]?\d){2,5}", cleaned):
        return "ID" + re.sub(r"\D", "", cleaned)
    tokens = re.findall(r"[a-z]+|\d+", cleaned)
    if 3 <= len(tokens) <= 6 and all(t in DIGITS for t in tokens):
        return "ID" + "".join(DIGITS[t] for t in tokens)
    return MISSING


def normalize_machine(value):
    if not value:
        return MISSING
    cleaned = re.sub(r"\b(?:machine|unit|station|number|no)\b\.?", "", value.lower()).strip(" :#-")
    n = number(cleaned)
    if n is None:
        n = ORDINALS.get(cleaned.replace(" ", "-"))
    return f"Machine {n}" if n is not None else MISSING


def normalize_activity(value):
    if not value:
        return MISSING
    words = lambda s: re.sub(r"\s+", " ", s.lower().replace("&", "and")).strip(" .,;:")
    matches = [a for a in CATALOG if words(a) == words(value)]
    return matches[0] if len(matches) == 1 else MISSING


def normalize_date(value):
    if not value:
        return None
    value = value.lower().strip()
    match = re.fullmatch(DATE_DMY, value) or re.fullmatch(DATE_MDY, value)
    if match:
        day = ORDINALS.get(match["day"].removeprefix("the ").replace(" ", "-"))
        if day is None:
            day = number(re.sub(r"(?<=\d)(?:st|nd|rd|th)$", "", match["day"]))
        year = normalize_year(match["year"])
        if day is None or year is None:
            return None
        parts = (year, ner_utils.MONTHS[match["month"]], day)
    else:
        parts = ner_utils.normalize_date(value, default_year=None)
    try:
        return datetime(*parts).date() if parts else None
    except ValueError:
        return None


def normalize_year(value):
    if re.fullmatch(r"\d{4}", value):
        year = int(value)
    else:
        tokens = value.replace("-", " ").split()

        def under_thousand(words):
            if words[:1] == ["and"]:
                words = words[1:]
            if not words:
                return 0
            if "hundred" not in words:
                return number(" ".join(words))
            pos = words.index("hundred")
            head = number(" ".join(words[:pos]))
            tail = words[pos + 1 :]
            if tail[:1] == ["and"]:
                tail = tail[1:]
            rest = number(" ".join(tail)) if tail else 0
            if head is None or not 1 <= head <= 99 or rest is None or not 0 <= rest < 100:
                return None
            return head * 100 + rest

        if "thousand" in tokens:
            pos = tokens.index("thousand")
            head = number(" ".join(tokens[:pos]))
            tail = under_thousand(tokens[pos + 1 :])
            if head is None or not 1 <= head <= 9 or tail is None or not 0 <= tail < 1000:
                return None
            year = head * 1000 + tail
        elif "hundred" in tokens:
            year = under_thousand(tokens)
        else:
            head = number(tokens[0]) if tokens else None
            tail = tokens[1:]
            if tail[:1] == ["oh"]:
                tail = tail[1:]
            rest = number(" ".join(tail))
            if head is None or not 10 <= head <= 99 or rest is None or not 0 <= rest < 100:
                return None
            year = head * 100 + rest
    return year if year is not None and 1000 <= year <= 9999 else None


def normalize_time(value):
    if not value:
        return None
    value = re.sub(r"^at\s+", "", value.lower().strip(" ,;"))
    meridiem = re.search(rf"\s*({MERIDIEM})$", value)
    suffix = meridiem.group(1).replace(".", "") if meridiem else ""
    core = value[: meridiem.start()].strip() if meridiem else value
    if re.fullmatch(r"\d{1,2}", core) and meridiem:
        result = (int(core), 0)
    elif re.fullmatch(r"\d{1,2}:\d{2}", core):
        result = tuple(map(int, core.split(":")))
    elif core in {"noon", "midnight"} or re.fullmatch(
        rf"(?:quarter (?:past|to)|half past) {NUMBER_WORD}", core
    ):
        result = ner_utils.normalize_time(core)
    else:
        tokens = core.replace("-", " ").replace("o'clock", "").split()
        leading_zero = len(tokens) >= 2 and tokens[0] in {"zero", "oh"} and tokens[1] in DIGITS
        width = (
            2
            if leading_zero
            or (len(tokens) >= 2 and tokens[0] == "twenty" and tokens[1] in {"one", "two", "three"})
            else 1
        )
        hour = number(tokens[1]) if leading_zero else number(" ".join(tokens[:width]))
        rest = tokens[width:]
        if len(rest) >= 2 and rest[0] in {"and", "oh", "zero"}:
            rest = rest[1:]
        if rest and rest[-1] == "minutes":
            rest = rest[:-1]
        minute = number(" ".join(rest)) if rest else 0
        result = (hour, minute) if hour is not None and minute is not None else None
    if result is None:
        return None
    hour, minute = result
    if suffix:
        if not 0 <= hour <= 12:
            return None
        if suffix == "at night":
            return None
        hour %= 12
        if suffix in {"pm", "in the afternoon", "in the evening"}:
            hour += 12
    return (hour, minute) if 0 <= hour < 24 and 0 <= minute < 60 else None


def candidates(text):
    activity = "|".join(
        re.escape(a).replace(r"\ ", r"\s+").replace(r"\&", "(?:&|and)") for a in CATALOG
    )
    patterns = {
        "ACTIVITY": rf"\b(?:{activity})\b",
        "MACHINE": MACHINE_PATTERN,
        "WORKER_ID": WORKER_PATTERN,
        "DATE": DATE_PATTERN,
        "TIME": TIME_PATTERN,
    }
    return {
        label: [ner_utils.Span(m.start(), m.end(), label) for m in re.finditer(p, text, re.I)]
        for label, p in patterns.items()
    }


NORMALIZERS = {
    "ACTIVITY": normalize_activity,
    "MACHINE": normalize_machine,
    "WORKER_ID": normalize_worker,
    "DATE": normalize_date,
    "TIME": normalize_time,
}


def build_spans(text, truth):
    dt = datetime.strptime(truth["Error time and Error date"], "%Y/%m/%d %H:%M")
    targets = {
        "ACTIVITY": truth["Activity"],
        "MACHINE": truth["Machine"],
        "WORKER_ID": truth["WorkerID"],
        "DATE": dt.date(),
        "TIME": (dt.hour, dt.minute),
    }
    found = candidates(text)
    spans, status = [], {}
    for label in LABELS:
        matches = [
            s for s in found[label] if NORMALIZERS[label](text[s.start : s.end]) == targets[label]
        ]
        if not matches:
            status[label] = "unresolved"
        elif any(s.start < other.end and other.start < s.end for s in matches for other in spans):
            status[label] = "overlap"
        else:
            spans.extend(matches)
            status[label] = "resolved"
    return sorted(spans, key=lambda s: s.start), status


def annotate(rows):
    result = []
    for index, row in enumerate(rows):
        spans, status = build_spans(row["error_text"], row["ground_truth"])
        result.append(
            {
                "row_id": index,
                "report_id": row["report_id"],
                "text": row["error_text"],
                "spans": [s.as_dict(row["error_text"]) for s in spans],
                "annotation_status": status,
            }
        )
    return result


def coverage(records):
    return {
        label: dict(Counter(r["annotation_status"][label] for r in records)) for label in LABELS
    }


def token_labels(offsets, spans):
    result, previous = [], None
    for start, end in offsets:
        if start == end:
            result.append(-100)
            previous = None
            continue
        index = next(
            (i for i, s in enumerate(spans) if start < s["end"] and end > s["start"]), None
        )
        label = (
            "O" if index is None else ("I-" if index == previous else "B-") + spans[index]["label"]
        )
        result.append(ner_utils.LABEL2ID[label])
        previous = index
    return result


def normalize_prediction(entities):
    values = {label: NORMALIZERS[label](ner_utils.best_entity(entities, label)) for label in LABELS}
    dt = MISSING
    if values["DATE"] is not None and values["TIME"] is not None:
        hour, minute = values["TIME"]
        dt = f"{values['DATE']:%Y/%m/%d} {hour:02d}:{minute:02d}"
    return {
        "Activity": values["ACTIVITY"],
        "Machine": values["MACHINE"],
        "WorkerID": values["WORKER_ID"],
        "Error time and Error date": dt,
    }
