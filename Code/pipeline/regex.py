import json
import re


def extract_json_block(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print("Fehler beim Extrahieren des JSON-Blocks:", e)
    return ""


NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ACTIVITY_ALIASES = {
    "turning and milling": "Turning & Milling",
    "turning & milling": "Turning & Milling",
    "turning milling": "Turning & Milling",
    "milling": "Milling",
    "turning": "Turning",
    "lapping": "Lapping",
    "laser marking": "Laser Marking",
    "round grinding": "Round Grinding",
    "round-grinding": "Round Grinding",
    "flat grinding": "Flat Grinding",
    "flat-grinding": "Flat Grinding",
    "wire cut": "Wire Cut",
    "wire-cut": "Wire Cut",
    "wirecut": "Wire Cut",
    "wire cutting": "Wire Cut",
    "grinding": "Grinding",
}

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def normalize_machine(text):
    lower = str(text).lower()

    match = re.search(r"\bmachine\s*(\d+)\b", lower)
    if match:
        return f"Machine {match.group(1)}"

    match = re.search(r"\bmachine\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", lower)
    if match:
        return f"Machine {NUMBER_WORDS[match.group(1)]}"

    return "Not specified"


def normalize_worker_id(text):
    text = str(text)

    match = re.search(r"\bID\s*[- ]?(\d{3,6})\b", text, flags=re.IGNORECASE)
    if match:
        return f"ID{match.group(1)}"

    match = re.search(r"\bworker(?:\s+ID)?\s+(\d(?:\s*-\s*\d){2,6})\b", text, flags=re.IGNORECASE)
    if match:
        digits = re.sub(r"\D", "", match.group(1))
        if digits:
            return f"ID{digits}"

    return "Not specified"


def normalize_activity(text):
    lower = str(text).lower()

    for alias in sorted(ACTIVITY_ALIASES.keys(), key=len, reverse=True):
        if alias in lower:
            return ACTIVITY_ALIASES[alias]

    return "Not specified"


def normalize_error_datetime(text):
    text = str(text)

    match = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{2})\b", text)
    if match:
        year, month, day, hour, minute = match.groups()
        return f"{int(year):04d}/{int(month):02d}/{int(day):02d} {int(hour):02d}:{int(minute):02d}"

    match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})(?:\s+around|\s+at)?\s+(\d{1,2}):(\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        day, month_name, year, hour, minute = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{int(year):04d}/{month:02d}/{int(day):02d} {int(hour):02d}:{int(minute):02d}"

    match = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})(?:\s+around|\s+at)?\s+(\d{1,2}):(\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        month_name, day, year, hour, minute = match.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{int(year):04d}/{month:02d}/{int(day):02d} {int(hour):02d}:{int(minute):02d}"

    return "Not specified"


def extract_information_with_regex_baseline(error_text):
    extracted = {
        "Activity": normalize_activity(error_text),
        "Machine": normalize_machine(error_text),
        "Error time and Error date": normalize_error_datetime(error_text),
        "WorkerID": normalize_worker_id(error_text),
    }
    return "```json\n" + json.dumps(extracted, ensure_ascii=False, indent=4) + "\n```"
