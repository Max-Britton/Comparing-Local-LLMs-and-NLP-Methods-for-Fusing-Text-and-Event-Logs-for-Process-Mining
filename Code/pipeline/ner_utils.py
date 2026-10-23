from __future__ import annotations

import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

ENTITY_LABELS = ("ACTIVITY", "MACHINE", "DATE", "TIME", "WORKER_ID")

BIO_LABELS = (
    "O",
    "B-ACTIVITY",
    "I-ACTIVITY",
    "B-MACHINE",
    "I-MACHINE",
    "B-DATE",
    "I-DATE",
    "B-TIME",
    "I-TIME",
    "B-WORKER_ID",
    "I-WORKER_ID",
)

LABEL2ID = {label: index for index, label in enumerate(BIO_LABELS)}

ID2LABEL = {index: label for label, index in LABEL2ID.items()}

NOT_SPECIFIED = "Not specified"

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

CARDINALS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}

ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
}

TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

    def as_dict(self, text: str) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": text[self.start : self.end],
        }


def token_labels(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[dict[str, Any]],
) -> list[int]:
    labels: list[int] = []
    previous_entity: str | None = None
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            previous_entity = None
            continue
        containing = next(
            (span for span in spans if start < int(span["end"]) and end > int(span["start"])),
            None,
        )
        if containing is None:
            labels.append(LABEL2ID["O"])
            previous_entity = None
            continue
        entity = str(containing["label"])
        prefix = "I" if previous_entity == entity else "B"
        labels.append(LABEL2ID[f"{prefix}-{entity}"])
        previous_entity = entity
    return labels


def require_transformer_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "NER dependencies are missing. Install them with "
            "`python -m pip install -r requirements.txt`. Python 3.14 "
            "requires PyTorch 2.10 or newer."
        ) from exc
    return torch, transformers


def set_global_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def choose_device(torch_module: Any, requested: str) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def make_dataset_class(torch_module: Any) -> type:
    class TokenDataset(torch_module.utils.data.Dataset):
        def __init__(self, records: Sequence[dict[str, Any]], tokenizer: Any, max_length: int):
            self.records = list(records)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            encoded = self.tokenizer(
                record["text"],
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
            encoded["labels"] = token_labels(offsets, record["spans"])
            encoded["row_id"] = record["row_id"]
            return encoded

    return TokenDataset


def make_collator(tokenizer: Any, torch_module: Any):
    def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        row_ids = [item.pop("row_id") for item in batch]
        labels = [item.pop("labels") for item in batch]
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        max_length = int(padded["input_ids"].shape[1])
        padded_labels = [label + [-100] * (max_length - len(label)) for label in labels]
        padded["labels"] = torch_module.tensor(padded_labels, dtype=torch_module.long)
        padded["row_ids"] = row_ids
        return padded

    return collate


def decode_entities(
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    confidences: Sequence[float],
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for (start, end), label_id, confidence in zip(offsets, label_ids, confidences):
        if start == end:
            continue
        label = ID2LABEL[int(label_id)]
        if label == "O":
            if active is not None:
                entities.append(active)
                active = None
            continue
        prefix, entity_type = label.split("-", maxsplit=1)
        should_start = (
            active is None
            or prefix == "B"
            or active["label"] != entity_type
            or start > active["end"] + 1
        )
        if should_start:
            if active is not None:
                entities.append(active)
            active = {
                "start": int(start),
                "end": int(end),
                "label": entity_type,
                "token_confidences": [float(confidence)],
            }
        else:
            active["end"] = int(end)
            active["token_confidences"].append(float(confidence))
    if active is not None:
        entities.append(active)
    for entity in entities:
        entity["text"] = text[entity["start"] : entity["end"]]
        entity["confidence"] = statistics.fmean(entity.pop("token_confidences"))
    return entities


def predict_entities(
    model: Any,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    device: Any,
    torch_module: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for record in records:
        start_time = time.perf_counter()
        encoded = tokenizer(
            record["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch_module.no_grad():
            logits = model(**inputs).logits[0]
            probabilities = torch_module.softmax(logits, dim=-1)
            confidences, label_ids = probabilities.max(dim=-1)
        entities = decode_entities(
            record["text"],
            offsets,
            label_ids.detach().cpu().tolist(),
            confidences.detach().cpu().tolist(),
        )
        predictions.append(
            {
                **record,
                "predicted_entities": entities,
                "latency_sec": time.perf_counter() - start_time,
            }
        )
    return predictions


def best_entity(entities: Sequence[dict[str, Any]], label: str) -> str | None:
    candidates = [entity for entity in entities if entity["label"] == label]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda entity: (float(entity.get("confidence", 0.0)), entity["end"] - entity["start"]),
    )
    return str(best["text"]).strip(" ,.;:()[]")


def normalize_date(value: str | None, default_year: int | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    text = value.strip()
    iso = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if iso:
        return tuple(int(part) for part in iso.groups())
    month_first = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*|\s+)?(\d{4})?\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_first and month_first.group(1).lower() in MONTHS:
        year = int(month_first.group(3)) if month_first.group(3) else default_year
        if year is not None:
            return year, MONTHS[month_first.group(1).lower()], int(month_first.group(2))
    day_first = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]+)"
        r"(?:,\s*|\s+)?(\d{4})?\b",
        text,
        flags=re.IGNORECASE,
    )
    if day_first and day_first.group(2).lower() in MONTHS:
        year = int(day_first.group(3)) if day_first.group(3) else default_year
        if year is not None:
            return year, MONTHS[day_first.group(2).lower()], int(day_first.group(1))
    lowered = text.lower()
    inverse_ordinals = {word: number for number, word in ORDINALS.items()}
    for ordinal, day in inverse_ordinals.items():
        match = re.search(
            rf"\b(?:the\s+)?{re.escape(ordinal)}\s+of\s+([a-z]+)"
            rf"(?:\s+in\s+the\s+year|,)?\s+(\d{{4}})\b",
            lowered,
        )
        if match and match.group(1) in MONTHS:
            return int(match.group(2)), MONTHS[match.group(1)], day
    last_february = re.search(
        r"\b(?:the\s+)?last\s+day\s+of\s+february\s+in\s+(\d{4})\b",
        lowered,
    )
    if last_february:
        year = int(last_february.group(1))
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return year, 2, 29 if is_leap else 28
    return None


def words_to_number(tokens: Sequence[str]) -> int | None:
    inverse = {word: number for number, word in CARDINALS.items()}
    inverse.update({word: number for number, word in TENS.items()})
    cleaned = [token for token in tokens if token not in {"oh", "and", "minutes"}]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return inverse.get(cleaned[0])
    if cleaned[0] in inverse and cleaned[1] in inverse:
        first, second = inverse[cleaned[0]], inverse[cleaned[1]]
        if first in TENS and second < 10:
            return first + second
    return None


def normalize_time(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    text = value.lower().replace("-", " ")
    special_tokens = re.findall(r"[a-z]+", text)
    if "midnight" in special_tokens:
        minute = words_to_number(special_tokens[special_tokens.index("midnight") + 1 :])
        return 0, minute or 0
    if "noon" in special_tokens:
        minute = words_to_number(special_tokens[special_tokens.index("noon") + 1 :])
        return 12, minute or 0
    numeric = re.search(
        r"\b(\d{1,2}):(\d{2})(?:\s*([ap])\.?m\.?)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if numeric:
        hour, minute = int(numeric.group(1)), int(numeric.group(2))
        meridiem = numeric.group(3)
        if meridiem:
            hour %= 12
            if meridiem.lower() == "p":
                hour += 12
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    tokens = re.findall(r"[a-z]+", text)
    inverse = {word: number for number, word in CARDINALS.items()}
    if "quarter" in tokens and "past" in tokens:
        hour = inverse.get(tokens[-1])
        return (hour, 15) if hour is not None else None
    if "half" in tokens and "past" in tokens:
        hour = inverse.get(tokens[-1])
        return (hour, 30) if hour is not None else None
    if "quarter" in tokens and "to" in tokens:
        next_hour = inverse.get(tokens[-1])
        return ((next_hour - 1) % 12, 45) if next_hour is not None else None
    if tokens:
        hour = inverse.get(tokens[0])
        if hour is not None:
            if len(tokens) == 1 or "o'clock" in text:
                minute_tokens = tokens[1:]
                minute = words_to_number(minute_tokens) if minute_tokens else 0
            else:
                minute = words_to_number(tokens[1:])
            if minute is not None and 0 <= minute <= 59:
                return hour, minute
    return None
