from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

FIELDS = ("Activity", "Machine", "Error time and Error date", "WorkerID")

ALLOWED_ACTIVITIES = (
    "Turning & Milling",
    "Round Grinding",
    "Laser Marking",
    "Lapping",
    "Wire Cut",
    "Flat Grinding",
)

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "Activity": {
            "anyOf": [
                {"type": "string", "enum": list(ALLOWED_ACTIVITIES)},
                {"type": "null"},
            ]
        },
        "Machine": {
            "anyOf": [
                {"type": "string", "pattern": r"^Machine [0-9]+$"},
                {"type": "null"},
            ]
        },
        "Error time and Error date": {
            "anyOf": [
                {
                    "type": "string",
                    "pattern": r"^[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}$",
                },
                {"type": "null"},
            ]
        },
        "WorkerID": {
            "anyOf": [
                {"type": "string", "pattern": r"^ID[0-9]+$"},
                {"type": "null"},
            ]
        },
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}


def parse_prediction(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return {field: parsed.get(field) for field in FIELDS}


def validate_prediction(prediction: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if set(prediction) != set(FIELDS):
        failures.append("wrong_keys")

    activity = prediction.get("Activity")
    if activity is not None and activity not in ALLOWED_ACTIVITIES:
        failures.append("invalid_activity")

    machine = prediction.get("Machine")
    if machine is not None and not re.fullmatch(r"Machine [0-9]+", str(machine)):
        failures.append("invalid_machine_format")

    timestamp = prediction.get("Error time and Error date")
    if timestamp is not None:
        if not re.fullmatch(r"[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}", str(timestamp)):
            failures.append("invalid_timestamp_format")
        elif pd.isna(pd.to_datetime(timestamp, format="%Y/%m/%d %H:%M", errors="coerce")):
            failures.append("invalid_timestamp_value")

    worker = prediction.get("WorkerID")
    if worker is not None and not re.fullmatch(r"ID[0-9]+", str(worker)):
        failures.append("invalid_worker_format")
    return failures


def score(raw, truth):
    prediction = {field: None for field in FIELDS}
    error = ""
    try:
        prediction = parse_prediction(raw)
        failures = validate_prediction(prediction)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failures = [error]
    valid = not failures
    correct = {field: bool(valid and prediction.get(field) == truth[field]) for field in FIELDS}
    return prediction, failures, valid, correct, error
