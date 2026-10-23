import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def config(output):
    return load(Path(output) / "config.json")


def records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def predictions(output, repetition=1):
    if repetition != 1:
        raise ValueError("Each result directory contains exactly one run.")
    return records(Path(output) / "records/predictions.jsonl")
