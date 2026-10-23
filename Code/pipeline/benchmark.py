from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from . import classification as classifier
from . import component_timing, extraction, learning_curve, prediction_records, progress
from . import ollama_runtime as lifecycle
from . import prompts as prompt_templates
from . import request_timing as timing
from .taxonomy import CLASSIFICATION_LABELS, ERROR_CLASSES

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
VERSION = "benchmark-1"
BASELINES = {
    "regex": "extraction",
    "ner": "extraction",
    "crf": "extraction",
    "ner_time": "extraction",
    "crf_time": "extraction",
    "tfidf_logreg": "classification",
    "tfidf_svm": "classification",
    "bert": "classification",
    "bert_ner": "extraction",
    "bert_ner_time": "extraction",
}


@dataclass(frozen=True)
class Settings:
    input_path: Path
    output: Path
    models: list = field(default_factory=list)
    tasks: list = field(default_factory=lambda: ["extraction", "classification"])
    extraction_baselines: list = field(default_factory=lambda: ["regex"])
    classification_baselines: list = field(default_factory=lambda: ["tfidf_logreg"])
    extraction_shots: list = field(default_factory=lambda: [1])
    classification_shots: list = field(default_factory=lambda: [1])
    training_percentages: list = field(default_factory=lambda: [100])
    progress: bool = True
    repetitions: int = 1
    fractions: tuple = (0.7, 0.1, 0.2)
    resplit: bool = False
    split_seed: int = 42
    training_subset_seed: int | None = None
    validation_subset_seed: int | None = None
    validation_only: bool = False
    split_path: Path | None = None
    model_seed: int = 8100
    llm_seed: int = 8100
    method_seeds: dict = field(default_factory=dict)
    levels: list = field(default_factory=lambda: [1, 2, 3, 4])
    host: str = "http://127.0.0.1:11435"
    temperature: float = 0
    num_ctx: int = 4096
    extraction_tokens: int = 180
    classification_tokens: int = 100
    thinking_tokens: int = 8192
    timeout: int = 600
    ner_model: str = "distilbert-base-uncased"
    ner_epochs: int = 10
    ner_batch_size: int = 4
    ner_lr: float = 2e-5
    ner_device: str = "auto"
    compact_outputs: bool = True
    validation_proportional: bool = True
    tune_baselines: bool = True
    linear_c_values: list = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    crf_reg_grid: list = field(default_factory=lambda: [
        {"c1": 0.1, "c2": 0.1}, {"c1": 1.0, "c2": 1.0}, {"c1": 0.01, "c2": 0.01}])
    crf_parameters: dict | None = None
    bert_model: str = "google-bert/bert-base-uncased"
    bert_epochs: int = 10
    bert_batch_size: int = 4
    bert_lr: float = 2e-5
    bert_device: str = "auto"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def rows_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_bytes(data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def freeze(path, data):
    atomic(path, data)


def save_csv(path, rows):
    flat = [
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            for k, v in r.items()
        }
        for r in rows
    ]
    atomic(path, pd.DataFrame(flat).to_csv(index=False).encode())


def method_id(model, shots=None):
    return (
        "llm_"
        + re.sub(r"[^a-zA-Z0-9_-]+", "_", model["name"])
        + "_"
        + str(model["thinking"]).lower()
        + (f"_{shots}shot" if shots is not None else "")
    )


def shot_counts(s, task):
    return getattr(s, task + "_shots")


def llm_conditions(s):
    return [
        {"model": model, "task": task, "shots": shots, "method": method_id(model, shots)}
        for model in s.models
        for task in s.tasks
        for shots in shot_counts(s, task)
    ]


def method_tasks(s):
    methods = {c["method"]: [BASELINES[c["baseline"]]] for c in learning_curve.conditions(s)}
    for condition in llm_conditions(s):
        methods.setdefault(condition["method"], []).append(condition["task"])
    return methods


def validate(s):
    if (
        not isinstance(s.training_percentages, list)
        or not s.training_percentages
        or any(type(p) is not int or not 1 <= p <= 100 for p in s.training_percentages)
        or len(set(s.training_percentages)) != len(s.training_percentages)
    ):
        raise ValueError(
            "BASELINE_TRAININGSANTEILE: verschiedene ganze Prozentwerte von 1 bis 100."
        )
    if type(s.progress) is not bool:
        raise ValueError("FORTSCHRITT_ANZEIGEN muss True oder False sein.")
    if (
        type(s.repetitions) is not int
        or s.repetitions != 1
        or len(s.fractions) != 3
        or any(not math.isfinite(x) or x < 0 for x in s.fractions)
        or not math.isclose(sum(s.fractions), 1)
        or s.fractions[2] <= 0
    ):
        raise ValueError(
            "Durchläufe müssen positiv sein; Train/Validierung/Eval ergeben 1.0, Eval > 0."
        )
    if (
        not s.tasks
        or set(s.tasks) - {"extraction", "classification"}
        or len(set(s.tasks)) != len(s.tasks)
    ):
        raise ValueError("Mindestens eine Aufgabe auswählen: extraction/classification.")
    if not s.levels or set(s.levels) - {1, 2, 3, 4} or len(set(s.levels)) != len(s.levels):
        raise ValueError("STUFEN muss eine Auswahl aus 1, 2, 3, 4 sein.")
    baselines = s.extraction_baselines + s.classification_baselines
    if len(set(baselines)) != len(baselines):
        raise ValueError("Eine Baseline wurde mehrfach angegeben.")
    for names, task in [
        (s.extraction_baselines, "extraction"),
        (s.classification_baselines, "classification"),
    ]:
        if any(BASELINES.get(name) != task for name in names) or (names and task not in s.tasks):
            raise ValueError(f"Ungültige Baseline-Liste für {task}: {names}")
    for task in ("extraction", "classification"):
        counts = shot_counts(s, task)
        if (
            not isinstance(counts, list)
            or any(type(n) is not int or n not in prompt_templates.SHOT_COUNTS for n in counts)
            or len(counts) != len(set(counts))
        ):
            raise ValueError(
                f"{task}_shots: Liste ohne Duplikate aus 0, 1, 3 verwenden; [] schaltet LLM-Aufträge dieser Aufgabe aus."
            )
    if not (s.models and any(shot_counts(s, task) for task in s.tasks)) and not baselines:
        raise ValueError("Mindestens ein Modell oder eine Baseline auswählen.")
    if any(name != "regex" for name in baselines) and s.fractions[0] <= 0:
        raise ValueError("Trainierbare Baselines benötigen TRAIN_ANTEIL > 0.")
    for model in s.models:
        if (
            set(model) != {"name", "thinking"}
            or not isinstance(model["name"], str)
            or not model["name"].strip()
        ):
            raise ValueError("MODELLE benötigt Einträge mit name und thinking.")
        think = model["thinking"]
        if type(think) is not bool and think not in ("low", "medium", "high"):
            raise ValueError("thinking: True, False, low, medium oder high verwenden.")
        if "gpt-oss" in model["name"].lower() and type(think) is bool:
            raise ValueError(
                "GPT-OSS benötigt thinking=low/medium/high und besitzt keinen Aus-Modus."
            )
    if len({method_id(m) for m in s.models}) != len(s.models):
        raise ValueError("Doppelte Modell-/Thinking-Kombination.")
    if any(
        type(n) is not int or n < 1
        for n in (
            s.num_ctx,
            s.extraction_tokens,
            s.classification_tokens,
            s.thinking_tokens,
            s.timeout,
            s.ner_epochs,
            s.ner_batch_size,
            s.bert_epochs,
            s.bert_batch_size,
        )
    ):
        raise ValueError("Tokenbudgets, Timeout und NER-Parameter müssen positiv sein.")
    if (
        not math.isfinite(s.temperature)
        or s.temperature < 0
        or not math.isfinite(s.ner_lr)
        or s.ner_lr <= 0
        or not math.isfinite(s.bert_lr)
        or s.bert_lr <= 0
    ):
        raise ValueError("Ungültige Temperatur oder Lernrate.")
    if type(s.resplit) is not bool or any(
        type(n) is not int or n < 0 for n in (s.split_seed, s.model_seed, s.llm_seed, *s.method_seeds.values())
    ):
        raise ValueError(
            "Split-Schalter muss True/False sein; Seeds sind nichtnegative ganze Zahlen."
        )
    if s.split_path is not None and s.resplit:
        raise ValueError(
            "SPLIT_DATEI verwendet einen festen Split; SPLIT_PRO_DURCHLAUF_NEU muss False sein."
        )
    for seed in (s.training_subset_seed, s.validation_subset_seed):
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("Teilmengen-Seeds müssen nichtnegative ganze Zahlen sein.")
    if type(s.validation_only) is not bool or (s.validation_only and s.models):
        raise ValueError("Nur-Validierung ist ausschließlich für Baselines vorgesehen.")
    if not s.tune_baselines or not s.compact_outputs:
        raise ValueError("Dieses Paket verwendet validierte Baselines und kompakte Ausgaben.")
    if s.output.resolve() == s.input_path.parent.resolve():
        raise ValueError(
            "Ergebnisse müssen in einen eigenen Ordner außerhalb der Eingabedaten geschrieben werden."
        )


def load_corpus(s):
    source = s.input_path
    data = source.read_bytes()
    if source.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    elif source.suffix.lower() == ".csv":
        import io

        raw = pd.read_csv(io.BytesIO(data), keep_default_na=False, dtype={"case_id": str}).to_dict(
            "records"
        )
    else:
        raise ValueError("TEXT_DATEI muss generated.jsonl oder generated.csv sein.")
    rows = []
    for row in raw:
        required = {
            "report_id",
            "difficulty_level",
            "error_text",
            "ground_truth",
            "true_category",
            "original_index",
            "case_id",
        }
        if required - set(row):
            raise ValueError("Fehlende Textspalten: " + ", ".join(sorted(required - set(row))))
        truth = (
            json.loads(row["ground_truth"])
            if isinstance(row["ground_truth"], str)
            else row["ground_truth"]
        )
        if (
            not isinstance(truth, dict)
            or set(truth) != {"Activity", "Machine", "WorkerID", "Error time and Error date"}
            or any(not isinstance(v, str) or not v.strip() for v in truth.values())
        ):
            raise ValueError("Unvollständige Ground Truth; keine stillschweigende Korrektur.")
        if row["true_category"] not in ERROR_CLASSES or row["difficulty_level"] not in (1, 2, 3, 4):
            raise ValueError("Unbekannte Klasse oder Stufe.")
        if not isinstance(row["error_text"], str) or not row["error_text"].strip():
            raise ValueError(
                "Datensatz enthält Generierungsversuche ohne Text; zuerst prüfen, nicht stillschweigend filtern."
            )
        if not str(row["case_id"]).strip() or not str(row["original_index"]).strip():
            raise ValueError("Case-ID und Ereignisindex fehlen für den gemeinsamen Split.")
        if row["difficulty_level"] in s.levels:
            rows.append(
                {k: row[k] for k in sorted(required) if k != "ground_truth"}
                | {"ground_truth": truth}
            )
    if not rows or len({r["report_id"] for r in rows}) != len(rows):
        raise ValueError("Keine eindeutigen Berichte für die gewählten Stufen.")
    return rows, digest(data)


def group_ids(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        keys = [
            ("case", str(row["case_id"])),
            ("event", str(row["original_index"])),
            ("text", digest(" ".join(row["error_text"].lower().split()).encode())),
        ]
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    return [f"group-{find(i)}" for i in range(len(rows))]


def balanced_holdout(rows, groups, fraction, seed, require_train_classes=False):
    if fraction == 1:
        return [], list(range(len(rows)))
    if len(set(groups)) < 2:
        raise ValueError("Zu wenige unabhängige Fertigungsfälle für den gewählten Split.")
    cats = sorted({r["true_category"] for r in rows})
    levels = sorted({r["difficulty_level"] for r in rows})
    features = np.array(
        [
            [1]
            + [int(r["true_category"] == c) for c in cats]
            + [int(r["difficulty_level"] == v) for v in levels]
            for r in rows
        ]
    )
    total = features.sum(axis=0)
    best = None
    splitter = GroupShuffleSplit(n_splits=256, test_size=fraction, random_state=seed)
    try:
        for train, test in splitter.split(features, groups=groups):
            if require_train_classes and set(cats) - {rows[i]["true_category"] for i in train}:
                continue
            observed = features[test].sum(axis=0)
            loss = float(np.mean(((observed - total * fraction) / np.maximum(total, 1)) ** 2))
            loss += float(4 * ((len(test) / len(rows)) - fraction) ** 2)
            if best is None or loss < best[0]:
                best = loss, train.tolist(), test.tolist()
    except ValueError as exc:
        raise ValueError("Zu wenige unabhängige Fälle für die gewählten Split-Anteile.") from exc
    if best is None:
        raise ValueError(
            "Kein Gruppensplit mit allen Klassen im Training gefunden. Mehr Texte oder größeren Trainingsanteil wählen."
        )
    return best[1], best[2]


def split_rows(rows, s, repetition):
    if s.split_path is not None:
        from .dataset_split import load_fixed

        return load_fixed(rows, s.input_path, s.split_path, s.fractions)
    groups = group_ids(rows)
    seed = s.split_seed + (repetition if s.resplit else 0)
    need_classes = bool(s.classification_baselines)
    dev, test = balanced_holdout(rows, groups, s.fractions[2], seed, need_classes)
    if not dev:
        train, validation = [], []
    elif s.fractions[0] == 0:
        train, validation = [], dev
    elif s.fractions[1] == 0:
        train, validation = dev, []
    else:
        train_idx, val_idx = balanced_holdout(
            [rows[i] for i in dev],
            [groups[i] for i in dev],
            s.fractions[1] / sum(s.fractions[:2]),
            seed + 1000,
            need_classes,
        )
        train, validation = [dev[i] for i in train_idx], [dev[i] for i in val_idx]
    result = {
        name: [rows[i] | {"split_group": groups[i]} for i in indices]
        for name, indices in [("train", train), ("validation", validation), ("eval", test)]
    }
    partitions = [set(r["split_group"] for r in values) for values in result.values()]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Interner Fehler: überlappende Split-Gruppen.")
    return result


def portable_settings(s):
    values = json.loads(json.dumps(asdict(s), default=str))
    package = ROOT.parent.resolve()
    for key in ("input_path", "output", "split_path", "ner_model", "bert_model"):
        if values.get(key) is not None:
            path = Path(values[key])
            if path.is_absolute():
                try:
                    values[key] = path.resolve().relative_to(package).as_posix()
                except ValueError:
                    raise ValueError(f"{key} must be inside the submission package.")
    return values


def prepare(s):
    validate(s)
    rows, source_sha = load_corpus(s)
    snapshots = {path.name: path for path in [*PROMPTS.glob("*.txt"), *PROMPTS.glob("*.json")]}
    fixed_split = split_rows(rows, s, 0) if s.split_path is not None else None
    split_manifest = s.split_path.read_bytes() if s.split_path is not None else None
    config = {
        "version": VERSION,
        "settings": portable_settings(s),
        "input_sha256": source_sha,
        "reports": len(rows),
        "package_versions": {
            name: package_version(name)
            for name in ["pandas", "numpy", "scikit-learn", "ollama"]
            + (
                ["torch", "transformers"]
                if any(x.startswith(("ner", "bert_ner")) for x in s.extraction_baselines) or "bert" in s.classification_baselines
                else []
            )
            + (
                ["python-crfsuite"]
                if any(x.startswith("crf") for x in s.extraction_baselines)
                else []
            )
        },
        "protocol_sha256": {n: digest(p.read_bytes()) for n, p in snapshots.items()},
        "scoring": "Strict reference extraction. Invalid output fails all four fields. Classification retains Unknown. No harness filtering.",
        "split_policy": (
            "Saved fixed membership. Exact class-level balance in evaluation/validation; intact case/event/text groups. Identical split across methods, shots and repetitions."
            if fixed_split is not None
            else "Grouped holdout: cases, source events and identical texts stay together. Approximate marginal class/rung balance. Validation never enters fitting."
        ),
        "split_manifest_sha256": digest(split_manifest) if split_manifest is not None else None,
        "llm_protocol": "Zero/one/three fixed examples per selected task. One-shot uses one fixed example; three-shot uses three. No in-corpus training/validation/evaluation examples. Separate stateless requests per condition.",
        "llm_conditions": llm_conditions(s),
        "baseline_conditions": learning_curve.conditions(s),
        "selection_protocol": "Fixed nested proportional validation subsets; fit only training. Paired nested training draws across methods. Linear C by validation macro-F1; CRF and NER by strict extraction accuracy then mean attribute accuracy; BERT by macro-F1 then loss. Exact ties retain earlier candidate/epoch. Within each fit, checkpoint and hyperparameter selection use validation data.",
        "prediction_partition": "validation" if s.validation_only else "eval",
        "retry_policy": "API failures remain in the append-only prediction journal and leave the job pending for the next start. Only returned answers enter model scores and regular timing; invalid answers are scored without retry. Failed-attempt timing is exported separately.",
        "timing_protocol": timing.VERSION
        + "; server-reported durations separately from client latency; no inferred network or thinking-only duration.",
    }
    freeze(s.output / "config.json", payload(config))
    if split_manifest is not None:
        freeze(s.output / "split_manifest.json", split_manifest)
    for name, path in snapshots.items():
        freeze(s.output / "prompts" / name, path.read_bytes())
    freeze(s.output / "corpus.jsonl", rows_bytes(rows))
    templates = {
        prompt_templates.prompt_key(task, shots): template
        for shots in prompt_templates.SHOT_COUNTS
        for task, template in prompt_templates.load(s.output / "prompts", shots).items()
        if task in s.tasks and shots in shot_counts(s, task)
    }
    if any(extraction.validate_prediction(row["ground_truth"]) for row in rows):
        raise ValueError("Ground Truth entspricht nicht dem Extraktionsschema.")
    prompts = [
        {
            "report_id": row["report_id"],
            **{
                task: prompt_templates.render(template, row["error_text"])
                for task, template in templates.items()
            },
        }
        for row in rows
    ]
    freeze(s.output / "prompts.jsonl", rows_bytes(prompts))
    for task in s.tasks:
        print(
            f"LLM {task}: Shots {shot_counts(s, task)}; {len(s.models)} Modellbedingungen.",
            flush=True,
        )
    for repetition in range(s.repetitions):
        split = fixed_split if fixed_split is not None else split_rows(rows, s, repetition)
        folder = s.output / "records"
        for name, subset in split.items():
            freeze(folder / (name + ".jsonl"), rows_bytes(subset))
            if not s.compact_outputs:
                save_csv(folder / (name + ".csv"), subset)
        if any(c["training_percent"] is not None for c in learning_curve.conditions(s)):
            subsets = learning_curve.prepare(
                folder / "training_subsets",
                split["train"],
                s.training_percentages,
                s.training_subset_seed if s.training_subset_seed is not None
                else s.split_seed + (repetition if s.resplit else 0),
            )
            validation_subsets = learning_curve.prepare_validation(
                folder / "validation_subsets", split["validation"], s.training_percentages,
                s.validation_subset_seed if s.validation_subset_seed is not None
                else s.split_seed + (repetition if s.resplit else 0), s.validation_proportional)
            print("Baseline-Validierung: " + ", ".join(f"{p}% = {len(rows)} Texte" for p, rows in validation_subsets.items()), flush=True)
            print(
                "Baseline-Training: "
                + ", ".join(f"{p}% = {len(rows)} Texte" for p, rows in subsets.items()),
                flush=True,
            )
        stats = {
            name: {
                "n": len(subset),
                "groups": len({r["split_group"] for r in subset}),
                "classes": dict(Counter(r["true_category"] for r in subset)),
                "rungs": dict(Counter(r["difficulty_level"] for r in subset)),
                "class_level": {
                    category: {
                        str(level): sum(
                            r["true_category"] == category and r["difficulty_level"] == level
                            for r in subset
                        )
                        for level in s.levels
                    }
                    for category in ERROR_CLASSES
                },
            }
            for name, subset in split.items()
        }
        freeze(folder / "split_summary.json", payload(stats))
        print(
            f"Durchlauf {repetition + 1}: Train {len(split['train'])}, Validierung {len(split['validation'])}, Eval {len(split['eval'])}",
            flush=True,
        )
    return {r["report_id"]: r for r in prompts}


def score_result(source, task, raw):
    if task == "classification":
        accepted = tuple(dict.fromkeys(CLASSIFICATION_LABELS + ERROR_CLASSES))
        category = classifier.parse_category(raw, accepted)
        predicted = classifier.ALIASES.get(category, category)
        return {
            "prediction": predicted,
            "parse_ok": predicted != "Unknown",
            "correct": predicted == source["true_category"],
        }
    prediction, failures, valid, correct, error = extraction.score(raw, source["ground_truth"])
    return {
        "prediction": prediction,
        "parse_ok": valid,
        "validation_failures": failures,
        "parse_error": error,
        "field_correct": correct,
        "correct": all(correct.values()),
    }


def baseline_json(prediction):
    return json.dumps({k: None if v == "Not specified" else v for k, v in prediction.items()})


def prediction_history(path, eval_rows, methods, tasks):
    records = read_rows(path)
    completed = prediction_records.completed(records)
    sources = {r["report_id"]: r for r in eval_rows}
    for rec in records:
        source = sources.get(rec["report_id"])
        if (
            source is None
            or rec["method"] not in methods
            or rec["task"] not in tasks
            or (isinstance(methods, dict) and rec["task"] not in methods.get(rec["method"], []))
            or rec["text_sha256"] != digest(source["error_text"].encode())
        ):
            raise ValueError("Gespeicherte Vorhersagen passen nicht zu diesem Experiment.")
        expected = score_result(source, rec["task"], rec["raw_output"])
        if any(rec[k] != v for k, v in expected.items()):
            raise ValueError("Bewertung einer gespeicherten Vorhersage wurde verändert.")
    return completed, {prediction_records.key(r) for r in completed}


def append_result(path, source, method, task, raw, latency=0, **meta):
    if isinstance(meta.get("raw_response"), dict):
        meta["raw_response"] = {k: v for k, v in meta["raw_response"].items() if k != "message"}
    if not method.startswith("llm_"):
        meta.update(component_timing.component(latency))
    rec = {
        "attempt_id": uuid.uuid4().hex,
        "report_id": source["report_id"],
        "difficulty_level": source["difficulty_level"],
        "true_category": source["true_category"],
        "method": method,
        "task": task,
        "text_sha256": digest(source["error_text"].encode()),
        "raw_output": raw,
        "latency_sec": latency,
        "api_error": "",
        **meta,
        "included_in_benchmark": not bool(meta.get("api_error")),
        **score_result(source, task, raw),
    }
    with path.open("a") as handle:
        handle.write(rows_bytes([rec]).decode())
        handle.flush()
    return rec


def run_baselines(s, folder, train, validation, eval_rows, done, repetition):
    history = folder / ("validation_predictions.jsonl" if s.validation_only else "predictions.jsonl")
    conditions = learning_curve.conditions(s)
    for index, condition in enumerate(conditions, 1):
        method, baseline = condition["method"], condition["baseline"]
        seed = s.method_seeds.get(method, s.method_seeds.get(baseline, s.model_seed)) + repetition
        task = BASELINES[baseline]
        percent = condition["training_percent"]
        selected_train = (
            read_rows(folder / "training_subsets" / f"train_{percent:03d}.jsonl")
            if percent is not None
            else []
        )
        if percent is not None and not selected_train:
            raise ValueError(f"Gespeicherte Trainingsmenge fehlt: {method}")
        selected_validation = (
            read_rows(folder / "validation_subsets" / f"validation_{percent:03d}.jsonl")
            if percent is not None else [])
        if percent is not None:
            from .selection import validate_partitions
            validate_partitions(selected_train, selected_validation)
        target_rows = (selected_validation if percent is not None else validation) if s.validation_only else eval_rows
        pending = [r for r in target_rows if (method, task, r["report_id"]) not in done]
        if not pending:
            continue
        meta_training = {
            "prediction_partition": "validation" if s.validation_only else "eval",
            "baseline": baseline,
            "training_percent": percent,
            "training_n": len(selected_train) if percent is not None else None,
            "validation_percent": percent if s.validation_proportional and percent is not None else (100 if percent is not None else None),
            "validation_n": len(selected_validation) if percent is not None else None,
        }
        progress.message(
            f"Lauf {repetition + 1}/{s.repetitions}, Baseline {index}/{len(conditions)}: "
            f"{method}; {len(pending)}/{len(target_rows)} Berichte offen."
        )
        desc = f"{method} · {task}"
        initial = len(target_rows) - len(pending)
        method_dir = folder / method
        method_dir.mkdir(exist_ok=True)
        if baseline == "regex":
            from . import regex

            for row in progress.iterate(
                pending, s, desc=desc, total=len(target_rows), initial=initial
            ):
                start = time.perf_counter()
                raw = regex.extract_information_with_regex_baseline(row["error_text"])
                prediction = json.loads(regex.extract_json_block(raw))
                append_result(
                    history,
                    row,
                    method,
                    task,
                    baseline_json(prediction),
                    time.perf_counter() - start,
                    raw_baseline_output=raw,
                    **meta_training,
                )
        elif baseline in {"ner", "crf", "ner_time", "crf_time", "bert_ner", "bert_ner_time"}:
            if baseline.startswith(("ner", "bert_ner")):
                from .ner import predict
            else:
                from .selection import predict_crf as predict
            base_dir = folder / condition["fit_method"]
            base_dir.mkdir(exist_ok=True)
            predictions = predict(
                replace(s, ner_model=s.bert_model) if baseline.startswith("bert_ner") else s,
                base_dir, selected_train, selected_validation, pending, seed
            )
            for row, pred, latency in progress.iterate(
                predictions, s, desc=desc, total=len(target_rows), initial=initial
            ):
                meta = dict(meta_training)
                if baseline.endswith("_time"):
                    from .time_rules import resolve

                    started = time.perf_counter()
                    pred, trace = resolve(row["error_text"], pred)
                    latency += time.perf_counter() - started
                    meta["temporal_resolution"] = trace
                append_result(history, row, method, task, baseline_json(pred), latency, **meta)
        elif baseline == "bert":
            from .bert_classifier import predict
            for row, label, latency in predict(s, method_dir, selected_train, selected_validation, pending, seed):
                append_result(history, row, method, task, json.dumps({"Category": label}), latency, **meta_training)
        else:
            from .selection import fit_linear
            model = fit_linear(s, method_dir, baseline, selected_train, selected_validation, seed)
            for row in progress.iterate(pending, s, desc=desc, total=len(target_rows), initial=initial):
                start = time.perf_counter()
                label = model.predict([row["error_text"]])[0]
                append_result(history, row, method, task, json.dumps({"Category": label}), time.perf_counter()-start, **meta_training)
        atomic(folder / "progress.json", payload({"completed_condition": method,
            "condition_index": index, "conditions_total": len(conditions),
            "completed_at_unix": time.time()}))


from .ollama_client import benchmark_session as model_session


def preflight(client, models):
    installed = {m.model: m.digest for m in client.list().models}
    details = {}
    for model in models:
        name = model["name"]
        if name not in installed:
            raise ValueError(
                f"Ollama-Modell nicht installiert oder Name nicht exakt: {name}. Verfügbar: {sorted(installed)}"
            )
        if name not in details:
            show = client.show(name).model_dump(mode="json")
            details[name] = {
                "digest": installed[name],
                "capabilities": show.get("capabilities") or [],
                "details": show.get("details"),
                "modelinfo": show.get("modelinfo"),
            }
        if model["thinking"] is not False and "thinking" not in details[name]["capabilities"]:
            raise ValueError(
                f"{name} meldet keine Thinking-Unterstützung. Keine automatische Änderung der Einstellung."
            )
        architecture = (details[name].get("modelinfo") or {}).get("general.architecture", "")
        if architecture in ("gptoss", "gpt-oss") and type(model["thinking"]) is bool:
            raise ValueError(f"{name} basiert auf GPT-OSS: thinking muss low/medium/high sein.")
    return details


def run_models(s, prompts, client_factory=None):
    conditions = llm_conditions(s)
    if not conditions:
        return
    if client_factory is None:
        from ollama import Client

        client_factory = Client
    with lifecycle.host_lock(s.host):
        client = client_factory(host=s.host, timeout=s.timeout)
        details = preflight(client, s.models)
        freeze(s.output / "models.json", payload(details))
        allowed = lifecycle.BENCHMARK_MODELS | {m["name"] for m in s.models}
        for name in dict.fromkeys(m["name"] for m in s.models):
            jobs = []
            for repetition in range(s.repetitions):
                folder = s.output / "records"
                eval_rows = read_rows(folder / "eval.jsonl")
                methods = method_tasks(s)
                _, done = prediction_history(
                    folder / "predictions.jsonl", eval_rows, methods, s.tasks
                )
                for condition in (c for c in conditions if c["model"]["name"] == name):
                    for row in eval_rows:
                        if (condition["method"], condition["task"], row["report_id"]) not in done:
                            jobs.append((repetition, folder, condition, row))
            if not jobs:
                continue
            audit = s.output / "ollama" / (uuid.uuid4().hex + ".json")
            with model_session(client, name, allowed, audit, s.host):
                for repetition, folder, condition, row in progress.iterate(
                    jobs,
                    s,
                    desc=f"LLM {name}",
                    describe=lambda job: (
                        f"Lauf {job[0] + 1}/{s.repetitions} · {job[2]['task']} · {job[2]['shots']}-shot · Thinking={job[2]['model']['thinking']}"
                    ),
                ):
                    local_start = time.perf_counter()
                    model, task, shots = condition["model"], condition["task"], condition["shots"]
                    schema = (
                        extraction.EXTRACTION_SCHEMA
                        if task == "extraction"
                        else {
                            "type": "object",
                            "properties": {
                                "Category": {"type": "string", "enum": list(CLASSIFICATION_LABELS)}
                            },
                            "required": ["Category"],
                            "additionalProperties": False,
                        }
                    )
                    tokens = (
                        s.extraction_tokens if task == "extraction" else s.classification_tokens
                    )
                    if model["thinking"] is not False:
                        tokens = s.thinking_tokens
                    options = {
                        "temperature": s.temperature,
                        "seed": s.llm_seed + repetition,
                        "num_ctx": s.num_ctx,
                        "num_predict": tokens,
                    }
                    prompt = prompts[row["report_id"]][prompt_templates.prompt_key(task, shots)]
                    request_started_at = datetime.now(timezone.utc).isoformat()
                    local_before = time.perf_counter() - local_start
                    start = time.perf_counter()
                    raw, response_meta, error, interrupted = "", {}, "", False
                    try:
                        response = client.chat(
                            model=name,
                            messages=[{"role": "user", "content": prompt}],
                            think=model["thinking"],
                            options=options,
                            format=schema,
                            stream=False,
                            keep_alive="10m",
                        )
                    except (Exception, KeyboardInterrupt) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        interrupted = isinstance(exc, KeyboardInterrupt)
                    wall_time = time.perf_counter() - start
                    post_start = time.perf_counter()
                    if not error:
                        raw = response.message.content.strip()
                        response_meta = response.model_dump(mode="json")
                    request_finished_at = datetime.now(timezone.utc).isoformat()
                    local_sec = local_before + time.perf_counter() - post_start
                    append_result(
                        folder / "predictions.jsonl",
                        row,
                        condition["method"],
                        task,
                        raw,
                        wall_time,
                        model=name,
                        model_digest=details[name]["digest"],
                        thinking=model["thinking"],
                        shots=shots,
                        options=options,
                        prompt_sha256=digest(prompt.encode()),
                        raw_response=response_meta,
                        api_error=error,
                        request_started_at=request_started_at,
                        request_finished_at=request_finished_at,
                        **(
                            timing.extract(response_meta, wall_time)
                            | component_timing.request_timing(local_sec, response_meta, wall_time)
                        ),
                    )
                    if error:
                        if interrupted:
                            raise KeyboardInterrupt
                        raise RuntimeError(
                            f"API-Fehlversuch gespeichert; Auftrag bleibt offen: {error}. "
                            "Verbindung wiederherstellen und erneut starten. Dieselbe Anfrage wird "
                            "mit unverändertem Prompt und Seed wiederholt; der Fehlversuch zählt nicht als Modellfehler."
                        )


def report(s):
    summaries, all_details = [], {}
    methods = method_tasks(s)
    method_shots = {c["method"]: c["shots"] for c in llm_conditions(s)}
    baseline_conditions = {c["method"]: c for c in learning_curve.conditions(s)}
    for repetition in range(s.repetitions):
        folder = s.output / "records"
        eval_rows = read_rows(folder / "eval.jsonl")
        records, done = prediction_history(
            folder / "predictions.jsonl", eval_rows, methods, s.tasks
        )
        failed_attempts = [
            r | {"retry_completed": prediction_records.key(r) in done}
            for r in read_rows(folder / "predictions.jsonl")
            if r.get("api_error")
        ]
        if not s.compact_outputs:
            atomic(folder / "failed_attempts.jsonl", rows_bytes(failed_attempts))
            save_csv(folder / "failed_attempts.csv", failed_attempts)
            save_csv(folder / "predictions.csv", [timing.enrich(r) for r in records])
        subset_manifest = folder / "training_subsets" / "manifest.json"
        counts = (
            {
                r["training_percent"]: r["training_n"]
                for r in json.loads(subset_manifest.read_text())["subsets"]
            }
            if subset_manifest.exists()
            else {}
        )
        val_manifest = folder / "validation_subsets/manifest.json"
        val_counts = {r["training_percent"]: r["validation_n"] for r in json.loads(val_manifest.read_text())["subsets"]} if val_manifest.exists() else {}
        for method, tasks in methods.items():
            for task in tasks:
                values = [r for r in records if r["method"] == method and r["task"] == task]
                for level in ["all"] + s.levels:
                    subset = [r for r in values if level == "all" or r["difficulty_level"] == level]
                    expected = sum(
                        level == "all" or r["difficulty_level"] == level for r in eval_rows
                    )
                    failures = [
                        r
                        for r in failed_attempts
                        if r["method"] == method
                        and r["task"] == task
                        and (level == "all" or r["difficulty_level"] == level)
                    ]
                    item = {
                        "run": repetition + 1,
                        "method": method,
                        "task": task,
                        "shots": method_shots.get(method),
                        "baseline": baseline_conditions.get(method, {}).get("baseline"),
                        "training_percent": baseline_conditions.get(method, {}).get(
                            "training_percent"
                        ),
                        "training_n": counts.get(
                            baseline_conditions.get(method, {}).get("training_percent")
                        ),
                        "validation_n": val_counts.get(baseline_conditions.get(method, {}).get("training_percent")),
                        "validation_percent": (baseline_conditions.get(method, {}).get("training_percent") if s.validation_proportional else 100) if method in baseline_conditions and baseline_conditions[method]["training_percent"] is not None else None,
                        "rung": level,
                        "n_expected": expected,
                        "n_scored": len(subset),
                        "n_pending": expected - len(subset),
                        "complete": len(subset) == expected,
                        "accuracy": sum(r["correct"] for r in subset) / len(subset)
                        if subset
                        else None,
                        "invalid": sum(not r["parse_ok"] for r in subset),
                        "api_errors": len(failures),
                        "api_failed_reports_pending": len(
                            {r["report_id"] for r in failures if not r["retry_completed"]}
                        ),
                        "mean_latency_sec": np.mean([r["latency_sec"] for r in subset]).item()
                        if subset
                        else None,
                        **timing.summarize(subset),
                    }
                    if task == "extraction":
                        item.update(
                            {
                                f"accuracy_{f}": sum(r["field_correct"][f] for r in subset)
                                / len(subset)
                                if subset
                                else None
                                for f in extraction.FIELDS
                            }
                        )
                    elif subset:
                        labels = sorted({r["true_category"] for r in subset})
                        truth, pred = (
                            [r["true_category"] for r in subset],
                            [r["prediction"] for r in subset],
                        )
                        details = classification_report(
                            truth, pred, labels=labels, output_dict=True, zero_division=0
                        )
                        item["macro_f1"] = details["macro avg"]["f1-score"]
                        fixed = classification_report(
                            truth,
                            pred,
                            labels=list(ERROR_CLASSES),
                            output_dict=True,
                            zero_division=0,
                        )
                        item["macro_f1_all_classes"] = fixed["macro avg"]["f1-score"]
                        choices = list(ERROR_CLASSES) + ["Other", "Unknown"]
                        all_details[f"{repetition + 1}/{method}/{level}"] = {
                            "complete": item["complete"],
                            "classification_report": details,
                            "labels": choices,
                            "confusion_matrix": confusion_matrix(
                                truth, pred, labels=choices
                            ).tolist(),
                        }
                    summaries.append(item)
    atomic(
        s.output / "metrics.json",
        payload({"results": summaries, "classification_details": all_details}),
    )
    if all_details and not s.compact_outputs:
        from .classification_outputs import export

        export(s.output, all_details, s.repetitions, s)
    aggregates = []
    frame = pd.DataFrame(summaries)
    metric_names = [
        c
        for c in frame
        if c.startswith("accuracy")
        or c
        in (
            "macro_f1",
            "macro_f1_all_classes",
            "mean_latency_sec",
            "mean_ollama_inference_sec",
            "mean_ollama_total_sec",
            "mean_ollama_load_sec",
            "generation_tokens_per_sec",
        )
    ]
    for (method, task, rung), group in frame.groupby(["method", "task", "rung"], sort=False):
        finished = group.loc[group.complete & group.n_scored.gt(0)]
        item = {
            "method": method,
            "task": task,
            "shots": method_shots.get(method),
            "baseline": baseline_conditions.get(method, {}).get("baseline"),
            "training_percent": baseline_conditions.get(method, {}).get("training_percent"),
            "rung": rung,
            "completed_runs": len(finished),
            "requested_runs": s.repetitions,
        }
        for metric in metric_names:
            values = finished[metric].dropna().astype(float)
            if len(values):
                item[metric + "_mean"] = float(values.mean())
                item[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        aggregates.append(item)
    if not s.compact_outputs:
        save_csv(s.output / "aggregate.csv", aggregates)
        if any(s.output.glob("records/predictions.jsonl")):
            timing.export(s.output)
    return summaries


def run(s, plan_only=False, client_factory=None):
    validate(s)
    s.output.mkdir(parents=True, exist_ok=True)
    prompts = prepare(s)
    print(f"Ergebnisse: {s.output}", flush=True)
    if plan_only:
        print("Nur geplant: kein Training und keine Modellaufrufe.")
        return
    try:
        for repetition in range(s.repetitions):
            folder = s.output / "records"
            eval_rows = [] if s.validation_only else read_rows(folder / "eval.jsonl")
            validation = read_rows(folder / "validation.jsonl")
            methods = method_tasks(s)
            _, done = prediction_history(
                folder / ("validation_predictions.jsonl" if s.validation_only else "predictions.jsonl"),
                validation if s.validation_only else eval_rows, methods, s.tasks
            )
            run_baselines(
                s,
                folder,
                read_rows(folder / "train.jsonl"),
                validation,
                eval_rows,
                done,
                repetition,
            )
        if not s.validation_only:
            run_models(s, prompts, client_factory)
    finally:
        if s.validation_only:
            from .series import validation_report
            validation_report(s)
        else:
            report(s)
    print(
        "Validierungsprüfung abgeschlossen; keine Evaluationsvorhersagen."
        if s.validation_only else "Benchmark abgeschlossen. Ergebnisse: summary.csv."
        if s.compact_outputs else "Benchmark abgeschlossen. Vergleich: aggregate.csv; einzelne Durchläufe: summary.csv.",
        flush=True,
    )
