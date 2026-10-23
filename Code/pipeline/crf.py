from __future__ import annotations

import json
import re
import time

from . import ner_utils as ner
from . import normalization as common

VERSION = "crf-extraction-1"
PARAMETERS = {"c1": 0.1, "c2": 0.1, "max_iterations": 100, "feature.possible_transitions": True}


def tokenize(text):
    return [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text)
    ]


def features(tokens):
    def describe(word):
        lower = word.lower()
        lexical = (
            "<number>" if word.isdigit() or lower in common.NUMBERS or lower == "oh" else lower
        )
        shape = re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", re.sub(r"\d", "d", word)))
        return {
            "word": lexical,
            "suffix2": lexical[-2:],
            "suffix3": lexical[-3:],
            "shape": re.sub(r"(.)\1+", r"\1", shape),
            "title": word.istitle(),
            "upper": word.isupper(),
            "digit": word.isdigit(),
        }

    descriptors = [describe(token[0]) for token in tokens]
    result = []
    for i in range(len(tokens)):
        values = {"bias": 1.0, "BOS": i == 0, "EOS": i == len(tokens) - 1}
        for delta in (-2, -1, 0, 1, 2):
            if 0 <= i + delta < len(tokens):
                values.update(
                    {f"{delta}:{key}": value for key, value in descriptors[i + delta].items()}
                )
        result.append(values)
    return result


def sequence_labels(tokens, spans, label):
    selected = [s for s in spans if s["label"] == label]
    labels, previous = [], None
    for _, start, end in tokens:
        index = next(
            (i for i, s in enumerate(selected) if start < s["end"] and end > s["start"]), None
        )
        labels.append("O" if index is None else ("I" if index == previous else "B"))
        previous = index
    return labels


def diagnostic_scores(output, trained, annotated, pycrfsuite):
    result = {split: {} for split in annotated}
    for name in trained:
        label = name.removesuffix(".crfsuite")
        tagger = pycrfsuite.Tagger()
        try:
            tagger.open(str(output / name))
            for split, records in annotated.items():
                tp = fp = fn = count = 0
                for record in records:
                    if record["annotation_status"][label] != "resolved":
                        continue
                    count += 1
                    tokens = tokenize(record["text"])
                    ys = tagger.tag(features(tokens))
                    ids = [ner.LABEL2ID["O" if y == "O" else y + "-" + label] for y in ys]
                    predicted = ner.decode_entities(
                        record["text"], [(a, b) for _, a, b in tokens], ids, [1.0] * len(ids)
                    )
                    p = {(s["start"], s["end"]) for s in predicted}
                    gold = {(s["start"], s["end"]) for s in record["spans"] if s["label"] == label}
                    tp += len(p & gold)
                    fp += len(p - gold)
                    fn += len(gold - p)
                result[split][label] = {
                    "resolved_reports": count,
                    "true_positive_spans": tp,
                    "false_positive_spans": fp,
                    "false_negative_spans": fn,
                    "span_f1": 2 * tp / (2 * tp + fp + fn)
                    if count and (2 * tp + fp + fn)
                    else None,
                }
        finally:
            tagger.close()
    return result


def predict(s, output, train, validation, pending, seed):
    from .benchmark import atomic, digest, freeze, payload

    try:
        import pycrfsuite
    except ImportError as exc:
        raise RuntimeError("CRF fehlt: python -m pip install -r requirements.txt") from exc
    parameters = s.crf_parameters or PARAMETERS
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "version": VERSION,
        "surface_version": common.VERSION,
        "parameters": parameters,
        "train_sha256": digest(payload(train)),
        "validation_sha256": digest(payload(validation)),
    }
    freeze(output / "fit_config.json", payload(signature))
    finished = output / "training.json"
    if not finished.exists():
        start = time.perf_counter()
        annotated = {"train": common.annotate(train), "validation": common.annotate(validation)}
        for name, records in annotated.items():
            freeze(output / (name + "_annotations.json"), payload(records))
        trained, used_ids = {}, {}
        from . import progress

        for label in progress.iterate(
            common.LABELS, s, desc=f"CRF Training ({len(train)} Texte)", unit="Attribut"
        ):
            records = [r for r in annotated["train"] if r["annotation_status"][label] == "resolved"]
            used_ids[label] = [r["report_id"] for r in records]
            if not records:
                continue
            trainer = pycrfsuite.Trainer(algorithm="lbfgs", verbose=False)
            trainer.set_params(parameters)
            for record in records:
                tokens = tokenize(record["text"])
                trainer.append(features(tokens), sequence_labels(tokens, record["spans"], label))
            path = output / (label + ".crfsuite")
            trainer.train(str(path))
            trained[path.name] = digest(path.read_bytes())
        if not trained:
            raise ValueError("Keine auflösbaren CRF-Trainingsspannen; Annotationen prüfen.")
        diagnostics = diagnostic_scores(output, trained, annotated, pycrfsuite)
        atomic(
            finished,
            payload(
                {
                    "version": VERSION,
                    "train_ids": [r["report_id"] for r in train],
                    "used_train_ids_by_entity": used_ids,
                    "seed": seed,
                    "parameters": parameters,
                    "fit_seconds": time.perf_counter() - start,
                    "selection": "Fixed L1/L2 regularization; no hyperparameter search or validation fitting.",
                    "annotation_coverage": {k: common.coverage(v) for k, v in annotated.items()},
                    "weak_span_diagnostics": diagnostics,
                    "unresolved_policy": "One CRF per type; omit unresolved type/report pairs from fitting only.",
                    "checkpoint_sha256": trained,
                }
            ),
        )
        for split, scores in diagnostics.items():
            progress.message(
                f"CRF {split}: "
                + ", ".join(
                    f"{label} F1={info['span_f1']:.3f}"
                    for label, info in scores.items()
                    if info["span_f1"] is not None
                ),
            )
    saved = json.loads(finished.read_text())
    taggers = {}
    try:
        for name, sha in saved["checkpoint_sha256"].items():
            path = output / name
            if digest(path.read_bytes()) != sha:
                raise ValueError("Gespeicherter CRF-Checkpoint wurde verändert.")
            tagger = pycrfsuite.Tagger()
            tagger.open(str(path))
            taggers[path.stem] = tagger
        for row in pending:
            start = time.perf_counter()
            text = row["error_text"]
            tokens = tokenize(text)
            x = features(tokens)
            entities = []
            if tokens:
                for label, tagger in taggers.items():
                    ys = tagger.tag(x)
                    ids = [ner.LABEL2ID["O" if y == "O" else y + "-" + label] for y in ys]
                    confidence = [tagger.marginal(y, i) for i, y in enumerate(ys)]
                    entities.extend(
                        ner.decode_entities(text, [(a, b) for _, a, b in tokens], ids, confidence)
                    )
            pred = common.normalize_prediction(entities)
            yield row, pred, time.perf_counter() - start
    finally:
        for tagger in taggers.values():
            tagger.close()
