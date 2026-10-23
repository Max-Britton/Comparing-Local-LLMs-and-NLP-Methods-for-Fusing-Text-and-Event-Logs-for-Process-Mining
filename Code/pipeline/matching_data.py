import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import learning_curve, matching, prediction_records, active_results
from . import request_timing as timing
from .matching_console import pass_rows, print_pass_summary

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows):
    keyed = {row["report_id"]: row for row in rows}
    if len(rows) != len(keyed):
        raise ValueError("Doppelte Report-IDs in Benchmark-Eingaben.")
    return keyed


def write_json(path, value):
    timing.atomic(Path(path), timing.as_json(value))


def write_jsonl(path, rows):
    timing.atomic(
        Path(path),
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
    )


def resolve_source(config, extraction_source, extraction_thinking=False, extraction_shots=None):
    if not isinstance(extraction_source, str) or not extraction_source:
        raise ValueError(
            "EXTRAKTIONS_QUELLE muss ein Modellname oder eine Extraktionsbaseline sein."
        )
    settings = config["settings"]
    baseline = next(
        (
            c
            for c in learning_curve.conditions(settings)
            if c["method"] == extraction_source
            and c["baseline"] in settings.get("extraction_baselines", [])
        ),
        None,
    )
    if baseline is not None:
        if extraction_shots is not None:
            raise ValueError("Extraktionsbaselines haben keine Shot-Varianten.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", extraction_source):
            raise ValueError("Ungültiger Methodenname der Extraktionsbaseline.")
        neural = baseline['baseline'] in {'ner', 'ner_time', 'bert_ner', 'bert_ner_time'}
        model = settings.get('bert_model' if baseline['baseline'].startswith('bert_ner') else 'ner_model') if neural else None
        if neural and (
            not isinstance(model, str) or not model
        ):
            raise ValueError("Das NER-Basismodell fehlt in der Benchmark-Konfiguration.")
        return {
            "kind": "stored_extraction_baseline",
            "method": extraction_source,
            "model": model,
            "thinking": None,
            "shots": None,
            "training_percent": baseline["training_percent"],
            "baseline": baseline["baseline"],
        }
    if type(extraction_thinking) is not bool and extraction_thinking not in (
        "low",
        "medium",
        "high",
    ):
        raise ValueError("EXTRAKTION_THINKING: False, True, low, medium oder high verwenden.")
    for model in settings.get("models", []):
        if (
            model["name"] == extraction_source
            and model.get("thinking") == extraction_thinking
            and type(model.get("thinking")) is type(extraction_thinking)
        ):
            available_shots = settings.get("extraction_shots", [1])
            if "extraction" not in settings.get("tasks", ["extraction"]):
                available_shots = []
            shots = extraction_shots
            if shots is None:
                if len(available_shots) != 1:
                    raise ValueError(
                        f"Extraktions-Shots ausdrücklich auswählen. Verfügbar: {available_shots}."
                    )
                shots = available_shots[0]
            if type(shots) is not int or shots not in (0, 1, 3) or shots not in available_shots:
                raise ValueError(
                    f"Extraktions-Shots {shots!r} fehlen im Quellbenchmark: {available_shots}."
                )
            method = (
                "llm_"
                + re.sub(r"[^a-zA-Z0-9_-]+", "_", extraction_source)
                + "_"
                + str(extraction_thinking).lower()
            )
            if "extraction_shots" in settings:
                method += f"_{shots}shot"
            return {
                "kind": "stored_llm_extraction",
                "method": method,
                "model": extraction_source,
                "thinking": extraction_thinking,
                "shots": shots,
            }
    available = [
        c["method"]
        for c in learning_curve.conditions(settings)
        if c["baseline"] in settings.get("extraction_baselines", [])
    ] + [f"{m['name']} (Thinking={m.get('thinking')})" for m in settings.get("models", [])]
    raise ValueError(
        f"Extraktionsquelle {extraction_source!r} mit Thinking={extraction_thinking} "
        f"ist in diesem Benchmark nicht konfiguriert. Verfügbar: {', '.join(available) or 'keine'}."
    )


def source_info(benchmark, extraction_source, extraction_thinking=False, extraction_shots=None):
    config = active_results.config(benchmark)
    return resolve_source(config, extraction_source, extraction_thinking, extraction_shots)


def selected_predictions(benchmark, repetition, spec, history=None):
    history = active_results.predictions(benchmark, repetition) if history is None else history
    tasks = ('extraction', 'classification') if spec['kind']=='stored_llm_extraction' else ('extraction',)
    return [p for p in prediction_records.completed(history)
            if p.get('method')==spec['method'] and p.get('task') in tasks]


def selected_sha(rows):
    return sha(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows).encode())


def prepare_inputs(
    benchmark,
    repetition,
    eventlog,
    output,
    *,
    extraction_source="ner_time",
    extraction_thinking=False,
    extraction_shots=None,
    matching_model="gemma4:31b",
    embedding_model="mxbai-embed-large",
    host="http://127.0.0.1:11435",
    settings=matching.Settings(),
    order_seeds=(None,),
    use_original_report=True,
    history=None,
):
    if type(use_original_report) is not bool:
        raise ValueError("MIT_ORIGINALBERICHT muss True oder False sein.")
    settings.validate()
    benchmark, output = Path(benchmark), Path(output)
    if (
        repetition < 1
        or not order_seeds
        or any(s is not None and type(s) is not int for s in order_seeds)
    ):
        raise ValueError("Benchmark-Durchlauf/Reihenfolgen ungültig.")
    if len(set(order_seeds)) != len(order_seeds):
        raise ValueError("Reihenfolgen-Seeds müssen verschieden sein.")
    if (output / "input_manifest.json").exists():
        raise ValueError(
            "Ausgabeordner bereits vorbereitet; jeder Start benötigt einen neuen Laufordner."
        )
    config = active_results.config(benchmark)
    spec = resolve_source(config, extraction_source, extraction_thinking, extraction_shots)
    is_llm = spec["kind"] == "stored_llm_extraction"
    folder = benchmark / "records"
    sources = records(folder / "eval.jsonl")
    source_map = unique(sources)
    selected = selected_predictions(benchmark, repetition, spec, history)
    extractions = unique([p for p in selected if p["task"] == "extraction"])
    categories = unique([p for p in selected if p["task"] == "classification"])
    if not sources or extractions.keys() != source_map.keys():
        raise ValueError(
            f"{spec['method']}: Extraktionen nicht vollständig ({len(extractions)}/{len(sources)}). "
            "Zuerst diese Methode im Benchmark abschließen oder einen anderen Benchmark wählen."
        )
    if not categories.keys() <= source_map.keys():
        raise ValueError(
            "Gespeicherte Klassifikation enthält Berichte außerhalb dieses Evaluationssatzes."
        )
    if len({str(r["original_index"]) for r in sources}) != len(sources):
        raise ValueError(
            "Mehrere Berichte pro Quellereignis passen nicht zum bestehenden Eins-zu-eins-Matching."
        )

    event_bytes = Path(eventlog).read_bytes()
    events = pd.read_csv(io.BytesIO(event_bytes), dtype=str, keep_default_na=False)
    required = {"original_index", *matching.EVENT_FIELDS}
    if not required.issubset(events.columns) or events["original_index"].duplicated().any():
        raise ValueError(
            "Eventlog benötigt eindeutige original_index und die vier Matching-Attribute."
        )
    event_rows = events.to_dict("records")
    lookup = {r["original_index"]: r for r in event_rows}
    eligible = [
        {key: r[key] for key in ("original_index", *matching.EVENT_FIELDS)}
        for r in event_rows
        if "machine" in r["Resource"].casefold()
    ]
    eligible_ids = {r["original_index"] for r in eligible}
    inputs, metadata, truth = [], [], []
    for source in sources:
        rid, target = source["report_id"], str(source["original_index"])
        prediction, category = extractions[rid], categories.get(rid)
        for record in (prediction, category):
            if record is None:
                continue
            if record.get("text_sha256") != sha(source["error_text"].encode()):
                raise ValueError(f"Gespeicherte Vorhersage passt nicht zum Benchmark-Text: {rid}")
            if is_llm and (
                record.get("model") != spec["model"]
                or record.get("thinking") != spec["thinking"]
                or type(record.get("thinking")) is not type(spec["thinking"])
            ):
                raise ValueError(f"Gespeicherte Vorhersage passt nicht zu Modell/Thinking: {rid}")
            if (
                is_llm
                and "extraction_shots" in config["settings"]
                and (type(record.get("shots")) is not int or record["shots"] != spec["shots"])
            ):
                raise ValueError(f"Gespeicherte Vorhersage passt nicht zur Shot-Variante: {rid}")
        if not isinstance(prediction.get("raw_output"), str):
            raise ValueError(
                f"raw_output fehlt für {rid}; leere Antworten müssen als leerer String vorliegen."
            )
        if target not in eligible_ids:
            raise ValueError(f"Quellereignis {target} fehlt im Maschinen-Eventlog.")
        event = lookup[target]
        if any(
            str(event[col]) != str(source["ground_truth"][field])
            for col, field in zip(matching.EVENT_FIELDS, matching.FIELDS)
        ):
            raise ValueError(f"Eventlog und Benchmark-Quellwerte unterscheiden sich: {rid}")
        inputs.append({"report_id": rid, "raw_output": prediction["raw_output"]})
        metadata.append(
            {
                "report_id": rid,
                "error_original_index": target,
                "Prompt_Level": source["difficulty_level"],
                "Error Text": source["error_text"],
                "Extracted Info": prediction["raw_output"],
                "Category": category["prediction"] if category else "",
            }
        )
        truth.append(
            {
                "report_id": rid,
                "target_event_id": target,
                **{col: event[col] for col in matching.EVENT_FIELDS},
            }
        )

    if use_original_report:
        by_id = unique(metadata)
        inputs = [dict(row, error_text=by_id[row["report_id"]]["Error Text"]) for row in inputs]
        if any(
            not isinstance(row["error_text"], str) or not row["error_text"].strip()
            for row in inputs
        ):
            raise ValueError("Matching mit Originalbericht benötigt für jeden Bericht error_text.")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "matching_inputs.jsonl", inputs)
    write_jsonl(output / "report_metadata.jsonl", metadata)
    write_jsonl(output / "validation_truth.jsonl", truth)
    write_jsonl(output / "selected_predictions.jsonl", selected)
    (output / "input_events.csv").write_bytes(event_bytes)
    manifest = {
        "version": matching.VERSION,
        "benchmark": "../benchmark",
        "benchmark_run": repetition,
        "source_model": spec["model"],
        "thinking": spec["thinking"],
        "source_shots": spec["shots"],
        "source_kind": spec["kind"],
        "source_method": spec["method"],
        "extraction_source": extraction_source,
        "matching_model": matching_model,
        "embedding_model": embedding_model,
        "host": host,
        "settings": asdict(settings),
        "order_seeds": list(order_seeds),
        "reports": len(inputs),
        "eventlog_rows": len(events),
        "eligible_events": len(eligible),
        "evaluation_sha256": sha((folder / "eval.jsonl").read_bytes()),
        "benchmark_config_sha256": sha((benchmark / "config.json").read_bytes()),
        "selected_predictions_sha256": sha((output / "selected_predictions.jsonl").read_bytes()),
        "eventlog_sha256": sha(event_bytes),
        "source_training": None
        if is_llm
        else {
            key: config["settings"].get(key)
            for key in ("fractions", "ner_model", "ner_epochs", "ner_batch_size", "ner_lr")
        },
        "input_policy": "Stored raw_output verbatim. Only report_id/raw_output and public event fields enter matching; truth is evaluated afterwards.",
        "classification_policy": "Optional same-model/same-thinking/same-shot classification is exported only. If unavailable, Category is empty. Classification never enters matching.",
        "cache_policy": "Empty in-memory cache for every order/run; no cross-run reuse.",
        "selection_policy": "Same configured matching settings for all sources; no fitting or automatic source substitution.",
    }
    manifest.update(
        use_original_report=use_original_report,
        prompt_version=matching.PROMPT_VERSION if use_original_report else "attributes-only-1",
        matching_inputs_sha256=sha((output / "matching_inputs.jsonl").read_bytes()),
        input_policy=(
            "Stored raw_output and original error_text verbatim. Only report_id/raw_output/error_text "
            "and public event fields enter matching. Original text is used exclusively in Pass 3; "
            "truth is evaluated afterwards."
            if use_original_report
            else "Stored raw_output verbatim. Only report_id/raw_output and public event fields enter matching; "
            "truth is evaluated afterwards."
        ),
    )
    write_json(output / "input_manifest.json", manifest)
    print(
        f"Bereit: {len(inputs)} gespeicherte {spec['method']}-Extraktionen; "
        f"{len(eligible)} Maschinenereignisse im vollständigen Log."
    )
    return inputs, eligible, metadata, truth, event_rows, manifest


def rates(rows):
    total = len(rows)
    assigned = sum(r["matched_event_id"] is not None for r in rows)
    correct = sum(r["Match"] for r in rows)
    return {
        "total": total,
        "assigned": assigned,
        "unassigned": total - assigned,
        "correct": correct,
        "wrong_assigned": assigned - correct,
        "accuracy": correct / total if total else None,
        "coverage": assigned / total if total else None,
        "precision_assigned": correct / assigned if assigned else None,
        "correct_event_id": sum(r["event_id_match"] for r in rows),
    }


def evaluate(results, metadata, truth, all_events, output, performance):
    output = Path(output)
    meta, reference = unique(metadata), unique(truth)
    if unique(results).keys() != reference.keys() or meta.keys() != reference.keys():
        raise ValueError("Auswertung unvollständig: Jeder Eingabebericht muss erhalten bleiben.")
    lookup = {str(e["original_index"]): e for e in all_events}
    rows = []
    for result in results:
        rid = result["report_id"]
        event = lookup.get(result["matched_event_id"])
        target = reference[rid]
        match = bool(event) and all(str(event[c]) == str(target[c]) for c in matching.EVENT_FIELDS)
        row = (
            meta[rid]
            | result
            | {
                "Match": match,
                "event_id_match": result["matched_event_id"] == target["target_event_id"],
                "filter_target_present": target["target_event_id"]
                in result["filter_candidate_ids"],
                "top10_target_present": (
                    target["target_event_id"] in result["top10_candidate_ids"]
                    if result["top10_candidate_ids"] is not None
                    else None
                ),
                "llm_target_present": (
                    target["target_event_id"] in result["llm_candidate_ids"]
                    if result["llm_candidate_ids"] is not None
                    else None
                ),
            }
        )
        row.update(
            {
                "event_" + c: event[c] if event else None
                for c in ("original_index", *matching.EVENT_FIELDS)
            }
        )
        rows.append(row)
    summary = rates(rows)
    summary["by_level"] = {
        str(level): rates([r for r in rows if r["Prompt_Level"] == level])
        for level in sorted({r["Prompt_Level"] for r in rows})
    }
    summary["by_method"] = {
        method: rates([r for r in rows if r["match_method"] == method])
        for method in sorted({r["match_method"] for r in rows})
    }
    summary["retrieval"] = {}
    for stage in ("filter", "top10", "llm"):
        key = stage + "_target_present"
        observed = [r[key] for r in rows if r[key] is not None]
        summary["retrieval"][stage] = {
            "correct_target_present": sum(observed),
            "evaluated": len(observed),
            "rate": sum(observed) / len(observed) if observed else None,
        }
    summary["llm_tokens"] = {
        k: sum(r[k] for r in rows)
        for k in ("match_prompt_tokens", "match_completion_tokens", "match_total_tokens")
    }
    summary["performance"] = performance
    summary["by_pass"] = pass_rows(summary)
    summary["scoring"] = (
        "Primary: same four exact event attributes as the paper; event ID correctness additionally reported. All input reports in denominator."
    )
    timing.save_csv(output / "matched_reports.csv", rows)
    timing.save_csv(output / "validation_results.csv", rows)
    write_jsonl(output / "matching_decisions.jsonl", results)
    write_json(output / "summary.json", summary)
    timing.save_csv(output / "pass_summary.csv", summary["by_pass"])
    assigned = {r["matched_event_id"]: r for r in rows if r["matched_event_id"] is not None}
    if len(assigned) != summary["assigned"]:
        raise ValueError("Ein Ereignis wurde mehrfach zugeordnet.")
    full = [
        dict(
            e,
            **{
                key: assigned.get(str(e["original_index"]), {}).get(key, "")
                for key in ("Error Text", "Extracted Info", "Category")
            },
        )
        for e in all_events
    ]
    timing.save_csv(output / "matched_eventlog.csv", full)
    return summary, rows


def print_summary(summary):
    print("\nFINAL MATCHING – alle Eingabeberichte, einschließlich leerer Extraktionen")
    for level, result in summary["by_level"].items():
        print(
            f"Stufe {level}: {result['correct']}/{result['total']} ({result['accuracy']:.1%}); "
            f"zugeordnet {result['assigned']}/{result['total']}"
        )
    print(f"Gesamt: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
    precision = summary["precision_assigned"]
    print(
        f"Zuordnungsquote: {summary['coverage']:.1%}; korrekt unter den Zuordnungen: "
        + (f"{precision:.1%}" if precision is not None else "nicht definiert (keine Zuordnung)")
    )
    print_pass_summary(summary)
    for stage, data in summary["retrieval"].items():
        print(
            f"Richtiges Ereignis in {stage}: {data['correct_target_present']}/{data['evaluated']}"
        )
    print(f"LLM-Tokens inklusive Ablehnungen: {summary['llm_tokens']['match_total_tokens']}")
    perf = summary["performance"]
    print(
        f"Pass 3 mit Originalbericht: {perf['use_original_report']}; Prompt: {perf['prompt_version']}"
    )
    print(f"Vergleich: {perf['score_mode']}; Pass-1-Normalisierung: {perf['pass1_normalized']}")
    print(f"Matching-Zeit inklusive aller Embeddings: {perf['component_total_sec']:.2f}s")
    counts = perf["cache"]["counts"]
    print(
        f"Embeddings: {counts['computed']} neu, {counts['memory_hits']} im selben Lauf wiederverwendet."
    )
