import csv
import hashlib
import io
import json
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import prediction_records

VERSION = "ollama-timing-2"
NS_FIELDS = {
    "total_duration": "ollama_total_sec",
    "load_duration": "ollama_load_sec",
    "prompt_eval_duration": "ollama_prompt_eval_sec",
    "eval_duration": "ollama_eval_sec",
}
MEASURES = (
    "wall_time_sec",
    *NS_FIELDS.values(),
    "ollama_inference_sec",
    "ollama_total_without_load_sec",
    "wall_minus_ollama_sec",
    "ollama_other_sec",
)
IDENTITY = (
    "attempt_id",
    "run",
    "report_id",
    "method",
    "task",
    "difficulty_level",
    "model",
    "model_digest",
    "thinking",
    "shots",
    "baseline",
    "training_percent",
    "training_n",
    "request_started_at",
    "request_finished_at",
    "api_error",
)


def number(value, *, integer=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (integer and int(value) != value)
    ):
        return None
    return int(value) if integer else float(value)


def extract(raw_response, wall_time_sec):
    raw = raw_response if isinstance(raw_response, dict) else {}
    result = {"timing_version": VERSION, "wall_time_sec": number(wall_time_sec)}
    warnings = []
    for source, target in NS_FIELDS.items():
        value = number(raw.get(source))
        if source in raw and raw[source] is not None and value is None:
            warnings.append("invalid_" + source)
        result[target] = value / 1_000_000_000 if value is not None else None
    for source, target in [
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "completion_tokens"),
        ("prompt_eval_cached_count", "prompt_cached_tokens"),
    ]:
        result[target] = number(raw.get(source), integer=True)
    total, load = result["ollama_total_sec"], result["ollama_load_sec"]
    prompt, generation = result["ollama_prompt_eval_sec"], result["ollama_eval_sec"]
    inference = prompt + generation if prompt is not None and generation is not None else None
    result["ollama_inference_sec"] = inference
    result["ollama_total_without_load_sec"] = (
        total - load if total is not None and load is not None else None
    )
    result["ollama_other_sec"] = (
        total - load - inference if all(x is not None for x in (total, load, inference)) else None
    )
    wall = result["wall_time_sec"]
    result["wall_minus_ollama_sec"] = (
        wall - total if wall is not None and total is not None else None
    )
    for name in ("ollama_total_without_load_sec", "ollama_other_sec", "wall_minus_ollama_sec"):
        if result[name] is not None and result[name] < 0:
            warnings.append("negative_" + name)
    tokens = result["completion_tokens"]
    result["generation_tokens_per_sec"] = (
        tokens / generation
        if tokens is not None and generation is not None and generation > 0
        else None
    )
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    result.update(
        ollama_done=raw.get("done"),
        ollama_done_reason=raw.get("done_reason"),
        ollama_created_at=raw.get("created_at"),
        ollama_model=raw.get("model"),
        answer_chars=len(message["content"]) if isinstance(message.get("content"), str) else None,
        thinking_chars=len(message["thinking"])
        if isinstance(message.get("thinking"), str)
        else None,
        timing_warnings=warnings,
    )
    count = sum(result[name] is not None for name in NS_FIELDS.values())
    result["ollama_timing_status"] = (
        "complete" if count == len(NS_FIELDS) else "partial" if count else "unavailable"
    )
    return result


def enrich(record):
    if not str(record.get("method", "")).startswith("llm_"):
        return dict(
            record,
            wall_time_sec=number(record.get("latency_sec")),
            ollama_timing_status="not_applicable",
        )
    return dict(record) | extract(record.get("raw_response"), record.get("latency_sec"))


def percentile95(values):
    values = sorted(values)
    position = 0.95 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(records):
    records = [enrich(r) for r in records]
    result = {
        "timing_n_requests": len(records),
        "timing_n_api_errors": sum(bool(r.get("api_error")) for r in records),
        "timing_n_length_stops": sum(r.get("ollama_done_reason") == "length" for r in records),
    }
    for measure in MEASURES:
        values = [r[measure] for r in records if r.get(measure) is not None]
        result["n_" + measure] = len(values)
        result["sum_" + measure] = sum(values) if values else None
        result["mean_" + measure] = statistics.fmean(values) if values else None
        result["median_" + measure] = statistics.median(values) if values else None
        result["p95_" + measure] = percentile95(values) if values else None
    for key in ("prompt_tokens", "completion_tokens", "prompt_cached_tokens"):
        values = [r[key] for r in records if r.get(key) is not None]
        result["n_" + key] = len(values)
        result["sum_" + key] = sum(values) if values else None
    rate_rows = [r for r in records if r.get("generation_tokens_per_sec") is not None]
    result["n_generation_tokens_per_sec"] = len(rate_rows)
    result["generation_tokens_per_sec"] = (
        (
            sum(r["completion_tokens"] for r in rate_rows)
            / sum(r["ollama_eval_sec"] for r in rate_rows)
        )
        if rate_rows
        else None
    )
    return result


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def as_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def save_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in record.items()
            }
        )
    atomic(path, output.getvalue())


def export(source, output=None):
    source = Path(source)
    output = Path(output) if output is not None else source / "timing"
    if output.resolve() == source.resolve() or output.name.startswith("run_"):
        raise ValueError("Zeitexport benötigt einen separaten Unterordner oder Ausgabeordner.")
    files = sorted(source.glob("records/predictions.jsonl"))
    if not files:
        raise ValueError("Keine gespeicherten Vorhersagen für den Zeitexport gefunden.")
    all_records, provenance = [], []
    for path in files:
        data = path.read_bytes()
        run = 1
        records = [
            json.loads(line) | {"run": run} for line in data.decode().splitlines() if line.strip()
        ]
        provenance.append(
            {
                "file": str(path.relative_to(source)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "records": len(records),
            }
        )
        all_records.extend(records)
    requests = []
    grouped = defaultdict(list)
    all_grouped = defaultdict(list)
    prediction_records.completed(all_records)
    telemetry_fields = set(extract({}, None))
    for record in all_records:
        enriched = enrich(record)
        requests.append(
            {key: enriched.get(key) for key in IDENTITY}
            | {k: v for k, v in enriched.items() if k in telemetry_fields}
            | {"included_in_benchmark": not bool(record.get("api_error"))}
        )
        for rung in ("all", record["difficulty_level"]):
            key = (record["run"], record["method"], record["task"], rung)
            all_grouped[key].append(record)
            if not record.get("api_error"):
                grouped[key].append(record)
    summaries = [
        {"run": run, "method": method, "task": task, "rung": rung, **summarize(rows)}
        for (run, method, task, rung), rows in grouped.items()
    ]
    save_csv(output / "requests.csv", requests)
    atomic(
        output / "requests.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in requests),
    )
    save_csv(output / "summary.csv", summaries)
    atomic(output / "summary.json", as_json({"version": VERSION, "results": summaries}))
    all_summaries = [
        {"run": run, "method": method, "task": task, "rung": rung, **summarize(rows)}
        for (run, method, task, rung), rows in all_grouped.items()
    ]
    save_csv(output / "all_attempts_summary.csv", all_summaries)
    atomic(
        output / "all_attempts_summary.json",
        as_json({"version": VERSION, "results": all_summaries}),
    )
    atomic(
        output / "provenance.json",
        as_json(
            {
                "version": VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sources": provenance,
                "definitions": {
                    "ollama_inference_sec": "(prompt_eval_duration + eval_duration) / 1e9; server-reported inference, without client transport or separate load phase",
                    "ollama_total_sec": "total_duration / 1e9; full server-reported request duration",
                    "wall_time_sec": "client-measured request/response latency (latency_sec); baseline values use their existing local inference measurement",
                    "wall_minus_ollama_sec": "signed wall minus server total; includes client/transport and different measurement boundaries, not pure network latency",
                    "ollama_other_sec": "signed server total minus load and inference; not an exact queue-time measurement",
                    "generation_tokens_per_sec": "sum generated tokens / sum generation duration over matching rows with positive duration",
                    "statistics": "summary: returned responses only, including invalid replies; API failures excluded. all_attempts_summary and requests retain every attempt and its overhead. Available values only; counts per measure; p95 uses linear interpolation.",
                },
                "limits": [
                    "No server timing is invented when absent.",
                    "Thinking and final-answer durations/token counts are not separately provided by these API fields.",
                    "Prompt cache counts are absent on older Ollama versions; character counts are not token counts.",
                    "Server load, hardware and contention still affect inference time. Training time is separate.",
                    "These exports contain recorded attempts, not an assertion that the experiment completed.",
                ],
            }
        ),
    )
    return summaries


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ollama-Zeitdaten aus gespeicherten Rohantworten exportieren; keine Modellaufrufe."
    )
    parser.add_argument("--source", type=Path, required=True, help="Benchmark-Ergebnisordner")
    parser.add_argument("--output", type=Path, help="Standard: <source>/timing")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("Ergebnisordner existiert nicht.")
    summaries = export(args.source, args.output)
    requests = sum(r["timing_n_requests"] for r in summaries if r["rung"] == "all")
    timed = sum(r["n_ollama_inference_sec"] for r in summaries if r["rung"] == "all")
    print(
        f"{requests} gespeicherte Aufrufe exportiert, davon {timed} mit Ollama-Inferenzzeit. "
        f"Ausgabe: {args.output or args.source / 'timing'}"
    )


if __name__ == "__main__":
    main()
