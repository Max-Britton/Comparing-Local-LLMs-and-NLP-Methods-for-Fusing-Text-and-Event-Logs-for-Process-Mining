from dataclasses import asdict
from pathlib import Path

from . import benchmark, component_timing, learning_curve, progress, results, active_results
from . import matching as engine
from . import matching_data as adapter
from .matching_support import EmbeddingCache, model_digest
from .ollama_client import Transport
from .ollama_runtime import host_lock
from .settings import matching_settings


def build_jobs(c, config):
    if type(c.DURCHLAEUFE) is not int or c.DURCHLAEUFE != 1:
        raise ValueError("DURCHLAEUFE muss eine positive ganze Zahl sein.")
    if c.MATCHING_LLM_MODUS not in ("gleiches_modell", "liste"):
        raise ValueError("MATCHING_LLM_MODUS: gleiches_modell oder liste.")
    percentages = getattr(c, 'MATCHING_TRAININGSANTEILE', None) or c.BASELINE_TRAININGSANTEILE
    if (
        not isinstance(percentages, list)
        or not percentages
        or any(type(p) is not int or not 1 <= p <= 100 for p in percentages)
        or len(set(percentages)) != len(percentages)
    ):
        raise ValueError(
            "BASELINE_TRAININGSANTEILE: verschiedene ganze Prozentwerte von 1 bis 100."
        )
    sources = c.MATCHING_QUELLEN
    if sources is None:
        llm_extraction = "extraction" in config["settings"].get("tasks", ["extraction"]) and bool(
            config["settings"].get("extraction_shots", [1])
        )
        sources = [*(c.MODELLE if llm_extraction else []), *c.EXTRAKTIONS_BASELINES]
    if not sources:
        raise ValueError("Keine Extraktionsquellen für Matching ausgewählt.")
    expanded_sources = []
    baseline_conditions = learning_curve.conditions(config["settings"])
    for source in sources:
        if isinstance(source, dict):
            if not {"name", "thinking"} <= source.keys() or set(source) - {
                "name",
                "thinking",
                "shots",
            }:
                raise ValueError(
                    "Matching-LLM-Quellen benötigen name, thinking und optional shots."
                )
            counts = (
                [source["shots"]]
                if "shots" in source
                else config["settings"].get("extraction_shots", [1])
            )
            if not counts:
                raise ValueError(
                    f"Keine LLM-Extraktions-Shots für {source['name']} im Quellbenchmark."
                )
            expanded_sources.extend(dict(source, shots=n) for n in counts)
        else:
            available = [v for v in baseline_conditions if v["baseline"] == source]
            if available:
                if source != "regex":
                    missing = set(percentages) - {v["training_percent"] for v in available}
                    if missing:
                        raise ValueError(
                            f"Trainingsanteile {sorted(missing)} für {source} fehlen im Quellbenchmark."
                        )
                expanded_sources.extend(
                    v["method"]
                    for v in available
                    if v["training_percent"] is None or v["training_percent"] in percentages
                )
            else:
                expanded_sources.append(source)
    if c.MATCHING_QUELL_DURCHLAUF is not None and (
        type(c.MATCHING_QUELL_DURCHLAUF) is not int or c.MATCHING_QUELL_DURCHLAUF < 1
    ):
        raise ValueError("MATCHING_QUELL_DURCHLAUF muss None oder eine positive ganze Zahl sein.")
    jobs, source_ids = [], set()
    for source in expanded_sources:
        name, think = (
            (source["name"], source["thinking"]) if isinstance(source, dict) else (source, False)
        )
        spec = adapter.resolve_source(
            config, name, think, source.get("shots") if isinstance(source, dict) else None
        )
        if spec["method"] in source_ids:
            raise ValueError("Doppelte Matching-Quelle: " + spec["method"])
        source_ids.add(spec["method"])
        pickers = (
            c.MATCHING_MODELLE
            if c.MATCHING_LLM_MODUS == "liste"
            else [
                {"name": name, "thinking": think}
                if isinstance(source, dict)
                else c.BASELINE_MATCHING_MODELL
            ]
        )
        if not pickers:
            raise ValueError("Mindestens ein Pass-3-Modell auswählen.")
        if len({benchmark.method_id(m) for m in pickers}) != len(pickers):
            raise ValueError("Doppelte Pass-3-Modellbedingung.")
        for picker in pickers:
            if not isinstance(picker, dict) or set(picker) != {"name", "thinking"}:
                raise ValueError("Pass-3-Modelle benötigen name und thinking.")
            if type(picker["thinking"]) is not bool and picker["thinking"] not in (
                "low",
                "medium",
                "high",
            ):
                raise ValueError("Ungültige Thinking-Einstellung für Pass 3.")
            if "gpt-oss" in picker["name"].lower() and type(picker["thinking"]) is bool:
                raise ValueError("GPT-OSS in Pass 3 benötigt low/medium/high.")
            pid = benchmark.method_id(picker)
            for repetition in range(1, c.DURCHLAEUFE + 1):
                source_run = c.MATCHING_QUELL_DURCHLAUF or repetition
                if source_run > config["settings"]["repetitions"]:
                    raise ValueError(
                        f"Benchmark-Durchlauf {source_run} fehlt. Für wiederholtes Matching "
                        "vorhandener Extraktionen MATCHING_QUELL_DURCHLAUF einstellen."
                    )
                jobs.append(
                    {
                        "run": repetition,
                        "source_run": source_run,
                        "source_name": name,
                        "source_thinking": think,
                        "source_shots": spec["shots"],
                        "source_training_percent": spec.get("training_percent"),
                        "source_method": spec["method"],
                        "pass3_model": dict(picker),
                        "pass3_method": pid,
                        "seed": c.MODELL_SEED + repetition - 1,
                        "order_seed": c.MATCHING_REIHENFOLGE_SEED + repetition - 1
                        if c.MATCHING_REIHENFOLGE_MISCHEN
                        else None,
                        "relative_output": f"{spec['method']}__via__{pid}",
                    }
                )
    return jobs


def source_folder(c):
    return Path(c.MATCHING_BENCHMARK_ORDNER or c.ERGEBNIS_ORDNER / "benchmark")


def chat_options(c, job):
    return {
        "temperature": c.TEMPERATUR,
        "seed": job["seed"],
        "num_ctx": c.KONTEXT_TOKEN,
        "num_predict": c.ANTWORT_TOKEN_MATCHING
        if job["pass3_model"]["thinking"] is False
        else c.ANTWORT_TOKEN_MIT_THINKING,
    }


def completion_error(c, job, folder, history=None):
    required = ("completed.json", "summary.json", "input_manifest.json", "job.json",
                "matching_decisions.jsonl", "validation_truth.jsonl")
    if not all((folder / name).is_file() for name in required):
        return "incomplete"
    stored = results.load(folder / "job.json")
    if any(stored.get(k) != v for k, v in job.items()):
        return "different job settings"
    if stored.get("chat_options") != chat_options(c, job):
        return "different model options"
    source = source_folder(c)
    spec = adapter.source_info(source, job["source_name"], job["source_thinking"], job["source_shots"])
    selected = adapter.selected_predictions(source, 1, spec, history)
    manifest = results.load(folder / "input_manifest.json")
    expected = {
        "settings": asdict(matching_settings(c)),
        "use_original_report": c.MIT_ORIGINALBERICHT,
        "embedding_model": c.EMBEDDING_MODELL,
        "evaluation_sha256": adapter.sha((source / "records/eval.jsonl").read_bytes()),
        "selected_predictions_sha256": adapter.selected_sha(selected),
        "eventlog_sha256": adapter.sha(c.EVENTLOG_DATEI.read_bytes()),
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        return "different data or matching parameters"
    n = manifest.get("reports")
    if not n or results.load(folder / "completed.json").get("reports") != n or results.load(folder / "summary.json").get("total") != n:
        return "incomplete report population"
    return None


def pending_jobs(c, jobs):
    history = active_results.predictions(source_folder(c))
    pending = []
    for job in jobs:
        folder = c.ERGEBNIS_ORDNER / "matching" / job["relative_output"]
        error = completion_error(c, job, folder, history)
        if error and (folder / "completed.json").exists():
            raise ValueError(f"Completed matching does not fit the inputs ({error}): {folder}. Start a new run.")
        if error:
            pending.append(job)
    return pending


def merge_models(output, details, embed_digest):
    path = output / "models.json"
    stored = results.load(path) if path.exists() else {"chat": {}, "embedding_digest": embed_digest}
    if stored["embedding_digest"] != embed_digest:
        raise ValueError("Embedding model changed within the run.")
    for name, detail in details.items():
        if name in stored["chat"] and stored["chat"][name]["digest"] != detail["digest"]:
            raise ValueError(f"Model changed within the run: {name}")
        stored["chat"][name] = detail
    return stored


def run(c, *, plan_only=False, prepare_only=False, client_factory=None, jobs=None, on_progress=None):
    benchmark_folder = source_folder(c)
    if not (benchmark_folder / "config.json").exists():
        raise ValueError("Benchmark fehlt. Zuerst MainBenchmark.py starten (oder --nur-planen).")
    config = active_results.config(benchmark_folder)
    all_jobs = build_jobs(c, config)
    if jobs is None:
        jobs = all_jobs
    elif not jobs or any(job not in all_jobs for job in jobs):
        raise ValueError('Ungültige Teilmenge der Matching-Aufträge.')
    settings = matching_settings(c)
    jobs = pending_jobs(c, jobs)
    print(f"{len(jobs)} Matching-Läufe: genau {c.DURCHLAEUFE} pro Quellen-/Pass-3-Kombination.")
    for job in jobs:
        print(
            f"  {job['source_method']} → {job['pass3_method']}; Lauf {job['run']}, Benchmark {job['source_run']}"
        )
    if plan_only:
        print("Nur Jobplan; keine Modelle, Embeddings oder Ergebnisdateien erzeugt.")
        return jobs
    if not jobs:
        return c.ERGEBNIS_ORDNER / "matching"
    root = c.ERGEBNIS_ORDNER / "matching"
    root.mkdir(parents=True, exist_ok=True)
    output = root
    manifest = {
        "version": "matching-suite-1",
        "jobs": all_jobs,
        "requested_runs": c.DURCHLAEUFE,
        "benchmark": "../benchmark",
        "settings": asdict(settings),
        "use_original_report": c.MIT_ORIGINALBERICHT,
        "options": {j["relative_output"]: chat_options(c, j) for j in all_jobs},
        "policy": "one fresh cache per job; no extra order runs; all configured jobs retained",
    }
    adapter.write_json(output / "suite.json", manifest)
    history = active_results.predictions(benchmark_folder)
    for job in progress.iterate(jobs, c, desc="Matching-Eingaben vorbereiten", unit="Lauf"):
        folder = output / job["relative_output"]
        if folder.exists():
            import shutil
            shutil.rmtree(folder)
        adapter.prepare_inputs(
            benchmark_folder,
            job["source_run"],
            c.EVENTLOG_DATEI,
            folder,
            extraction_source=job["source_name"],
            extraction_thinking=job["source_thinking"],
            extraction_shots=job["source_shots"],
            matching_model=job["pass3_model"]["name"],
            embedding_model=c.EMBEDDING_MODELL,
            host=c.OLLAMA_HOST,
            settings=settings,
            order_seeds=[job["order_seed"]],
            use_original_report=c.MIT_ORIGINALBERICHT,
            history=history,
        )
        adapter.write_json(
            folder / "job.json",
            job
            | {
                "chat_options": chat_options(c, job),
                "embedding_api": "/api/embed",
                "truncate": False,
            },
        )
    if prepare_only:
        print(f"Alle Eingaben geprüft und vorbereitet: {output}. Keine Modelle gestartet.")
        return output
    if client_factory is None:
        from ollama import Client

        client_factory = Client
    try:
        with host_lock(c.OLLAMA_HOST):
            client = client_factory(host=c.OLLAMA_HOST, timeout=c.TIMEOUT_SEKUNDEN)
            pickers = {j["pass3_method"]: j["pass3_model"] for j in jobs}
            details = benchmark.preflight(client, list(pickers.values()))
            installed = list(client.list().models)
            embed_digest = model_digest(installed, c.EMBEDDING_MODELL)
            adapter.write_json(
                output / "models.json", merge_models(output, details, embed_digest)
            )
            for index, job in enumerate(jobs, 1):
                folder = output / job["relative_output"]
                reports = adapter.records(folder / "matching_inputs.jsonl")
                metadata = adapter.records(folder / "report_metadata.jsonl")
                truth = adapter.records(folder / "validation_truth.jsonl")
                import pandas as pd

                full_log = pd.read_csv(
                    folder / "input_events.csv", dtype=str, keep_default_na=False
                ).to_dict("records")
                events = [
                    {k: e[k] for k in ("original_index", *engine.EVENT_FIELDS)}
                    for e in full_log
                    if "machine" in e["Resource"].casefold()
                ]
                print(
                    f"\nMatching {index}/{len(jobs)}: {job['source_method']} / {job['pass3_method']}, "
                    f"Wiederholung {job['run']}/{c.DURCHLAEUFE}; Cache leer."
                )
                transport = Transport(
                    client,
                    job["pass3_model"],
                    options=chat_options(c, job),
                    audit_path=folder / "model_lifecycle.json",
                )
                recorded = engine.RecordedOllama(transport, folder / "timing")
                cache = EmbeddingCache(host=c.OLLAMA_HOST, digest=embed_digest)
                cache.identity["api"] = "/api/embed"
                cache.identity["truncate"] = False
                try:
                    assignments, perf = engine.run_matching(
                        reports,
                        events,
                        recorded,
                        embedding_model=c.EMBEDDING_MODELL,
                        matching_model=job["pass3_model"]["name"],
                        cache=cache,
                        settings=settings,
                        order_seed=job["order_seed"],
                        use_original_report=c.MIT_ORIGINALBERICHT,
                        progress=c.FORTSCHRITT_ANZEIGEN,
                    )
                    perf["final_timing"] = component_timing.component(
                        perf["component_total_sec"], recorded.records
                    )
                    perf["model_management_sec"] = transport.management_sec
                    summary, _ = adapter.evaluate(
                        assignments, metadata, truth, full_log, folder, perf
                    )
                    recorded.save()
                    adapter.print_summary(summary)
                    adapter.write_json(
                        folder / "completed.json",
                        {"reports": len(assignments), "run": job["run"]},
                    )
                finally:
                    recorded.save()
                    adapter.write_json(
                        folder / "embedding_cache_statistics.json", cache.summary()
                    )
                    cache.close()
                    transport.close()
                if on_progress is not None:
                    on_progress()
        adapter.write_json(output / "completed.json", {"jobs": len(all_jobs)})
    finally:
        if on_progress is not None:
            on_progress()
    from .retention import compact_matching
    compact_matching(output)
    print(f"Ergebnisse: {output}")
    return output
