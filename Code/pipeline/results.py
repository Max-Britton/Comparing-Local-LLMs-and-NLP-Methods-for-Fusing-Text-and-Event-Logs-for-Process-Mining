import json
from pathlib import Path


TIME_FIELDS = (
    "component_wall_sec",
    "local_python_sec",
    "ollama_server_sec",
    "ollama_load_sec",
    "compute_sec",
    "compute_without_load_sec",
    "client_transport_residual_sec",
)


def load(path):
    return json.loads(Path(path).read_text())


def matching_records(output, jobs):
    records = []
    for job in jobs:
        path = Path(output) / job["relative_output"] / "summary.json" if output else None
        if path is None or not path.exists():
            summary = {}
        else:
            summary = load(path)
        complete = path is not None and path.exists() and path.with_name("completed.json").exists()
        for rung in ["all", "1", "2", "3", "4"]:
            scores = summary if rung == "all" else summary.get("by_level", {}).get(rung, {})
            item = {
                "run": job["run"],
                "method": job["source_method"],
                "task": "matching",
                "pass3_method": job["pass3_method"],
                "shots": job.get("source_shots"),
                "training_percent": job.get("source_training_percent"),
                "rung": rung,
                "complete": complete,
            }
            item.update(
                {
                    k: scores.get(k)
                    for k in (
                        "total",
                        "assigned",
                        "correct",
                        "wrong_assigned",
                        "unassigned",
                        "accuracy",
                        "coverage",
                        "precision_assigned",
                    )
                }
            )
            if rung == "all":
                perf = summary.get("performance", {})
                item.update({k: perf.get("final_timing", {}).get(k) for k in TIME_FIELDS})
                item["model_management_sec"] = perf.get("model_management_sec")
                item["llm_tokens"] = summary.get("llm_tokens", {}).get("match_total_tokens")
                for index, row in enumerate(summary.get("by_pass", []), 1):
                    for k, v in row.items():
                        if type(v) in (int, float):
                            item[f"pass{index}_{k}"] = v
            records.append(item)
    return records
