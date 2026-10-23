import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csc_matrix, eye, hstack, vstack

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "validation", "eval")
VERSION = "fixed-grouped-class-level-split-1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_source(path):
    from .benchmark import Settings, group_ids, load_corpus

    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ValueError("Für den festen Split die ursprüngliche JSONL-Datei verwenden.")
    rows, source_sha = load_corpus(Settings(input_path=source, output=source.parent / "unused"))
    return rows, group_ids(rows), source_sha


def profile(rows, groups, assignments):
    result = {}
    for part in ("all", *PARTITIONS):
        selected = [
            (row, group)
            for row, group in zip(rows, groups)
            if part == "all" or assignments[row["report_id"]] == part
        ]
        result[part] = {
            "reports": len(selected),
            "groups": len({g for _, g in selected}),
            "classes": dict(sorted(Counter(r["true_category"] for r, _ in selected).items())),
            "levels": dict(
                sorted(Counter(str(r["difficulty_level"]) for r, _ in selected).items())
            ),
            "class_level": [
                {
                    "category": category,
                    "level": level,
                    "reports": sum(
                        r["true_category"] == category and r["difficulty_level"] == level
                        for r, _ in selected
                    ),
                }
                for category in sorted({r["true_category"] for r in rows})
                for level in sorted({r["difficulty_level"] for r in rows})
            ],
        }
    return result


def check_assignments(rows, groups, assignments, sizes):
    if set(assignments) != {r["report_id"] for r in rows}:
        raise ValueError("Split muss jeden Bericht genau einmal enthalten.")
    if set(assignments.values()) - set(PARTITIONS):
        raise ValueError("Unbekannter Teilbestand im Split.")
    if dict(Counter(assignments.values())) != {part: n for part, n in sizes.items() if n}:
        raise ValueError("Split-Größen entsprechen nicht den festgelegten Anzahlen.")
    owners = {}
    for row, group in zip(rows, groups):
        part = assignments[row["report_id"]]
        if owners.setdefault(group, part) != part:
            raise ValueError(
                "Ein Fertigungsfall, Quellereignis oder gleicher Text überschneidet die Teilbestände."
            )
    stats = profile(rows, groups, assignments)
    for part in ("eval", "validation"):
        cells = stats[part]["class_level"]
        if len({cell["reports"] for cell in cells}) != 1:
            raise ValueError(f"{part}: Klasse × Stufe ist nicht exakt ausgeglichen.")
    for part in ("all", "train", "validation", "eval"):
        for field in ("classes", "levels"):
            if len(set(stats[part][field].values())) != 1:
                raise ValueError(f"{part}: {field} ist nicht ausgeglichen.")
    return stats


def select_groups(rows, groups, eval_size, validation_size, seed, time_limit):
    strata = sorted({(r["true_category"], r["difficulty_level"]) for r in rows})
    categories = {r["true_category"] for r in rows}
    levels = {r["difficulty_level"] for r in rows}
    if len(strata) != len(categories) * len(levels):
        raise ValueError("Im Datensatz fehlt mindestens eine Klasse-Stufe-Kombination.")
    if any(type(n) is not int or n < 1 for n in (eval_size, validation_size)):
        raise ValueError("Eval- und Validierungsgröße müssen positive ganze Zahlen sein.")
    if type(seed) is not int or seed < 0 or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Seed muss nichtnegativ und das Zeitlimit positiv sein.")
    if not eval_size or eval_size + validation_size >= len(rows):
        raise ValueError("Evaluation und Training müssen mindestens einen Bericht enthalten.")
    if any(n % len(strata) for n in (eval_size, validation_size)):
        raise ValueError(
            f"Eval/Validierung müssen für exakte Zellbalance durch {len(strata)} teilbar sein."
        )
    group_names = sorted(set(groups))
    group_index = {name: i for i, name in enumerate(group_names)}
    stratum_index = {key: i for i, key in enumerate(strata)}
    matrix = np.zeros((len(strata), len(group_names)))
    for row, group in zip(rows, groups):
        matrix[
            stratum_index[row["true_category"], row["difficulty_level"]], group_index[group]
        ] += 1
    n = len(group_names)
    constraints = vstack(
        [
            block_diag([csc_matrix(matrix), csc_matrix(matrix)]),
            hstack([eye(n), eye(n)]),
        ]
    ).tocsc()
    targets = np.r_[
        np.repeat(eval_size // len(strata), len(strata)),
        np.repeat(validation_size // len(strata), len(strata)),
    ]
    permutation = np.random.default_rng(seed).permutation(2 * n)
    start = time.perf_counter()
    solution = milp(
        np.zeros(2 * n),
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            constraints[:, permutation], np.r_[targets, np.zeros(n)], np.r_[targets, np.ones(n)]
        ),
        options={"time_limit": time_limit, "mip_rel_gap": 0},
    )
    elapsed = time.perf_counter() - start
    if solution.x is None:
        reason = (
            "keine Lösung vorhanden"
            if solution.status == 2
            else "keine Lösung innerhalb des Zeitlimits gefunden"
        )
        raise ValueError(f"Exakt balancierter Gruppensplit: {reason}. Keine Gruppen aufgeteilt.")
    selected = np.zeros(2 * n)
    selected[permutation] = np.rint(solution.x)
    assignments = {
        row["report_id"]: (
            "eval"
            if selected[group_index[group]]
            else "validation"
            if selected[n + group_index[group]]
            else "train"
        )
        for row, group in zip(rows, groups)
    }
    return assignments, {
        "status": int(solution.status),
        "message": solution.message,
        "elapsed_sec": elapsed,
        "time_limit_sec": time_limit,
    }


def create(source, output, eval_size=600, validation_size=120, seed=42, time_limit=120):
    from importlib.metadata import version

    from .benchmark import payload, save_csv

    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(
            f"Split-Ordner existiert bereits: {output}. Für eine andere Aufteilung neuen Ordner wählen."
        )
    rows, groups, source_sha = read_source(source)
    assignments, solver = select_groups(rows, groups, eval_size, validation_size, seed, time_limit)
    sizes = {
        "train": len(rows) - eval_size - validation_size,
        "validation": validation_size,
        "eval": eval_size,
    }
    stats = check_assignments(rows, groups, assignments, sizes)
    source_lines = source.read_bytes().splitlines(keepends=True)
    if sha(b"".join(source_lines)) != source_sha:
        raise ValueError("Quelldatei während der Split-Erstellung verändert.")
    lines = {json.loads(line)["report_id"]: line for line in source_lines if line.strip()}
    manifest = {
        "version": VERSION,
        "source_sha256": source_sha,
        "source_name": source.name,
        "sizes": sizes,
        "seed": seed,
        "solver": solver,
        "scipy_version": version("scipy"),
        "assignments": assignments,
        "statistics": stats,
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".split-") as temporary:
        staging = Path(temporary) / "split"
        staging.mkdir()
        for part in PARTITIONS:
            data = b"".join(
                lines[r["report_id"]] for r in rows if assignments[r["report_id"]] == part
            )
            name = part + ".jsonl"
            (staging / name).write_bytes(data)
            manifest["files"][name] = {"sha256": sha(data), "reports": sizes[part]}
        save_csv(
            staging / "membership.csv",
            [
                {
                    "report_id": r["report_id"],
                    "partition": assignments[r["report_id"]],
                    "case_id": r["case_id"],
                    "original_index": r["original_index"],
                    "true_category": r["true_category"],
                    "difficulty_level": r["difficulty_level"],
                    "split_group": g,
                }
                for r, g in zip(rows, groups)
            ],
        )
        save_csv(
            staging / "distribution.csv",
            [
                {"partition": part, **cell}
                for part in ("all", *PARTITIONS)
                for cell in stats[part]["class_level"]
            ],
        )
        (staging / "manifest.json").write_bytes(payload(manifest))
        staging.rename(output)
    return manifest


def load_fixed(rows, source, manifest_path, fractions):
    manifest = json.loads(Path(manifest_path).read_text())
    full_rows, groups, source_sha = read_source(source)
    if manifest.get("version") != VERSION or source_sha != manifest["source_sha256"]:
        raise ValueError("Gespeicherter Split passt nicht zur TEXT_DATEI.")
    sizes = manifest["sizes"]
    if any(
        not math.isclose(fraction, sizes[part] / len(full_rows))
        for part, fraction in zip(PARTITIONS, fractions)
    ):
        raise ValueError(
            f"Anteile passen nicht zum festen Split ({sizes['train']}/{sizes['validation']}/{sizes['eval']}). SPLIT_DATEI=None verwendet einen freien Anteilssplit."
        )
    stats = check_assignments(full_rows, groups, manifest["assignments"], sizes)
    if stats != manifest["statistics"]:
        raise ValueError("Die dokumentierte Split-Verteilung wurde verändert.")
    group_lookup = {r["report_id"]: g for r, g in zip(full_rows, groups)}
    for part in PARTITIONS:
        data = (Path(manifest_path).parent / (part + ".jsonl")).read_bytes()
        if sha(data) != manifest["files"][part + ".jsonl"]["sha256"]:
            raise ValueError(f"Die gespeicherte Split-Datei {part}.jsonl wurde verändert.")
        if [r["report_id"] for r in map(json.loads, data.splitlines()) if r] != [
            r["report_id"] for r in full_rows if manifest["assignments"][r["report_id"]] == part
        ]:
            raise ValueError("Split-Datei und gespeicherte Mitgliedschaft unterscheiden sich.")
    return {
        part: [
            r | {"split_group": group_lookup[r["report_id"]]}
            for r in rows
            if manifest["assignments"][r["report_id"]] == part
        ]
        for part in PARTITIONS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "InputData/Reports.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "InputData/Splits")
    parser.add_argument("--eval", type=int, default=600)
    parser.add_argument("--validation", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = create(args.input, args.output, args.eval, args.validation, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Abgebrochen: {exc}\n")
    for part in PARTITIONS:
        stats = manifest["statistics"][part]
        print(
            f"{part}: {stats['reports']} Texte, {stats['groups']} getrennte Fertigungsfall-Gruppen"
        )
    print(f"Fester Split gespeichert: {args.output}")
