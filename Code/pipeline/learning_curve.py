import math
import random
from collections import Counter, defaultdict


def method_id(baseline, percent):
    return baseline if percent in (None, 100) else f"{baseline}__train_{percent:03d}"


def conditions(settings):
    values = settings if isinstance(settings, dict) else vars(settings)
    return [
        {
            "method": method_id(name, percent),
            "baseline": name,
            "training_percent": percent,
            "fit_method": method_id(name.removesuffix("_time"), percent)
            if name != "regex"
            else None,
        }
        for name in values.get("extraction_baselines", [])
        + values.get("classification_baselines", [])
        for percent in ([None] if name == "regex" else values.get("training_percentages", [100]))
    ]


def nested_subsets(train, percentages, seed, balance_levels=False):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in sorted(train, key=lambda r: r["report_id"]):
        buckets[(row["true_category"], row["difficulty_level"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    classes = sorted({key[0] for key in buckets})
    rng.shuffle(classes)
    levels = {c: sorted(k[1] for k in buckets if k[0] == c) for c in classes}
    for choices in levels.values():
        rng.shuffle(choices)
    available = {k: len(v) for k, v in buckets.items()}
    totals = Counter(r["true_category"] for r in train)
    rung_totals = Counter(r["difficulty_level"] for r in train)
    rung_used = Counter()
    used, class_used, order = Counter(), Counter(), []
    for i in range(len(train)):
        category = max(
            (c for c in classes if class_used[c] < totals[c]),
            key=lambda c: (i + 1) * totals[c] / len(train) - class_used[c],
        )
        level = max(
            (
                level
                for level in levels[category]
                if used[category, level] < available[category, level]
            ),
            key=lambda level: (
                (i + 1) * rung_totals[level] / len(train) - rung_used[level]
                if balance_levels else 0,
                (class_used[category] + 1) * available[category, level] / totals[category]
                - used[category, level],
            ),
        )
        key = category, level
        order.append(buckets[key][used[key]]["report_id"])
        used[key] += 1
        class_used[category] += 1
        rung_used[level] += 1
    subsets = {}
    for percent in percentages:
        count = min(len(train), max(1, math.floor(len(train) * percent / 100 + 0.5)))
        ids = set(order[:count])
        subsets[percent] = [r for r in train if r["report_id"] in ids]
    return subsets


def prepare(folder, train, percentages, seed):
    from .benchmark import freeze, payload, rows_bytes, save_csv

    subsets = nested_subsets(train, percentages, seed, balance_levels=True)
    manifest = {
        "seed": seed,
        "pool_size": len(train),
        "policy": "Nested class-stratified training-only subsets; globally balanced difficulty counts where feasible, proportional rung coverage within classes. Same subsets across methods within a draw; separate seed per series run. No warm-start across sizes. Validation/evaluation unchanged.",
        "subsets": [],
    }
    membership = []
    for percent, rows in subsets.items():
        file = f"train_{percent:03d}.jsonl"
        freeze(folder / file, rows_bytes(rows))
        manifest["subsets"].append(
            {
                "training_percent": percent,
                "training_n": len(rows),
                "file": file,
                "classes": dict(Counter(r["true_category"] for r in rows)),
                "rungs": dict(Counter(r["difficulty_level"] for r in rows)),
                "class_level": {
                    f"{c} / {level}": n
                    for (c, level), n in sorted(
                        Counter((r["true_category"], r["difficulty_level"]) for r in rows).items()
                    )
                },
            }
        )
        membership.extend(
            {
                "training_percent": percent,
                "report_id": r["report_id"],
                "true_category": r["true_category"],
                "difficulty_level": r["difficulty_level"],
                "split_group": r["split_group"],
            }
            for r in rows
        )
    freeze(folder / "manifest.json", payload(manifest))
    save_csv(folder / "membership.csv", membership)
    return subsets


def training_directory(method):
    base, separator, suffix = method.partition("__train_")
    return base.removesuffix("_time") + (separator + suffix if separator else "")


def prepare_validation(folder, validation, percentages, seed, proportional=True):
    from .benchmark import freeze, payload, rows_bytes, save_csv

    subsets = nested_subsets(validation, percentages, seed, balance_levels=True) if proportional else {
        p: list(validation) for p in percentages}
    manifest = {"seed": seed, "pool_size": len(validation),
        "policy": "Nested class-stratified validation-only subsets; proportional difficulty coverage where sample size allows. Identical across methods. Never merged into fitting data.",
        "proportional": proportional, "subsets": []}
    membership = []
    for percent, rows in subsets.items():
        name = f"validation_{percent:03d}.jsonl"
        freeze(folder / name, rows_bytes(rows))
        manifest["subsets"].append({"training_percent": percent,
            "validation_percent": percent if proportional else 100,
            "validation_n": len(rows), "file": name,
            "classes": dict(Counter(r["true_category"] for r in rows)),
            "rungs": dict(Counter(r["difficulty_level"] for r in rows))})
        membership.extend({"training_percent": percent, "report_id": r["report_id"],
            "true_category": r["true_category"], "difficulty_level": r["difficulty_level"],
            "split_group": r["split_group"]} for r in rows)
    freeze(folder / "manifest.json", payload(manifest))
    save_csv(folder / "membership.csv", membership)
    return subsets
