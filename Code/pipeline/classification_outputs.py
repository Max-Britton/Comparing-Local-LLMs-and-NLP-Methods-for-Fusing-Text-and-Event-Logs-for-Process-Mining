from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import progress


def matrix_figure(path, matrix, labels, title, *, mean=False):
    fig = Figure(figsize=(11, 9), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    values = np.asarray(matrix)
    im = ax.imshow(values, cmap="Blues", vmin=0)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    ax.tick_params(axis="both", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha("right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}" if mean else str(int(value)),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > values.max() * 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, label="Mean reports per run" if mean else "Reports")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    fig.clear()


def save_matrix(path, matrix, labels):
    from .benchmark import atomic

    table = pd.DataFrame(matrix, index=labels, columns=labels)
    table.index.name = "true_class / predicted_class"
    atomic(path.with_suffix(".csv"), table.to_csv().encode())


def export(output, details, requested_runs, settings):
    from .benchmark import atomic, payload

    groups = defaultdict(list)
    for key, info in progress.iterate(
        list(details.items()), settings, desc="Konfusionsmatrizen speichern", unit="Matrix"
    ):
        run, method, rung = key.split("/")
        path = output / "records" / method / "confusion" / f"rung_{rung}"
        save_matrix(path, info["confusion_matrix"], info["labels"])
        atomic(path.with_suffix(".json"), payload(info))
        if rung == "all":
            state = "" if info["complete"] else " [partial run]"
            matrix_figure(
                path, info["confusion_matrix"], info["labels"], f"{method} · run {run}{state}"
            )
        groups[method, rung].append(info)
    for (method, rung), group in progress.iterate(
        list(groups.items()), settings, desc="Mittlere Konfusionsmatrizen", unit="Matrix"
    ):
        complete = [r for r in group if r["complete"]]
        if not complete:
            continue
        labels = complete[0]["labels"]
        if any(r["labels"] != labels for r in complete):
            raise ValueError("Konfusionsmatrizen haben unterschiedliche Klassenreihenfolgen.")
        matrix = np.mean([r["confusion_matrix"] for r in complete], axis=0)
        path = output / "confusion_means" / method / f"rung_{rung}"
        save_matrix(path, matrix, labels)
        atomic(
            path.with_suffix(".json"),
            payload(
                {
                    "labels": labels,
                    "confusion_matrix_mean": matrix.tolist(),
                    "completed_runs": len(complete),
                    "requested_runs": requested_runs,
                    "paper_ready": len(complete) == requested_runs,
                    "aggregation": "Arithmetic mean of counts, complete runs only; not pooled counts.",
                }
            ),
        )
        if rung == "all":
            matrix_figure(
                path,
                matrix,
                labels,
                f"{method} · mean of {len(complete)}/{requested_runs} runs",
                mean=True,
            )
