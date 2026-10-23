def pass_rows(summary):
    methods = summary.get("by_method", {})
    groups = [
        ("Pass 1: Filter", ("single_candidate",)),
        ("Pass 2: Ähnlichkeit", ("cosine_direct", "attribute_direct")),
        ("Pass 3: LLM", ("llm_choice",)),
    ]
    rows = []
    for label, names in groups:
        assigned = sum(methods.get(name, {}).get("assigned", 0) for name in names)
        correct = sum(methods.get(name, {}).get("correct", 0) for name in names)
        rows.append(
            {
                "pass": label,
                "assigned": assigned,
                "correct": correct,
                "wrong": assigned - correct,
                "precision": correct / assigned if assigned else None,
            }
        )
    return rows


def print_pass_summary(summary):
    rows = pass_rows(summary)
    print("\nZUORDNUNGEN PRO MATCHING-PASS")
    print("Pass                    | Zugeordnet | Richtig | Falsch | Richtig/zugeordnet")
    print("------------------------+------------+---------+--------+-------------------")
    for row in rows:
        precision = f"{row['precision']:.1%}" if row["precision"] is not None else "–"
        print(
            f"{row['pass']:<23} | {row['assigned']:>10} | {row['correct']:>7} | "
            f"{row['wrong']:>6} | {precision:>18}"
        )
    print(f"Ohne Zuordnung: {summary['unassigned']}/{summary['total']}.")
    print(
        "Zuordnungsquote = zugeordnete Berichte / alle Berichte; zählt richtige und falsche Zuordnungen."
    )
