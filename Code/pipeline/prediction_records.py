def key(record):
    return record["method"], record["task"], record["report_id"]


def completed(records):
    selected, seen = [], set()
    for record in records:
        identity = (record.get("run"), *key(record))
        if identity in seen:
            raise ValueError("Weitere Anfrage nach bereits abgeschlossener Vorhersage gespeichert.")
        if not record.get("api_error"):
            selected.append(record)
            seen.add(identity)
    return selected
