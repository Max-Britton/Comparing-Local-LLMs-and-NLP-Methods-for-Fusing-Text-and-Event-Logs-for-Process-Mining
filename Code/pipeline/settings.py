def matching_settings(c):
    from .matching import Settings

    result = Settings(
        cosine_threshold=c.KOSINUS_SCHWELLE,
        minimum_margin=c.MINDESTABSTAND,
        time_margin_seconds=c.ZEITFENSTER_SEKUNDEN,
        top_k=c.TOP_K,
        minimum_direct_attributes=c.MIN_ATTRIBUTE_DIREKT,
        pass1_normalize=c.NORMALISIERUNG_PASS1,
        score_mode=c.VERGLEICH,
        hybrid_threshold=c.HYBRID_SCHWELLE,
        activity_weight=c.GEWICHT_AKTIVITAET,
        machine_weight=c.GEWICHT_MASCHINE,
        date_weight=c.GEWICHT_DATUM,
        time_weight=c.GEWICHT_UHRZEIT,
        worker_weight=c.GEWICHT_WORKER,
        time_scale_seconds=c.ZEIT_HALBWERT_MINUTEN * 60,
    )
    result.validate()
    return result
