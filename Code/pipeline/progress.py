from tqdm.auto import tqdm


def bar(settings, *, total, desc, unit="Bericht", **kwargs):
    enabled = getattr(settings, "progress", getattr(settings, "FORTSCHRITT_ANZEIGEN", True))
    return tqdm(
        total=total, desc=desc, unit=unit, disable=not enabled, dynamic_ncols=True, **kwargs
    )


def iterate(items, settings, *, desc, total=None, unit="Bericht", describe=None, **kwargs):
    if total is None:
        total = len(items)
    with bar(settings, total=total, desc=desc, unit=unit, **kwargs) as progress:
        for item in items:
            if describe is not None:
                progress.set_postfix_str(describe(item), refresh=False)
            yield item
            progress.update(1)


def message(text):
    tqdm.write(text)
