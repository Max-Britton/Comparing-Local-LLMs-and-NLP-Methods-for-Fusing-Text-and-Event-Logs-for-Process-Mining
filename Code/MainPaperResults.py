import argparse

import Config as c
from paper import analysis, data, figures, tables


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    results = data.Results(c.ERGEBNIS_ORDNER)
    out = c.PAPER_ORDNER
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    print(f'{len(results.runs)} runs read.', flush=True)

    (out / 'tables/LLMs_and_configurations.tex').write_text(tables.models_table(results.models()))
    classification = analysis.classification_table(results)
    (out / 'tables/Classification_performance.tex').write_text(tables.classification_table(*classification))
    extraction = analysis.extraction_table(results)
    (out / 'tables/Extraction_accuracy.tex').write_text(tables.extraction_table(*extraction))

    figure_rows = [{'name': x['name'], 'group': x['group'], 'overall': x['best'], 'levels': x['levels']}
                   for x in extraction[0]]
    figure_rows += [{'name': b['name'], 'group': figures.BASELINE_GROUP, 'overall': b['best'], 'levels': b['levels']}
                    for b in extraction[1]]
    figures.difficulty_figure(figure_rows, 'Correctly extracted reports (%)', out / 'figures/Extraction.pdf')
    figures.difficulty_figure(analysis.matching_figure(results), 'Correctly matched reports (%)', out / 'figures/Matching.pdf')
    runtime = analysis.runtime_figure(results)
    figures.runtime_figure(runtime, out / 'figures/Runtime.pdf')

    print('Reading report-level results ...', flush=True)
    events = data.read_events(c.EVENTLOG_DATEI)
    best_cls = [data.llm_method(k, results.best_shots('classification', k)) for k, _, _ in data.LLMS]
    best_ext = [data.llm_method(k, results.best_matching_shots(k)) for k, _, _ in data.LLMS]
    runs = analysis.report_level(results, events, best_cls + [k for k, _ in data.CLASSIFICATION_BASELINES],
                                 best_ext + [k for k, _ in data.EXTRACTION_BASELINES])
    pipeline, _, _ = analysis.pipeline_table(results, runs)
    (out / 'tables/Pipeline.tex').write_text(tables.pipeline_table(pipeline))

    print(f'Written to {out}')


if __name__ == '__main__':
    main()
