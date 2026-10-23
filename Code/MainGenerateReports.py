import argparse
import signal

import Config as c
from generation import corpus, harness


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pruefen', action='store_true', help='Compare the planned prompts with InputData/Reports.jsonl.')
    parser.add_argument('--nur-planen', action='store_true', help='Write the planned prompts without model calls.')
    args = parser.parse_args()

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        briefs = corpus.plan(c.GENERIERUNG_EREIGNISSE, c.GENERIERUNG_STAPEL, c.GENERIERUNG_SEED)
        print(f'{len(briefs)} reports planned.', flush=True)
        if args.pruefen:
            same, total = corpus.compare(briefs, c.TEXT_DATEI)
            print(f'{same} of {len(briefs)} prompts are identical to {c.TEXT_DATEI.name} ({total} reports).')
            if same != len(briefs):
                parser.exit(1)
        elif args.nur_planen:
            path = c.GENERIERUNG_AUSGABE.with_name('Prompts.jsonl')
            corpus.write_prompts(briefs, path, c.GENERIERUNG_MODELL, c.GENERIERUNG_SEED)
            print(f'Prompts: {path}')
        else:
            path = corpus.generate(briefs, c.GENERIERUNG_AUSGABE, c.OLLAMA_HOST, c.GENERIERUNG_MODELL,
                                   c.GENERIERUNG_SEED, c.TIMEOUT_SEKUNDEN)
            print(f'Reports: {path}')
            if c.GENERIERUNG_HARNESS:
                notes = harness.annotate(corpus.read_reports(path), path.with_name('Harness.jsonl'), c.OLLAMA_HOST,
                                         c.GENERIERUNG_MODELL, c.TIMEOUT_SEKUNDEN)
                print(f'Harness annotations: {notes}')
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted. Start again to continue; stored reports are kept.\n')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'\nAbgebrochen: {exc}\n')


if __name__ == '__main__':
    main()
