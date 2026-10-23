import argparse
import signal
import Config as c
from pipeline.experiment import run_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--durchlauf', type=int, help='Only this run; otherwise all result runs.')
    parser.add_argument('--nur-planen', action='store_true', help='Read-only plan without model calls.')
    parser.add_argument('--nur-llms', action='store_true', help='Match only LLM extractions; leave baseline matching pending.')
    args = parser.parse_args()
    folders = None if args.durchlauf is None else [c.DETAIL_ORDNER / f'run_{args.durchlauf:02d}']
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        run_matching(c, folders, plan_only=args.nur_planen, llm_only=args.nur_llms)
    except KeyboardInterrupt:
        parser.exit(130, '\nUnterbrochen. Fertige Matching-Kombinationen bleiben erhalten.\n')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'\nAbgebrochen: {exc}\n')


if __name__ == '__main__':
    main()
