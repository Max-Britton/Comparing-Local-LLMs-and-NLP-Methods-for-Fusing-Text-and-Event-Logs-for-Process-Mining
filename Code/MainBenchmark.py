import argparse
import os
import signal
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import Config as c
from pipeline.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--durchlauf', type=int, help='Resume the specified run; otherwise resume the active or create the next run.')
    parser.add_argument('--nur-planen', action='store_true', help='Read-only plan without model calls.')
    args = parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        run(c, requested=args.durchlauf, plan_only=args.nur_planen)
    except KeyboardInterrupt:
        parser.exit(130, '\nUnterbrochen. Erneut starten, um den Durchlauf fortzusetzen.\n')
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'\nAbgebrochen: {exc}\n')


if __name__ == '__main__':
    main()
