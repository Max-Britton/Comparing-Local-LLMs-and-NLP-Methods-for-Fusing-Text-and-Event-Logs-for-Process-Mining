import argparse
import Config as c
from pipeline.audit import check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        check(c)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(1,f'Prüfung fehlgeschlagen: {exc}\n')


if __name__=='__main__':
    main()
