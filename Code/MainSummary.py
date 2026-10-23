import argparse

import Config as c
from paper import data, summary


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    results = data.Results(c.ERGEBNIS_ORDNER)
    path, count = summary.write(results, c.ERGEBNIS_ORDNER / 'Summary.csv')
    print(f'{count} rows written to {path}')


if __name__ == '__main__':
    main()
