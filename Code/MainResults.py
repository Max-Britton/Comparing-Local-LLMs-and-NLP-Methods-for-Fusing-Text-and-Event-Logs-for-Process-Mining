import Config as c
from pipeline.csv_results import finalize
from pipeline.experiment import run_complete, run_directories
from pipeline.export import export_run


def main():
    folders = run_directories(c)
    if not folders:
        print('No working data; all runs are stored as CSV in', c.ERGEBNIS_ORDNER)
    for folder in folders:
        if run_complete(c, folder):
            print(finalize(c, folder))
        else:
            print(export_run(c, folder), '(run not complete yet; working data kept)')


if __name__ == '__main__':
    main()
