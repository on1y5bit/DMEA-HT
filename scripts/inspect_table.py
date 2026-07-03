from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Print table columns and a small preview.")
    parser.add_argument("--table", required=True)
    parser.add_argument("--nrows", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.table)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, nrows=args.nrows)
    else:
        frame = pd.read_csv(path, nrows=args.nrows)

    print("COLUMNS")
    for column in frame.columns:
        print(column)
    print("PREVIEW")
    print(frame.head(args.nrows).to_string())


if __name__ == "__main__":
    main()

