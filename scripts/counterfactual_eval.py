from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    preds = pd.read_csv(args.predictions)
    rows = []
    for field in ["n_images", "n_visits", "has_bio", "bio_missing_count", "report_length"]:
        if field not in preds.columns:
            continue
        corr = preds[["prob", field]].apply(pd.to_numeric, errors="coerce").corr(method="spearman").iloc[0, 1]
        rows.append({"perturbation_proxy": field, "spearman_prob": corr})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
