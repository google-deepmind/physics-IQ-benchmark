#!/usr/bin/env python3
"""Compute mean ± std across up to 4 result CSVs for one model and write a LaTeX table.

Usage
-----
    python pivot_table_from_csvs.py run1.csv run2.csv run3.csv run4.csv --score-type verified
    python pivot_table_from_csvs.py run1.csv run2.csv run3.csv run4.csv --score-type original --save-latex ./output/table.tex
"""

import argparse
from pathlib import Path
from pprint import pprint

import pandas as pd

from physiq.calculate_iq_score_stable import (
    ORIG_SCORE_KEY,
    SCORES_LIST,
    VERIFIED_SCORE_KEY,
    VERIFIED_SCORES_LIST,
    IQTable,
)

SCORE_COLS = {
    "original": [ORIG_SCORE_KEY] + SCORES_LIST,
    "verified": [VERIFIED_SCORE_KEY] + VERIFIED_SCORES_LIST,
}

SCORE_HEADERS = {
    ORIG_SCORE_KEY: "Phys-IQ orig.",
    "score_spatial": "SP orig.",
    "score_spatiotemporal": "ST orig.",
    "score_weighted_spatial": "WS orig.",
    "score_mse": "MSE orig.",
    VERIFIED_SCORE_KEY: "Phys-IQ verified",
    "score_spatial_view": "SP verified",
    "score_spatiotemporal_view": "ST verified",
    "score_weighted_spatial_view": "WS verified",
    "score_mse_view": "MSE verified",
}


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path, metavar="CSV")
    parser.add_argument(
        "--score-type", choices=["original", "verified"], default="verified"
    )
    parser.add_argument(
        "--save-latex",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the LaTeX table to this exact file path (e.g. ./output/table.tex)",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save a one-row CSV to this exact file path (e.g. ./output/scores.csv)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name written to the 'Model' column of the CSV row",
    )
    args = parser.parse_args()

    if len(args.csv_files) > 4:
        parser.error("At most 4 CSV files are supported.")

    scores_df = pd.DataFrame(
        [IQTable.from_csv(p, metadata={}).get_output_dict() for p in args.csv_files]
    )

    score_cols = [c for c in SCORE_COLS[args.score_type] if c in scores_df.columns]
    pivot = scores_df[score_cols].agg(["mean", "std"]) * 100

    pprint(pivot)

    def _fmt(col):
        m, s = pivot.loc["mean", col], pivot.loc["std", col]
        return f"${m:.1f}$" if pd.isna(s) else f"${m:.1f} \\pm {s:.1f}$"

    table = pd.DataFrame(
        {
            "Score": [SCORE_HEADERS.get(c, c.replace("_", " ")) for c in score_cols],
            "Mean ± Std": [_fmt(c) for c in score_cols],
        }
    ).set_index("Score")

    if args.save_latex:
        args.save_latex.parent.mkdir(parents=True, exist_ok=True)
        args.save_latex.write_text(table.to_latex(escape=False))
        print(f"\nLaTeX table written to {args.save_latex}")

    if args.save_csv:
        header = SCORE_HEADERS.get
        row = {"Model": args.model_name} if args.model_name else {}
        for col in score_cols:
            label = header(col, col.replace("_", " "))
            row[f"{label} mean"] = round(pivot.loc["mean", col], 2)
            row[f"{label} std"] = round(pivot.loc["std", col], 2)
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(args.save_csv, index=False)
        print(f"\nCSV written to {args.save_csv}")

    return pivot


if __name__ == "__main__":
    main()
