from analysis import (
    get_experiment_tables,
    BASEPATH,
    OUTPUT_PATH,
    N_BOOTSTRAP,
    SEED,
    parse_args,
    ORIG_SCORE_KEY,
    VERIFIED_SCORE_KEY,
    SCORES_LIST,
    _subdir,
)
from calculate_iq_score_stable import IQTable
import pandas as pd
import argparse
from pathlib import Path
from pprint import pprint

evaluations = [
    {
        "Data": "Original",
        "Eval": "Verified",
        "Path": "experiments/results/exp1_full_30fps/results/exp1a_original_in_verified.csv",
    },
    {
        "Data": "Verified",
        "Eval": "Original",
        "Path": "experiments/results/exp1_full_30fps/results/exp1a_verified_in_original.csv",
    },
    {
        "Data": "Original",
        "Eval": "Original",
        "Path": "experiments/results/exp0_original_in_original_30fps/results/exp0_original_in_original.csv",
    },
    {
        "Data": "Verified",
        "Eval": "Verified",
        "Path": "experiments/results/exp1_full_30fps/results/exp1b_verified_in_verified.csv",
    },
]

for ev in evaluations:
    ev["Path"] = BASEPATH / ev["Path"]
    ev["FPS"] = 30


def generate_latex_table_gt(
    output_df: pd.DataFrame,
    output_path: Path,
    table_name: str,
    score_cols: list,
):
    """Write a LaTeX table for deterministic GT runs (one value per cell, no std).

    Rows are indexed by (Data, Eval) — the two dimensions of the GT evaluation
    matrix.  Each cell is formatted as ``$X.X$`` (scores multiplied by 100).
    """
    table = output_df.set_index(["Data", "Eval"])[score_cols] * 100
    latex_table = table.apply(lambda col: col.map(lambda v: f"${v:.1f}$"))
    latex_table.columns = [col.replace("_", " ") for col in latex_table.columns]
    latex_str = latex_table.to_latex(escape=False)
    with open(_subdir(output_path, "tables") / table_name, "w") as f:
        f.write(latex_str)


def main():
    args = parse_args()
    output_dicts = []
    for e in evaluations:
        print(f"Processing {e['Data']} in {e['Eval']} from {e['Path']}...")
        metadata = {"Data": e["Data"], "Eval": e["Eval"], "FPS": e["FPS"]}
        iq_table = IQTable.from_csv(e["Path"], metadata=metadata)
        output_dict = iq_table.get_output_dict()
        output_dicts.append(output_dict)
    output_df = pd.DataFrame(output_dicts)
    pprint(output_df)

    score_cols = [VERIFIED_SCORE_KEY] + SCORES_LIST
    generate_latex_table_gt(output_df, args.output_dir, "gt_runs_table.tex", score_cols)


if __name__ == "__main__":
    main()
