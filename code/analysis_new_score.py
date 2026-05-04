from calculate_iq_score_stable import IQTable
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
    get_experiment_tables,
)
import pandas as pd
import argparse
from pathlib import Path
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

# def cauch_
out_dir = Path(OUTPUT_PATH) / "score_analysis"
out_dir.mkdir(parents=True, exist_ok=True)


def plot_df_hist(df, cols, name):
    for col in cols:
        # print(f"Plotting {col}...")
        sns.histplot(df[col])
        plt.title(f"Histogram of {col}")
        plt.savefig(out_dir / f"{name}_{col}_histogram.png")
        plt.clf()


if __name__ == "__main__":
    args = parse_args()
    exp_tables = get_experiment_tables(args.results_dir)

    tab_orig = exp_tables[0]
    tab_verified = exp_tables[16]

    pprint(f"Original score distribution:")
    pprint(tab_orig.df[tab_orig.variance_keys].describe())
    pprint(f"Verified score distribution:")
    pprint(tab_verified.df[tab_verified.variance_keys].describe())
    # print(f"Plotting {name} score distributions...")
    plot_df_hist(tab_orig.df, tab_orig.variance_keys, "original")
    plot_df_hist(tab_verified.df, tab_verified.variance_keys, "verified")

    # for score in tab_orig.metric_keys:
    #     sns.histplot(
    #         tab_orig.compute_metric_ratio(score),
    #         label="Original",
    #         color="blue",
    #         alpha=0.5,
    #     )
    #     sns.histplot(
    #         tab_verified.compute_metric_ratio(score),
    #         label="Verified",
    #         color="orange",
    #         alpha=0.5,
    #     )
    #     plt.title(f"Histogram of {score}")
    #     plt.legend()
    #     plt.savefig(f"{score}_comparison_histogram.png")
    #     plt.clf()
    #     print("\n")
    #     print("-" * 20)
    #     print("Score Descriptiontion:")
    #     print(f"{score} - Original - Sample:")
    #     print(tab_orig.compute_metric_scores_scenario(score).describe())
    #     print(f"{score} - Original - PhysIQ:")
    #     print(tab_orig.get_score(score))

    #     print(f"{score} - Verified - Sample:")
    #     print(tab_verified.compute_metric_scores_scenario(score).describe())
    #     print(f"{score} - Verified - PhysIQ:")
    #     print(tab_verified.get_score(score))

    # Full score analysis
    tables = {
        "Original": tab_orig,
        "Verified": tab_verified,
    }
    for name, table in tables.items():
        print(f"{name} score distribution:")
        df_scores = table.compute_scores_scenario()
        print(df_scores.describe())
        print(f"{name} original phys-iq score:")
        print(table.compute_final_score_orig())
        plot_df_hist(df_scores, df_scores.columns, name.lower())

        df_scores = table.compute_scores_scenario_by_view()
        print(f"{name} - score distribution by view:")
        print(df_scores.describe())
        plot_df_hist(df_scores, df_scores.columns, name.lower() + "-sample")
