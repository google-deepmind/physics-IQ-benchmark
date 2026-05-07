from calculate_iq_score_stable import IQTable, clip
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
import numpy as np

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


def unfold_perspective_columns(df, metrics: list[str], keep: list[str], views=('left', 'center', 'right')):
    """
    Melt multiple metric groups into long format, preserving specified columns.
    
    Each metric in `metrics` must have columns named:
        {metric}_perspective-left, {metric}_perspective-center, {metric}_perspective-right
    
    Returns a dataframe with the kept columns, a 'view' column, and one column per metric.
    """
    view_dfs = []
    for view in views:
        chunk = df[keep + [f'{m}_perspective-{view}' for m in metrics]].copy()
        chunk.columns = keep + metrics
        chunk['view'] = view
        view_dfs.append(chunk)

    df_long = pd.concat(view_dfs, ignore_index=True)
    df_long['view'] = pd.Categorical(df_long['view'], categories=list(views), ordered=True)
    return df_long.sort_values(keep + ['view']).reset_index(drop=True)


def get_sample_level_df(tab_orig: IQTable):
    unfold_views_df = tab_orig.df.copy()
    unfold_views_df = unfold_perspective_columns(unfold_views_df, keep=["scenario"], metrics=tab_orig.metric_keys+tab_orig.variance_keys)
    unfold_views_df["scenario"] = unfold_views_df["scenario"].map(lambda s: s.replace(".mp4", ""))  # Keep only the scenario name, remove perspective suffix
    for col in tab_orig.get_list_keys():
        unfold_views_df[col+"_temporal"] = unfold_views_df[col]
        unfold_views_df[col] =unfold_views_df[col].map(lambda x: np.mean(x))


    scores_raw_cols = [m+"-score_raw" for m in tab_orig.metric_keys]
    for m, v in zip(tab_orig.metric_keys, tab_orig.variance_keys):
        if "MSE" in m:
            unfold_views_df[m+"-score_raw"] = unfold_views_df[v] / (unfold_views_df[m] + tab_orig.ratio_eps)
        else:
            unfold_views_df[m+"-score_raw"] = unfold_views_df[m] / (unfold_views_df[v] + tab_orig.ratio_eps)
        unfold_views_df[m+"-score"] = clip(unfold_views_df[m+"-score_raw"])

    
    phys_iq_col = "Physics-IQ verified score"
    unfold_views_df[phys_iq_col] = unfold_views_df[[ m+"-score" for m in tab_orig.metric_keys]].mean(axis=1)
    scores_cols = [phys_iq_col]+[m+"-score" for m in tab_orig.metric_keys]
    id_col = "ID"
    sample_name_cols = ["scenario", "view"]
    sample_index_cols = [id_col] + sample_name_cols
    # enumerate from 0 to len(unfold_views_df)-1
    unfold_views_df[id_col] = [i for i in range(len(unfold_views_df))]
    for k in tab_orig.metadata:
        unfold_views_df[k] = tab_orig.metadata[k]

    col_dict = {
        "id_col": id_col,   
        "sample_name_cols": sample_name_cols,
        "sample_index_cols": sample_index_cols,
        "scores_cols": scores_cols,
        "phys_iq_col": phys_iq_col,
        "sores_raw_cols": scores_raw_cols,
        "metric_cols" : tab_orig.metric_keys,
        "variance_cols" : tab_orig.variance_keys,
        "meta_cols": list(tab_orig.metadata.keys()),    
    }

    return unfold_views_df, col_dict

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

    unfold_views_df, col_dict = get_sample_level_df(tab_orig)

    vis_df = unfold_views_df[col_dict["sample_index_cols"] + col_dict["scores_cols"]]

    pprint(vis_df[scores_cols].describe())
    pprint(vis_df)



    # pprint(unfold_views_df)

