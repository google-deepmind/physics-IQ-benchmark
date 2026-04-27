import json
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import IPython
from scipy import stats
from calculate_iq_score_stable import ORIG_SCORE_KEY, SCORES_LIST, VERIFIED_SCORE_KEY, calculate_iq_score_update, IQTable
from pprint import pprint


BASEPATH = Path("/Users/carsten/projects/anates/physics-iq/results_share_neurips")
OUTPUT_PATH = Path("./output")


RUNKEY = "run"
EXPKEY = "exp"
N_BOOTSTRAP = 100
SEED = 1234

MODEL_NAMES = [
    "wan-i2v",
    "p-video",
    "hunyuan-video-v1",
    "sora-2",
    "sora2new",
    "grok-imagine-video"
]
FPS_KEY = "fps"
MODEL_KEY = "Model"
EVAL_KEY = "Evaluation" # verified or original
RUN_NAME_KEY = "Run Name"
RUN_KEY = "Run"
SAMPLES_KEY = "Eval Samples"
PROMPT_KEY = "Prompt"

RANKING_EVAL_SETTINGS = [
    {MODEL_KEY : "wan-i2v", FPS_KEY: 16},
    {MODEL_KEY : "sora-2", FPS_KEY: 30},
    {MODEL_KEY : "hunyuan-video-v1", FPS_KEY: 24},
    {MODEL_KEY : "p-video", FPS_KEY: 24},
    {MODEL_KEY : "grok-imagine-video", FPS_KEY: 24},
]
COMPARISON_KEYS = [{EVAL_KEY: "verified_full", PROMPT_KEY: "bpp"}, {EVAL_KEY: "original", PROMPT_KEY: "op"}]



get_prompt_from_run = lambda x: "op" if "op" in x else ("bpp" if "bpp" in x else "--")
get_eval_type_from_run = lambda x: x.split("__")[1] if "__" in x else (x.split("in_")[1] if "in_" in x else "?")
get_model_name_from_run = lambda x: max((model for model in MODEL_NAMES if model in str(x)), key=len)
# regex to extract run number, e.g. ...run_01..., run_02..., etc.
get_run_number_from_run = lambda x: int(x.split("run_")[1].split("__")[0]) if "run_" in x else -1
def get_fps(file_path)-> str | int:
    try:
        with open(file_path.parent.parent/"summary.json") as f:
            json_dict = json.load(f)
                # import pdb; pdb.set_trace()
            return json_dict[json_dict.keys().__iter__().__next__()][FPS_KEY] 
    except Exception as e:
        print(f"Could not extract FPS for {file_path.parent.parent}: {e}")
        return "?"

def get_experiment_tables(base_path: Path)-> list[IQTable]:
    exp_tables = []
    filepaths = list(base_path.glob("**/results/**/results/*.csv"))
    filepaths.sort()
    print(f"Found {len(filepaths)} CSV files in {base_path} for analysis.")
    for fp in filepaths:
        if not any(model in str(fp.stem) for model in MODEL_NAMES):
            print(f"Skipping {fp} as it does not match any known model name.")
            continue



        meta_info = {
            RUN_NAME_KEY: fp.stem,
            MODEL_KEY: get_model_name_from_run(fp.stem),
            EVAL_KEY: get_eval_type_from_run(fp.stem),
            PROMPT_KEY: get_prompt_from_run(fp.stem),
            FPS_KEY : get_fps(fp),
            RUN_KEY: get_run_number_from_run(fp.stem),
        }


        tab = IQTable.from_csv(fp, metadata=meta_info)
        exp_tables.append(tab)

    return exp_tables



def generate_latex_table(OUTPUT_PATH, pivot_df_meanstd, table_name, score_cols):
    latex_table = pd.DataFrame(
        {col: pivot_df_meanstd[col].apply(
            lambda row: f"${row['mean']:.1f} \\pm {row['std']:.1f}$", axis=1
        ) for col in score_cols}
    )
    # Replace underscores with spaces so column names and index values render in LaTeX
    latex_table.columns = [col.replace('_', ' ') for col in latex_table.columns]
    latex_table.index = pd.MultiIndex.from_tuples(
        [tuple(str(v).replace('_', ' ') for v in idx) for idx in latex_table.index],
        names=latex_table.index.names
    )
    # TODO: verify whether we need all of this filtering and if we could not move most of it towards another location.
    mask_valid = latex_table.index.get_level_values(EVAL_KEY).isin(["verified","verified full", "original"])
    latex_table = latex_table[mask_valid]
    mask_models = latex_table.index.get_level_values(MODEL_KEY).isin(MODEL_NAMES)
    latex_table = pd.concat([latex_table[mask_models], latex_table[~mask_models]])
    latex_str = latex_table.to_latex(escape=False)
    with open(OUTPUT_PATH/table_name, "w") as f:
        f.write(latex_str)

def mean_score_evaluation(scores_df):
    print("\nModel rankings based on original evaluation:")
    # original score = prompt: op, evaluation: original
    original_df = scores_df[(scores_df[PROMPT_KEY]=="op") & (scores_df[EVAL_KEY]=="original")]
    original_scores = original_df[[MODEL_KEY, ORIG_SCORE_KEY]].groupby(MODEL_KEY).mean(numeric_only=True)

    original_scores["rank"] = original_scores[ORIG_SCORE_KEY].rank(ascending=False)
    original_scores_ordered = original_scores.sort_values(by=ORIG_SCORE_KEY, ascending=False)
    # print(original_scores_ordered)
    # print(original_scores)
    # print("\nModel rankings based on verified evaluation:")
    # verified score = prompt: bpp, evaluation: verified_full
    verified_df = scores_df[(scores_df[PROMPT_KEY]=="bpp") & (scores_df[EVAL_KEY]=="verified_full")]
    verified_scores = verified_df[[MODEL_KEY, VERIFIED_SCORE_KEY]].groupby(MODEL_KEY).mean(numeric_only=True)

    verified_scores["rank"] = verified_scores[VERIFIED_SCORE_KEY].rank(ascending=False)
    verified_scores_ordered = verified_scores.sort_values(by=VERIFIED_SCORE_KEY, ascending=False)
    # print(verified_scores_ordered)
    # print(verified_scores)

    ranking_df = original_scores.join(verified_scores, lsuffix="_original", rsuffix="_verified", how="inner")
    ranking_df['rank_delta']  = ranking_df['rank_verified'] - ranking_df['rank_original']
    pprint(ranking_df)

    rho, p_rho = stats.spearmanr(ranking_df['rank_original'], ranking_df['rank_verified'])
    print(f"Spearman ρ = {rho:.3f}  (p = {p_rho:.3f})")

    # 4. Kendall's τ
    tau, p_tau = stats.kendalltau(ranking_df['rank_original'], ranking_df['rank_verified'])
    print(f"Kendall  τ = {tau:.3f}  (p = {p_tau:.3f})")

if __name__ == "__main__":
    exp_tables = get_experiment_tables(BASEPATH)
    exp_df = [tab.get_full_df() for tab in exp_tables]
    exp_df = pd.concat(exp_df, ignore_index=True).reset_index(drop=True)
    scores_df = []
    for tab in exp_tables:
        scores_df.append(tab.get_output_dict())
    scores_df = pd.DataFrame(scores_df)

    # Generate LaTeX table with mean and std for each score column, grouped by model, evaluation type, fps, and prompt
    pivot_df_meanstd = scores_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY])[[ORIG_SCORE_KEY, VERIFIED_SCORE_KEY]+SCORES_LIST].agg(['mean', 'std']) *100
    table_name = "full_model_table.tex"
    pprint(pivot_df_meanstd)
    score_cols = [VERIFIED_SCORE_KEY] + SCORES_LIST
    generate_latex_table(OUTPUT_PATH, pivot_df_meanstd, table_name, score_cols)



    # TODO: filter each model for one main setting (e.g. 30FPS)
    # TODO: filter for only specific models.

    # comparison of rankings between original and verified evaluation based on mean
    # filter so that only when the setting matches one of RANKING_EVAL_SETTINGS
    mask = pd.Series(False, index=scores_df.index)
    for setting in RANKING_EVAL_SETTINGS:
        setting_mask = pd.Series(True, index=scores_df.index)
        for key, value in setting.items():
            setting_mask &= (scores_df[key] == value)
        mask |= setting_mask
    scores_df_filtered = scores_df[mask]
    mean_score_evaluation(scores_df_filtered)


    # filter exp_tables for only the settings in RANKING_EVAL_SETTINGS
    filtered_exp_tables = []
    for setting in RANKING_EVAL_SETTINGS:
        for tab in exp_tables:
            match = True
            for key, value in setting.items():
                if tab.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered_exp_tables.append(tab)
                break # only take the first matching table for each setting
    
    # bootstrapped 
    bootstrap_sample_list = []
    for _ in range(N_BOOTSTRAP):
        bootstrap_sample_list.append(np.random.choice([1,2,3,4], size=len(exp_tables[0].df), replace=True))
    
    bootstrap_dfs = []
    # for each bootstrap sample, calculate the mean score for each model and evaluation type
    for i, bootstrap_sample in enumerate(bootstrap_sample_list):
        # create a new IQTable for each bootstrap sample by resampling the original data
        boot_iq_tables = []
        for setting, comp in product(RANKING_EVAL_SETTINGS, COMPARISON_KEYS):
            setting = setting.copy()
            setting.update(comp)
            series = []
            for tab in exp_tables:
                match = True
                for key, value in setting.items():
                    if tab.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    n_run = tab.metadata[RUN_KEY]
                    # filter the bootstrap sample for the current run number
                    mask = bootstrap_sample == n_run
                    series.append(tab.get_full_df()[mask])
            # import pdb; pdb.set_trace()
            boot_df = pd.concat(series)
            assert len(boot_df) == len(bootstrap_sample), f"Bootstrap sample length {len(bootstrap_sample)} does not match boot_df length {len(boot_df)}"
            assert boot_df.index.is_unique, "Bootstrap sample has duplicate indices and therefore is not a valid permutation based bootstrap sample from different runs."
            bootstrap_setting = setting.copy()
            bootstrap_setting["bootstrap_sample"] = i
            boot_iq_table = IQTable(boot_df, metadata=bootstrap_setting)
            boot_iq_tables.append(boot_iq_table)
        
        bootstrap_scores_df = []
        for tab in boot_iq_tables:
            bootstrap_scores_df.append(tab.get_output_dict())
        bootstrap_scores_df = pd.DataFrame(bootstrap_scores_df)
        bootstrap_dfs.append(bootstrap_scores_df)
    
    # Ranking distribution based on bootstrapped samples
    ranks_original = []
    ranks_verified = []
    ranking_dfs = []
    for bootstrap_scores_df in bootstrap_dfs:
        ranking_df = bootstrap_scores_df[[MODEL_KEY, EVAL_KEY, ORIG_SCORE_KEY, VERIFIED_SCORE_KEY]].groupby([MODEL_KEY, EVAL_KEY]).mean(numeric_only=True)
        ranking_dfs.append(ranking_df)
        ranking_df_original = ranking_df.xs("original", level=EVAL_KEY)
        ranking_df_verified = ranking_df.xs("verified_full", level=EVAL_KEY)
        ranking_df_original["rank"] = ranking_df_original[ORIG_SCORE_KEY].rank(ascending=False)
        ranking_df_verified["rank"] = ranking_df_verified[VERIFIED_SCORE_KEY].rank(ascending=False)
        ranks_original.append(ranking_df_original["rank"])
        ranks_verified.append(ranking_df_verified["rank"])

    # Calculate the distribution of Spearman's ρ and Kendall's τ across bootstrap samples
    spearman_rhos = []
    kendall_taus = []
    for ranking_df_original, ranking_df_verified in zip(ranks_original, ranks_verified):
        merged_ranking_df = ranking_df_original.to_frame().join(ranking_df_verified.to_frame(), lsuffix="_original", rsuffix="_verified", how="inner")
        rho, _ = stats.spearmanr(merged_ranking_df['rank_original'], merged_ranking_df['rank_verified'])
        tau, _ = stats.kendalltau(merged_ranking_df['rank_original'], merged_ranking_df['rank_verified'])
        spearman_rhos.append(rho)
        kendall_taus.append(tau)
    
    ci_rho = np.percentile(spearman_rhos, [2.5, 97.5])
    ci_tau = np.percentile(kendall_taus, [2.5, 97.5])
    rho_mean = np.mean(spearman_rhos)
    tau_mean = np.mean(kendall_taus)
    print(f"Mean Spearman's ρ: {rho_mean:.3f}, 95% CI: [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]")
    print(f"Mean Kendall's τ: {tau_mean:.3f}, 95% CI: [{ci_tau[0]:.3f}, {ci_tau[1]:.3f}]")

    # Create a plot of the distribution of Spearman's ρ and Kendall's τ across bootstrap samples
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(spearman_rhos, bins=20, color='skyblue', edgecolor='black')
    plt.title("Bootstrap Distribution of Spearman's ρ")
    plt.xlabel("Spearman's ρ")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(kendall_taus, bins=20, color='salmon', edgecolor='black')
    plt.title("Bootstrap Distribution of Kendall's τ")
    plt.xlabel("Kendall's τ")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH/"bootstrap_correlation_distributions.png")


    # Calculate the spearman and tau for each bootstrap sample and create a scatter plot of the ranks in original vs. verified evaluation across bootstrap samples
    plt.figure(figsize=(6, 6))
    for ranking_df_original, ranking_df_verified in zip(ranks_original, ranks_verified):
        merged_ranking_df = ranking_df_original.to_frame().join(ranking_df_verified.to_frame(), lsuffix="_original", rsuffix="_verified", how="inner")
        plt.scatter(merged_ranking_df['rank_original'], merged_ranking_df['rank_verified'], alpha=0.5)
    plt.title("Bootstrap Distribution of Model Rankings (Original vs. Verified Evaluation)")
    plt.xlabel("Rank in Original Evaluation")
    plt.ylabel("Rank in Verified Evaluation")
    plt.savefig(OUTPUT_PATH/"bootstrap_ranking_scatter.png")

    # Calculate the spearman and tau within each evaluation across bootstrap samples
    spearman_rhos_within_original = []
    spearman_rhos_within_verified = []
    kendall_taus_within_original = []
    kendall_taus_within_verified = []
    import pdb; pdb.set_trace()
    for r_df1, r_df2 in zip(ranks_original[:-1], ranks_original[1:]):
        rho, _ = stats.spearmanr(r_df1, r_df2)
        tau, _ = stats.kendalltau(r_df1, r_df2)
        spearman_rhos_within_original.append(rho)
        kendall_taus_within_original.append(tau)
    for r_df1, r_df2 in zip(ranks_verified[:-1], ranks_verified[1:]):
        rho, _ = stats.spearmanr(r_df1, r_df2)
        tau, _ = stats.kendalltau(r_df1, r_df2)
        spearman_rhos_within_verified.append(rho)
        kendall_taus_within_verified.append(tau)
    # Visualize the distribution of Spearman's ρ and Kendall's τ within each evaluation type across bootstrap samples
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(spearman_rhos_within_original, bins=20, color='lightgreen', edgecolor='black')
    plt.title("Bootstrap Distribution of Spearman's ρ (Original Evaluation)")
    plt.xlabel("Spearman's ρ")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(spearman_rhos_within_verified, bins=20, color='lightcoral', edgecolor='black')
    plt.title("Bootstrap Distribution of Spearman's ρ (Verified Evaluation)")
    plt.xlabel("Spearman's ρ")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH/"bootstrap_within_evaluation_correlation_distributions.png")


    # Calculate the distribution of ranking values within each evaluation type across bootstrap samples
    ranking_values_original = pd.concat(ranks_original, axis=1)
    ranking_values_verified = pd.concat(ranks_verified, axis=1)
    ranking_values_original.columns = [f"bootstrap_{i}" for i in range(len(ranks_original))]
    ranking_values_verified.columns = [f"bootstrap_{i}" for i in range(len(ranks_verified))]
    # Create box plots of the ranking distributions for each model and evaluation type
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ranking_values_original.boxplot()
    plt.title("Bootstrap Distribution of Model Rankings (Original Evaluation)")
    plt.xlabel("Model")
    plt.ylabel("Rank")
    plt.xticks(rotation=45)
    plt.subplot(1, 2, 2)
    ranking_values_verified.boxplot()
    plt.title("Bootstrap Distribution of Model Rankings (Verified Evaluation)")
    plt.xlabel("Model")
    plt.ylabel("Rank")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH/"bootstrap_ranking_distributions.png")





    



