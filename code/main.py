import json
from pathlib import Path
import pandas as pd
import numpy as np
import IPython
from scipy import stats
from calculate_iq_score_stable import ORIG_SCORE_KEY, SCORES_LIST, VERIFIED_SCORE_KEY, calculate_iq_score_update, IQTable
from pprint import pprint


BASEPATH = Path("/Users/carsten/projects/anates/physics-iq/results_share_neurips")

# artifact_file = BASEPATH / "experiments/results/simulations/benchmark_artifact_analysis/results/artifact_pair_variance_impact.csv"

subpaths = [
    # 30FPS version
    "experiments/results/exp1_full_30fps/results/exp1a_verified_in_original.csv",
    "experiments/results/exp1_full_30fps/results/exp1a_original_in_verified.csv",
    # 24FPS version
    "experiments/results/exp1_full_24fps/results/exp1a_verified_in_original.csv",
    "experiments/results/exp1_full_24fps/results/exp1a_original_in_verified.csv",
]

USE_SUBPATHS = False



RUNKEY = "run"
EXPKEY = "exp"

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
RUN_NUM = "Run Number"
SAMPLES_KEY = "Eval Samples"
PROMPT_KEY = "Prompt"



if __name__ == "__main__":
    iq_score_info = []
    iq_tables = []
    filepaths = [BASEPATH / subpath for subpath in subpaths] if USE_SUBPATHS else list(BASEPATH.glob("**/**/**/**/*.csv"))
    filepaths.sort()
    # Start by computing the physics-iq score for each file on the entire set.
    for file_path in filepaths:
        if "simulations" in str(file_path):
            continue
        # check if one model name is in the file path, if not, skip the file
        if not any(model in str(file_path.parent.parent.stem) for model in MODEL_NAMES):
            print(f"Skipping {file_path} as it does not match any known model name.")
            continue

        # Model name is the longest matching name from MODEL_NAMES that is in the file path
        model_name = max((model for model in MODEL_NAMES if model in str(file_path)), key=len)
        print(f"Processing {file_path} for model {model_name}...")

        iq_table = IQTable.from_csv(file_path)
        out_dict = iq_table.get_output_dict()
        # out_dict = calculate_iq_score_update(file_path)

        out_dict[RUNKEY] = file_path.stem
        out_dict[EXPKEY] = file_path.parent.parent.stem
        out_dict[MODEL_KEY] = model_name
        try:
            with open(file_path.parent.parent/"summary.json") as f:
                json_dict = json.load(f)
                # import pdb; pdb.set_trace()
                out_dict[FPS_KEY] = json_dict[json_dict.keys().__iter__().__next__()][FPS_KEY] 
        except Exception as e:
            print(f"Could not extract FPS for {file_path.parent.parent}: {e}")
            out_dict[FPS_KEY] = "?"
                
        iq_score_info.append(out_dict.copy())
    ana_df = pd.DataFrame(iq_score_info)

    pprint("Computation of physics-iq scores over the entire dataset:")
    pprint(
        ana_df[[EXPKEY,RUNKEY, FPS_KEY, ORIG_SCORE_KEY, VERIFIED_SCORE_KEY] + SCORES_LIST ]
    )
    pprint("\n\n")

    # import IPython; IPython.embed()
    # assert False

    # pprint("Computation of physics-iq scores but now only over the subset of samples that was changed.")
    # for file_path in filepaths:
    #     if "simulations" in str(file_path):
    #         continue

    #     out_dict = calculate_iq_score_update(file_path)
    #     out_dict[RUNKEY] = file_path.stem
    #     out_dict[EXPKEY] = file_path.parent.parent.stem
    #     iq_score_info.append(out_dict.copy())
    # ana_df = pd.DataFrame(iq_score_info)

    # Parse RUNKEY to extract model name, evaluation type, and run number
    # ana_df[MODEL_KEY] = ana_df[RUNKEY].apply(lambda x: next((m for m in MODEL_NAMES if m in x), "?"))

    # This is only required for GT evaluations
    # If model name is not found, try to extract it from the run key by splitting on "in_" and taking the first part, then splitting on "_" and taking the second part
    # ana_df[MODEL_KEY] = ana_df.apply(lambda row: row[MODEL_KEY] if row[MODEL_KEY] != "?" else (row[RUNKEY].split("_in_")[0].split("_")[1]+" GT" if "_in_" in row[RUNKEY] else "?"), axis=1)
    ana_df[EVAL_KEY] = ana_df[RUNKEY].apply(lambda x: x.split("__")[1] if "__" in x else x.split("in_")[1] if "in_" in x else "?")
    # ana_df[SAMPLES_KEY] = ana_df[RUNKEY].apply(lambda x: "changed_samples" if "verified_in_original" in x else "original_samples")
    ana_df[PROMPT_KEY] = ana_df[RUNKEY].apply(lambda x: "op" if "op" in x else ("bpp" if "bpp" in x else "--"))

    pprint(ana_df[[EXPKEY, MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY, RUNKEY, ORIG_SCORE_KEY, VERIFIED_SCORE_KEY] + SCORES_LIST])

    # if True:
    col = VERIFIED_SCORE_KEY
    prompt_conditions = ["bpp", "op"]

    df_g1 = ana_df[ana_df["Prompt"] == prompt_conditions[0]]
    df_g2 = ana_df[ana_df["Prompt"] == prompt_conditions[1]]

    df_g1[f"{col}_centered"] = df_g1.groupby("Model")[col].transform(lambda x: x - x.mean())
    df_g2[f"{col}_centered"] = df_g2.groupby("Model")[col].transform(lambda x: x - x.mean())


    # Filter to only bpp/op prompts
    df_sub = ana_df[ana_df["Prompt"].isin(prompt_conditions)].copy()

    # Subtract each model's overall mean for this score (removes between-model mean differences)
    df_sub[f"{col}_centered"] = df_sub.groupby(["Model"])[col].transform(lambda x: x - x.mean())

    # Per-model SD per prompt condition
    sd_table = (
        df_sub.groupby(["Model", "Prompt"])[col]
        .std()
        .unstack("Prompt")
    )
    print(sd_table)

    # Levene's test: bpp vs op on centered residuals
    groups = [
        df_sub.loc[df_sub["Prompt"] == cond, f"{col}_centered"].dropna()
        for cond in prompt_conditions
    ]
    # Levene's test on centered residuals
    stat, p = stats.levene(*groups)
    print(f"Levene's test (bpp vs op): W={stat:.3f}, p={p:.4f}")
    # Fligner's-killeen test
    stat, p = stats.fligner(*groups)
    print(f"Fligner-Killeen test (bpp vs op): stat={stat:.3f}, p={p:.4f}")

    

    pprint("\n\nCompute with unormalized values:")
    # Levene's test: bpp vs op on centered residuals
    groups = [
        df_sub.loc[df_sub["Prompt"] == cond, f"{col}"].dropna()
        for cond in prompt_conditions
    ]
    # Levene's test on centered residuals
    stat, p = stats.levene(*groups)
    print(f"Levene's test (bpp vs op): W={stat:.3f}, p={p:.4f}")
    # Fligner's-killeen test
    stat, p = stats.fligner(*groups)
    print(f"Fligner-Killeen test (bpp vs op): stat={stat:.3f}, p={p:.4f}")


    # # Paired dot plot
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(5, 4))
    # for model, row in sd_table.iterrows():
    #     ax.plot(prompt_conditions, row[prompt_conditions].values, marker="o", label=model, alpha=0.7)

    # ax.set_ylabel(f"SD of {col}")
    # ax.set_xlabel("Prompt condition")
    # ax.set_title(f"Per-model spread: bpp vs op")
    # ax.legend(fontsize=8)
    # plt.tight_layout()
    # plt.show()

    # assert False


    pprint(ana_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY]).mean(numeric_only=True))


    pivot_df_describe = ana_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY])[[ORIG_SCORE_KEY, VERIFIED_SCORE_KEY]+SCORES_LIST].describe()

    # Assess for each model and evaluation type whether the STD significantly changes between the original and verified evaluation


    pivot_df_meanstd = ana_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY])[[ORIG_SCORE_KEY, VERIFIED_SCORE_KEY]+SCORES_LIST].agg(['mean', 'std']) *100



    pprint(pivot_df_meanstd)
    # Create a latex table mean \pm std for each model and evaluation type
    score_cols = [VERIFIED_SCORE_KEY] + SCORES_LIST
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
    if False: 
        # Sort rows: verified before original within each model/prompt group
        latex_table = latex_table.reset_index()
        latex_table[EVAL_KEY] = pd.Categorical(latex_table[EVAL_KEY], categories=["verified", "original"], ordered=True)
        latex_table = latex_table.sort_values([MODEL_KEY, EVAL_KEY, PROMPT_KEY]).set_index([MODEL_KEY, EVAL_KEY, PROMPT_KEY])
        # Add light gray background to verified rows (\usepackage[table]{xcolor} required)
        row_evals = [str(idx[1]) for idx in latex_table.index]
        raw_latex = latex_table.to_latex(escape=False)
        lines = raw_latex.split('\n')
        new_lines = []
        data_row_idx = 0
        in_data = False
        for line in lines:
            if '\\midrule' in line:
                in_data = True
            elif '\\bottomrule' in line:
                in_data = False
            if in_data and '&' in line:
                if data_row_idx < len(row_evals) and row_evals[data_row_idx] == 'verified':
                    line = '\\rowcolor{gray!20} ' + line
                data_row_idx += 1
            new_lines.append(line)
        latex_str = '\n'.join(new_lines)
    elif False:
        # No changes to ordering or filtering
        latex_str = latex_table.to_latex(escape=False)
    else:
        # Filter to only include verified and original rows, and order verified before original within each model/prompt group
        mask_valid = latex_table.index.get_level_values(EVAL_KEY).isin(["verified","verified full", "original"])
        latex_table = latex_table[mask_valid]
        mask_models = latex_table.index.get_level_values(MODEL_KEY).isin(MODEL_NAMES)
        latex_table = pd.concat([latex_table[mask_models], latex_table[~mask_models]])
        latex_str = latex_table.to_latex(escape=False)

    # save latex table to file
    with open("latex_table.tex", "w") as f:
        f.write(latex_str)
    # pprint("\n\nLatex table with mean and std for each model and evaluation type:")
    # print(latex_str)



