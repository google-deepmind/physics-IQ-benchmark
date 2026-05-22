"""Statistical analysis and visualisation for Physics-IQ benchmark results.

Loads per-run IQTable objects from a results directory, computes point-estimate
rankings and aggregated scores, then runs a bootstrap analysis to assess how
stable the model rankings are across evaluation protocols (original vs. verified).

Typical usage
-------------
    python analysis.py --results-dir /path/to/results --output-dir ./output

Output directory will contain four publication-quality figures (PDF + PNG),
a LaTeX score table, and five CSV/JSON result files.  See ``save_results`` for
the full list.

Configuration
-------------
Edit ``MODEL_NAMES`` to add new models and ``RANKING_EVAL_SETTINGS`` to specify
which FPS setting to use per model in the ranking comparison.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from itertools import product
from pprint import pprint
from scipy import stats

from calculate_iq_score_stable import (
    ORIG_SCORE_KEY,
    SCORES_LIST,
    VERIFIED_SCORE_KEY,
    VERIFIED_SCORES_LIST,
    IQTable,
)
from plot_settings import model_to_color, model_to_plotting_name


@dataclass(frozen=True)
class EvalCondition:
    """Names the three things that vary between comparison arms.

    ``eval_type`` and ``prompt`` are used to filter ``scores_df`` rows.
    ``score_key`` is the column used for ranking / slope chart y-axis.
    ``label`` is the human-readable x-axis tick label.
    """

    eval_type: str  # "original" | "verified_full"
    prompt: str  # "op" | "bpp"
    score_key: str  # ORIG_SCORE_KEY | VERIFIED_SCORE_KEY
    label: str  # human-readable, e.g. "Original (op)"

    def filter(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Return rows of *df* matching this condition."""
        return df[(df[EVAL_KEY] == self.eval_type) & (df[PROMPT_KEY] == self.prompt)]


@dataclass(frozen=True)
class Comparison:
    """Declarative specification of one comparison to run in the analysis pipeline.

    Boolean flags determine which analyses are executed for this pair of conditions.
    ``subscores`` holds only the *delta-panel* columns (not the final-score key,
    which lives in ``condition_a.score_key``).  ``save_summary`` marks the one
    primary comparison whose ranking / bootstrap results feed ``save_results()``.
    """

    name:              str
    condition_a:       EvalCondition
    condition_b:       EvalCondition
    label_a:         str | None = None  # custom label for condition A in stats output (defaults to condition_a.label)
    label_b:         str | None = None  # custom label for condition B in stats output (defaults to condition_b.label)
    run_stats:         bool = False   # compare_conditions_stats → <stem>_stats.csv
    run_impact_plot:   bool = False   # _plot_slope_delta
    run_ranking:       bool = False   # mean_score_evaluation
    run_bootstrap:     bool = False   # run_bootstrap_analysis
    subscores:         "tuple[str, ...] | None" = None
    stats_alternative: str = "two-sided"
    save_summary:      bool = False   # feeds save_results() at end of main()
    delta_label:      str | None = None  # custom label for delta plots

    @property
    def stem(self) -> str:
        """Filesystem-safe output-file prefix derived from name."""
        import re
        return re.sub(r"[^\w]+", "_", self.name.lower()).strip("_")


# Publication-quality rcParams (NeurIPS style)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

# Semantic colors used consistently across figures
_C_RHO = "#2166AC"  # Spearman ρ (blue)
_C_TAU = "#D6604D"  # Kendall τ  (orange-red)
_C_ORIG = "#1B7837"  # original evaluation (green)
_C_VER = "#762A83"  # verified evaluation (purple)
_C_CROSS = "#636363"  # cross-evaluation comparisons (neutral grey)


BASEPATH = Path(".")
OUTPUT_PATH = Path("./output")

RUNKEY = "run"
EXPKEY = "exp"
N_BOOTSTRAP = 500
SEED = 1234

MODEL_NAMES = [
    "wan-i2v",
    "p-video",
    "hunyuan-video-v1",
    "sora-2",
    "sora2new",
    "sora2october",
    "grok-imagine-video",
]
SORA2_MODELS = ["sora-2", "sora2new", "sora2october"]
FPS_KEY = "fps"
MODEL_KEY = "Model"
EVAL_KEY = "Evaluation"
RUN_NAME_KEY = "Run Name"
RUN_KEY = "Run"
SAMPLES_KEY = "Eval Samples"
PROMPT_KEY = "Prompt"

RANKING_EVAL_SETTINGS = [
    {MODEL_KEY: "wan-i2v", FPS_KEY: 16},
    {MODEL_KEY: "sora-2", FPS_KEY: 30},
    {MODEL_KEY: "hunyuan-video-v1", FPS_KEY: 24},
    {MODEL_KEY: "p-video", FPS_KEY: 24},
    {MODEL_KEY: "grok-imagine-video", FPS_KEY: 24},
]
COMPARISON_KEYS = [
    {EVAL_KEY: "verified_full", PROMPT_KEY: "bpp"},
    {EVAL_KEY: "original", PROMPT_KEY: "op"},
]

SCORES_COLS_ORIG = [ORIG_SCORE_KEY] + SCORES_LIST
SCORES_COLS_VERIFIED = [VERIFIED_SCORE_KEY] + VERIFIED_SCORES_LIST
SCORE_COLS_FULL = []
for ver_col, orig_col in zip(SCORES_COLS_VERIFIED, SCORES_COLS_ORIG):
    SCORE_COLS_FULL.extend([orig_col, ver_col])
SCORES_COLS_DICT = {
    "original-score" : SCORES_COLS_ORIG,
    "verified-score" : SCORES_COLS_VERIFIED,
    "full-score" : SCORE_COLS_FULL,
}

# Canonical condition instances used throughout the analysis
CONDITION_ORIGINAL = EvalCondition("original", "op", ORIG_SCORE_KEY, "Original (op)")
CONDITION_ORIGINAL_BPP = EvalCondition(
    "original", "bpp", ORIG_SCORE_KEY, "Original (bpp)"
)
CONDITION_ORIGINALGT_OP_VERIFIED_SCORE = EvalCondition(
    "original", "op", VERIFIED_SCORE_KEY, "Original (op) verified score"
)
CONDITION_ORIGINALGT_BPP_VERIFIED_SCORE = EvalCondition(
    "original", "bpp", VERIFIED_SCORE_KEY, "Original (bpp) verified score"
)
CONDITION_VERIFIED = EvalCondition(
    "verified_full", "bpp", VERIFIED_SCORE_KEY, "Verified (bpp)"
)
CONDITION_VERIFIED_OP = EvalCondition(
    "verified_full", "op", VERIFIED_SCORE_KEY, "Verified (op)"
)
CONDITION_VERIFIEDGT_OP_ORIGINAL_SCORE = EvalCondition(
    "verified_full", "op", ORIG_SCORE_KEY, "Verified (op) original score"
)
CONDITION_VERIFIEDGT_BPP_ORIGINAL_SCORE = EvalCondition(
    "verified_full", "bpp", ORIG_SCORE_KEY, "Verified (bpp) original score"
)

COMPARISONS: "list[Comparison]" = [
    # ── Prompt impact (op → bpp) for each GT × score combination ──────────────
    Comparison(
        name="Original GT & Score: op vs bpp Prompt",
        condition_a=CONDITION_ORIGINAL,
        condition_b=CONDITION_ORIGINAL_BPP,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(SCORES_LIST),
        stats_alternative="greater",  # hypothesis: bpp > op
        delta_label="bpp - op",
        label_a="op Prompt", label_b="bpp Prompt",
    ),
    Comparison(
        name="Original GT & Verified Score: op vs bpp Prompt",
        condition_a=CONDITION_ORIGINALGT_OP_VERIFIED_SCORE,
        condition_b=CONDITION_ORIGINALGT_BPP_VERIFIED_SCORE,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(VERIFIED_SCORES_LIST),
        delta_label="bpp - op",
        label_a="op Prompt", label_b="bpp Prompt",
    ),
    Comparison(
        name="Verified GT & Score: op vs bpp Prompt",
        condition_a=CONDITION_VERIFIED_OP,
        condition_b=CONDITION_VERIFIED,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(VERIFIED_SCORES_LIST),
        delta_label="bpp - op",
        label_a="op Prompt", label_b="bpp Prompt",
    ),
    Comparison(
        name="Verified GT & Original Score: op vs bpp Prompt",
        condition_a=CONDITION_VERIFIEDGT_OP_ORIGINAL_SCORE,
        condition_b=CONDITION_VERIFIEDGT_BPP_ORIGINAL_SCORE,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(SCORES_LIST),
        delta_label="bpp - op",
        label_a="op Prompt", label_b="bpp Prompt",
    ),
    # ── Evaluation GT impact (original → verified) for each score × prompt ────
    Comparison(
        name="Original score & op Prompt: Original vs Verified GT",
        condition_a=CONDITION_ORIGINAL,
        condition_b=CONDITION_VERIFIEDGT_OP_ORIGINAL_SCORE,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(SCORES_LIST),
        stats_alternative="less",  # hypothesis: verified GT < original GT
        delta_label="verified - original GT",
        label_a="original GT", label_b="verified GT",
    ),
    Comparison(
        name="Verified score & op Prompt: Original vs Verified GT",
        condition_a=CONDITION_ORIGINALGT_OP_VERIFIED_SCORE,
        condition_b=CONDITION_VERIFIED_OP,
        run_stats=True, run_impact_plot=True,
        subscores=tuple(VERIFIED_SCORES_LIST),
        delta_label="verified - original GT",
        label_a="original GT", label_b="verified GT",
    ),
    # ── Ranking comparisons (original vs verified scoring formula) ─────────────
    Comparison(
        name="Original GT & op Prompt: Original vs Verified score",
        condition_a=CONDITION_ORIGINAL,
        condition_b=CONDITION_ORIGINALGT_OP_VERIFIED_SCORE,
        run_ranking=True, run_bootstrap=True,
        delta_label="verified - original score",
        label_a="original score", label_b="verified score",
    ),
    Comparison(
        name="Original GT & bpp Prompt: Original vs Verified score",
        condition_a=CONDITION_ORIGINAL_BPP,
        condition_b=CONDITION_ORIGINALGT_BPP_VERIFIED_SCORE,
        run_ranking=True, run_bootstrap=True,
        delta_label="verified - original score",
        label_a="original score", label_b="verified score",
    ),
    Comparison(
        name="Original vs Verified",
        condition_a=CONDITION_ORIGINAL,
        condition_b=CONDITION_VERIFIED,
        run_ranking=True, run_bootstrap=True,
        save_summary=True,  # primary comparison — feeds save_results()
        label_a="Original", label_b="Verified",
    ),
]


def get_prompt_from_run(x):
    return "op" if "op" in x else ("bpp" if "bpp" in x else "--")


def get_eval_type_from_run(x):
    # Two filename conventions exist: newer files use "__<eval_type>__" segments;
    # older files use "in_<eval_type>" as a suffix.
    return x.split("__")[1] if "__" in x else (x.split("in_")[1] if "in_" in x else "?")


def get_model_name_from_run(x):
    return max((model for model in MODEL_NAMES if model in str(x)), key=len)


def get_run_number_from_run(x):
    return int(x.split("run_")[1].split("__")[0]) if "run_" in x else -1


def get_fps(file_path) -> str | int:
    try:
        with open(file_path.parent.parent / "summary.json") as f:
            json_dict = json.load(f)
            return json_dict[next(iter(json_dict.keys()))][FPS_KEY]
    except Exception as e:
        print(f"Could not extract FPS for {file_path.parent.parent}: {e}")
        return "?"


def get_experiment_tables(base_path: Path) -> list[IQTable]:
    """Discover all result CSVs under *base_path* and return one IQTable per run.

    Expected directory layout::

        <base_path>/**/results/**/results/<run_stem>.csv
                                         summary.json   ← FPS read from here
    """
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
            FPS_KEY: get_fps(fp),
            RUN_KEY: get_run_number_from_run(fp.stem),
        }
        tab = IQTable.from_csv(fp, metadata=meta_info)
        exp_tables.append(tab)
    return exp_tables


def filter_by_settings(df: pd.DataFrame, settings: list[dict]) -> pd.DataFrame:
    """Keep rows where *all* key–value pairs of any one settings dict match (OR of ANDs)."""
    mask = pd.Series(False, index=df.index)
    for setting in settings:
        setting_mask = pd.Series(True, index=df.index)
        for key, value in setting.items():
            setting_mask &= df[key] == value
        mask |= setting_mask
    return df[mask]


def generate_latex_table(
    output_path: Path,
    pivot_df_meanstd,
    table_name: str,
    score_cols: list,
    model_subset: list[str] | None = None,
    adaptive_std: bool = False,
):
    """Write a LaTeX table of score cells to *output_path/tables/table_name*.

    Parameters
    ----------
    model_subset:
        When given, only these models are included and they appear first; all
        other rows follow.  Defaults to ``MODEL_NAMES`` (all known models first).
    adaptive_std:
        When True, cells where std is NaN (single run) are formatted as
        ``$X.X$`` instead of ``$X.X \\pm nan$``.
    """
    effective_models = model_subset if model_subset is not None else MODEL_NAMES

    def _fmt(row):
        if adaptive_std and pd.isna(row["std"]):
            return f"${row['mean']:.1f}$"
        return f"${row['mean']:.1f} \\pm {row['std']:.1f}$"

    latex_table = pd.DataFrame(
        {col: pivot_df_meanstd[col].apply(_fmt, axis=1) for col in score_cols}
    )
    latex_table.columns = [col.replace("_", " ") for col in latex_table.columns]

    # Rename non-model index levels (underscore → space); keep MODEL_KEY raw for filtering
    model_key_level = latex_table.index.names.index(MODEL_KEY)
    latex_table.index = pd.MultiIndex.from_tuples(
        [
            tuple(str(v) if i == model_key_level else str(v).replace("_", " ")
                  for i, v in enumerate(idx))
            for idx in latex_table.index
        ],
        names=latex_table.index.names,
    )

    mask_valid = latex_table.index.get_level_values(EVAL_KEY).isin(
        ["verified", "verified full", "original"]
    )
    latex_table = latex_table[mask_valid]
    mask_models = latex_table.index.get_level_values(MODEL_KEY).isin(effective_models)
    latex_table = pd.concat([latex_table[mask_models], latex_table[~mask_models]])

    # Apply plotting names to model level now that filtering is done
    latex_table.index = pd.MultiIndex.from_tuples(
        [
            tuple(model_to_plotting_name(v) if i == model_key_level else v
                  for i, v in enumerate(idx))
            for idx in latex_table.index
        ],
        names=latex_table.index.names,
    )
    latex_str = latex_table.to_latex(escape=False)
    with open(_subdir(output_path, "tables") / table_name, "w") as f:
        f.write(latex_str)


def generate_latex_table_sora2(
    scores_df: pd.DataFrame,
    output_path: Path,
    table_name: str,
    score_cols: list,
):
    """LaTeX table for Sora 2 model variants — delegates to ``generate_latex_table``.

    FPS is kept in the groupby index because variants may have been run at
    different frame rates.  Uses ``adaptive_std=True`` so single-run entries
    render as ``$X.X$`` rather than ``$X.X \\pm nan$``.
    """
    df = scores_df[scores_df[MODEL_KEY].isin(SORA2_MODELS) & (scores_df[FPS_KEY] == 30)]
    pivot = (
        df.groupby([MODEL_KEY, EVAL_KEY, PROMPT_KEY])[score_cols].agg(
            ["mean", "std"]
        )
        * 100
    )
    generate_latex_table(
        output_path,
        pivot,
        table_name,
        score_cols,
        model_subset=SORA2_MODELS,
        adaptive_std=True,
    )


# Variance metric keys and human-readable labels used across the variance analysis
_VAR_KEYS = [
    IQTable.variance_spatial_key,
    IQTable.variance_weighted_spatial_key,
    IQTable.variance_spatiotemporal_iou_key,
    IQTable.variance_mse_key,
]
_VAR_DISPLAY = {
    IQTable.variance_spatial_key: "Spatial",
    IQTable.variance_weighted_spatial_key: "Weighted Spatial",
    IQTable.variance_spatiotemporal_iou_key: "Spatiotemporal",
    IQTable.variance_mse_key: "MSE",
}


def build_scenario_variance_df(exp_tables: list[IQTable]) -> pd.DataFrame:
    """Extract per-scenario physical variance values from all IQTables.

    Returns a long-format DataFrame with one row per scenario per IQTable, with
    columns for each variance metric plus the run metadata.  The ``scenario_id``
    column is taken from a ``"scenario"`` column in the raw CSV if present,
    otherwise from the integer row index.
    """
    chunks = []
    for tab in exp_tables:
        full_df = tab.get_full_df()
        var_keys = [k for k in _VAR_KEYS if k in full_df.columns]
        sub = full_df[var_keys].copy()
        sub["scenario_id"] = (
            full_df["scenario"].values
            if "scenario" in full_df.columns
            else full_df.index.astype(str).values
        )
        sub[MODEL_KEY] = tab.metadata.get(MODEL_KEY)
        sub[EVAL_KEY] = tab.metadata.get(EVAL_KEY)
        sub[PROMPT_KEY] = tab.metadata.get(PROMPT_KEY)
        sub[FPS_KEY] = tab.metadata.get(FPS_KEY)
        sub[RUN_KEY] = tab.metadata.get(RUN_KEY)
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)


def analyze_variance_shifts(scenario_var_df: pd.DataFrame, output_path: Path):
    """Compute per-scenario variance deltas (verified_full − original) and save statistics.

    Because physical variance is a property of the scenario itself (not of the
    model), values are first deduplicated across models/runs before matching.
    Outputs ``variance_shift_by_scenario.csv`` and ``variance_shift_summary.csv``
    to *output_path/data/*.
    """
    var_keys = [k for k in _VAR_KEYS if k in scenario_var_df.columns]

    # Variance is scenario-level, not model-level — average across models/runs first
    dedup = (
        scenario_var_df.groupby([EVAL_KEY, "scenario_id"])[var_keys]
        .mean()
        .reset_index()
    )
    orig = dedup[dedup[EVAL_KEY] == "original"].set_index("scenario_id")[var_keys]
    ver = dedup[dedup[EVAL_KEY] == "verified_full"].set_index("scenario_id")[var_keys]

    common_ids = orig.index.intersection(ver.index)
    if len(common_ids) == 0:
        print(
            "No matched scenarios between original and verified evaluation — skipping variance shift analysis."
        )
        return pd.DataFrame()

    delta_df = (ver.loc[common_ids] - orig.loc[common_ids]).reset_index()
    delta_df["eval_comparison"] = "verified_full - original"
    delta_df.to_csv(
        _subdir(output_path, "data") / "variance_shift_by_scenario.csv", index=False
    )

    summary_rows = []
    for key in var_keys:
        deltas = delta_df[key].dropna().values
        if len(deltas) < 2:
            summary_rows.append({"metric": key, "n_matched": len(deltas)})
            continue
        try:
            stat_w, p_w = stats.wilcoxon(deltas, alternative="two-sided")
        except ValueError:
            stat_w, p_w = np.nan, np.nan
        std_d = np.std(deltas, ddof=1)
        summary_rows.append(
            {
                "metric": key,
                "n_matched": len(deltas),
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(std_d),
                "frac_positive": float(np.mean(deltas > 0)),
                "wilcoxon_stat": float(stat_w) if not np.isnan(stat_w) else np.nan,
                "wilcoxon_p": float(p_w) if not np.isnan(p_w) else np.nan,
                "cohens_d": float(np.mean(deltas) / std_d) if std_d > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pprint(summary_df)
    summary_df.to_csv(
        _subdir(output_path, "data") / "variance_shift_summary.csv", index=False
    )
    return delta_df


def plot_variance_distributions(scenario_var_df: pd.DataFrame, output_path: Path):
    """2×2 violin plots of per-scenario physical variance: original vs. verified evaluation."""
    var_keys = [k for k in _VAR_KEYS if k in scenario_var_df.columns]

    dedup = (
        scenario_var_df.groupby([EVAL_KEY, "scenario_id"])[var_keys]
        .mean()
        .reset_index()
    )
    orig_rows = dedup[dedup[EVAL_KEY] == "original"]
    ver_rows = dedup[dedup[EVAL_KEY] == "verified_full"]

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
    axes = axes.flatten()
    rng = np.random.default_rng(seed=0)

    for ax, key in zip(axes, var_keys):
        groups = [orig_rows[key].dropna().values, ver_rows[key].dropna().values]
        labels = ["Original", "Verified"]
        colors = [_C_ORIG, _C_VER]
        positions = [1, 2]

        valid = [
            (pos, g, c) for pos, g, c in zip(positions, groups, colors) if len(g) > 1
        ]
        if valid:
            vp = ax.violinplot(
                [g for _, g, _ in valid],
                positions=[p for p, _, _ in valid],
                showmedians=False,
                showextrema=False,
                widths=0.5,
            )
            for body, (_, _, color) in zip(vp["bodies"], valid):
                body.set_facecolor(color)
                body.set_alpha(0.5)
                body.set_edgecolor(color)
                body.set_linewidth(0.5)

        for pos, data, color in zip(positions, groups, colors):
            if len(data) == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(
                pos + jitter, data, color=color, s=8, alpha=0.3, linewidths=0, zorder=2
            )
            mean = np.mean(data)
            ci = np.percentile(data, [2.5, 97.5])
            ax.errorbar(
                pos,
                mean,
                yerr=[[max(0.0, mean - ci[0])], [max(0.0, ci[1] - mean)]],
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                capsize=3,
                capthick=1.1,
                elinewidth=1.1,
                markersize=6,
                zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(_VAR_DISPLAY[key])
        ax.set_ylabel("Physical variance")
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Per-scenario Physical Variance: Original vs. Verified Evaluation", fontsize=11
    )
    fig.tight_layout()
    _save_fig(fig, output_path, "variance_distributions")


def plot_variance_scenario_scatter(scenario_var_df: pd.DataFrame, output_path: Path):
    """Scatter of per-scenario original vs verified variance, coloured by |Δ|.

    Directly shows which scenarios have the largest shift in physical variance
    between the two evaluation pipelines.  Top-5 outliers per panel are
    annotated with their scenario identifier.
    """
    var_keys = [k for k in _VAR_KEYS if k in scenario_var_df.columns]

    dedup = (
        scenario_var_df.groupby([EVAL_KEY, "scenario_id"])[var_keys]
        .mean()
        .reset_index()
    )
    orig = dedup[dedup[EVAL_KEY] == "original"].set_index("scenario_id")[var_keys]
    ver = dedup[dedup[EVAL_KEY] == "verified_full"].set_index("scenario_id")[var_keys]
    common_ids = orig.index.intersection(ver.index)

    if len(common_ids) == 0:
        print("No matched scenarios — skipping variance scenario scatter plot.")
        return

    # constrained_layout handles the per-panel colorbars without tight_layout conflicts
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.0), constrained_layout=True)
    axes = axes.flatten()

    for ax, key in zip(axes, var_keys):
        x = orig.loc[common_ids, key].values
        y = ver.loc[common_ids, key].values
        delta_abs = np.abs(y - x)

        lo = min(x.min(), y.min()) * 0.95
        hi = max(x.max(), y.max()) * 1.05
        ax.plot(
            [lo, hi], [lo, hi], color="#888888", linewidth=0.9, linestyle=":", zorder=1
        )

        sc = ax.scatter(
            x, y, c=delta_abs, cmap="viridis", s=20, alpha=0.75, linewidths=0, zorder=2
        )
        fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, label="|Δ variance|")

        # Annotate top-5 outliers
        top5 = np.argsort(delta_abs)[-5:][::-1]
        for i in top5:
            ax.annotate(
                str(common_ids[i]),
                (x[i], y[i]),
                fontsize=6.5,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
                color="#333333",
            )

        if len(x) > 1:
            r = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.04,
                0.96,
                f"r = {r:.3f}",
                transform=ax.transAxes,
                fontsize=8.5,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9
                ),
            )

        ax.set_xlabel("Original evaluation")
        ax.set_ylabel("Verified evaluation")
        ax.set_title(_VAR_DISPLAY[key])

    _save_fig(fig, output_path, "variance_scenario_scatter")


def mean_score_evaluation(
    scores_df: pd.DataFrame,
    condition_a: "EvalCondition | None" = None,
    condition_b: "EvalCondition | None" = None,
) -> dict:
    """Compare point-estimate model rankings between two evaluation conditions.

    Defaults to ``CONDITION_ORIGINAL`` vs ``CONDITION_VERIFIED``.  Spearman ρ
    and Kendall τ measure how well the two protocols agree on model ordering.

    Returns a dict with keys: ``ranking_df``, ``spearman_rho``, ``spearman_p``,
    ``kendall_tau``, ``kendall_p``.
    """
    if condition_a is None:
        condition_a = CONDITION_ORIGINAL
    if condition_b is None:
        condition_b = CONDITION_VERIFIED

    print(f"\nModel rankings: {condition_a.label} vs {condition_b.label}")
    scores_a = (
        condition_a.filter(scores_df)[[MODEL_KEY, condition_a.score_key]]
        .groupby(MODEL_KEY)
        .mean(numeric_only=True)
    )
    scores_a["rank"] = scores_a[condition_a.score_key].rank(ascending=False)

    scores_b = (
        condition_b.filter(scores_df)[[MODEL_KEY, condition_b.score_key]]
        .groupby(MODEL_KEY)
        .mean(numeric_only=True)
    )
    scores_b["rank"] = scores_b[condition_b.score_key].rank(ascending=False)

    ranking_df = scores_a.join(
        scores_b, lsuffix="_original", rsuffix="_verified", how="inner"
    )
    ranking_df["rank_delta"] = ranking_df["rank_verified"] - ranking_df["rank_original"]
    pprint(ranking_df)

    rho, p_rho = stats.spearmanr(
        ranking_df["rank_original"], ranking_df["rank_verified"]
    )
    print(f"Spearman ρ = {rho:.3f}  (p = {p_rho:.3f})")
    tau, p_tau = stats.kendalltau(
        ranking_df["rank_original"], ranking_df["rank_verified"]
    )
    print(f"Kendall  τ = {tau:.3f}  (p = {p_tau:.3f})")

    return {
        "ranking_df": ranking_df,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "kendall_tau": tau,
        "kendall_p": p_tau,
    }


def compare_conditions_stats(
    scores_df: pd.DataFrame,
    condition_a: "EvalCondition",
    condition_b: "EvalCondition",
    score_cols: list[str],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Paired Wilcoxon, t-test, and Cohen's d for condition_a vs condition_b.

    Pairs are matched on ``(MODEL_KEY, FPS_KEY)``.  Returns one row per score
    column with columns ``score``, ``wilcoxon_stat``, ``wilcoxon_p``,
    ``ttest_stat``, ``ttest_p``, ``cohens_d``.

    Importable by ``analysis_gt_runs.py`` for reuse without loading all of
    ``analysis.py``'s heavy dependencies.
    """
    df_a = condition_a.filter(scores_df).set_index([MODEL_KEY, FPS_KEY])
    df_b = condition_b.filter(scores_df).set_index([MODEL_KEY, FPS_KEY])
    common = df_a.index.intersection(df_b.index)

    rows = []
    for score in score_cols:
        sa = df_a.loc[common, score]
        sb = df_b.loc[common, score]
        diff = sb - sa
        try:
            stat_w, p_w = stats.wilcoxon(sb.values, sa.values, alternative=alternative)
        except ValueError:
            stat_w, p_w = np.nan, np.nan
        stat_t, p_t = stats.ttest_rel(sb.values, sa.values, alternative=alternative)
        std_d = diff.std(ddof=1)
        rows.append(
            {
                "score": score,
                "wilcoxon_stat": float(stat_w) if not np.isnan(stat_w) else np.nan,
                "wilcoxon_p": float(p_w) if not np.isnan(p_w) else np.nan,
                "ttest_stat": float(stat_t),
                "ttest_p": float(p_t),
                "cohens_d": float(diff.mean() / std_d) if std_d > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


_DISPLAY_OVERRIDES: dict[str, str] = {"mse": "MSE"}

# Columns where a lower value is better.  Deltas for these columns are negated
# in _plot_slope_delta so "bar extends right = condition_b is better" holds
# uniformly across all subscore panels.
_LOWER_IS_BETTER: frozenset[str] = frozenset([
    IQTable.mse_key,           # "v1_mse"         raw metric
    "score_mse",               # SCORES_LIST entry
    IQTable.variance_mse_key,  # "variance_mse"   variance column
])


def _subscore_display(col: str) -> str:
    """Human-readable panel title for a subscore column name.

    Strips ``score_`` / ``final_score_`` prefixes, handles the ``_view`` suffix
    from ``VERIFIED_SCORES_LIST``, and applies display overrides (e.g. ``mse``
    → ``MSE``).
    """
    key = col.removeprefix("score_").removeprefix("final_score_")
    if key.endswith("_view"):
        base = key[:-5]
        base_label = _DISPLAY_OVERRIDES.get(base, base.replace("_", " ").title())
        return f"{base_label} (view)"
    return _DISPLAY_OVERRIDES.get(key, key.replace("_", " ").title())


def _subdir(output_path: Path, name: str) -> Path:
    """Return *output_path/name*, creating it if it does not exist."""
    d = output_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_fig(fig: plt.Figure, output_path: Path, stem: str):
    """Save figure as PDF under figures/ and PNG under preview/."""
    fig.savefig(_subdir(output_path, "figures") / f"{stem}.pdf")
    fig.savefig(_subdir(output_path, "preview") / f"{stem}.png")
    plt.close(fig)


def _hist_panel(ax, values, color, xlabel, title):
    """Draw a styled histogram with mean line, 95% CI shading, and annotation."""
    mean = np.mean(values)
    ci = np.percentile(values, [2.5, 97.5])
    ax.hist(values, bins=20, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(mean, color=color, linewidth=1.8, linestyle="--", zorder=3)
    ax.axvspan(ci[0], ci[1], alpha=0.15, color=color, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.text(
        0.03,
        0.97,
        f"mean = {mean:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9
        ),
    )


def plot_correlation_distributions(
    spearman_rhos: list,
    kendall_taus: list,
    rho_mean: float,
    ci_rho: np.ndarray,
    tau_mean: float,
    ci_tau: np.ndarray,
    output_path: Path,
):
    """Bootstrap distributions of ranking correlations (original vs. verified evaluation)."""
    fig, (ax_rho, ax_tau) = plt.subplots(1, 2, figsize=(6.5, 2.8))
    _hist_panel(
        ax_rho,
        spearman_rhos,
        _C_RHO,
        xlabel="Spearman's ρ (original vs. verified)",
        title="Ranking Correlation: Spearman's ρ",
    )
    _hist_panel(
        ax_tau,
        kendall_taus,
        _C_TAU,
        xlabel="Kendall's τ (original vs. verified)",
        title="Ranking Correlation: Kendall's τ",
    )
    fig.tight_layout()
    _save_fig(fig, output_path, "bootstrap_correlation_distributions")


def _draw_ranking_scatter(
    ax: plt.Axes,
    ranks_original: list,
    ranks_verified: list,
    rho_mean: float,
    tau_mean: float,
    label_a: "str | None" = None,
    label_b: "str | None" = None,
    swap: bool = False,
) -> None:
    """Draw bootstrap rank scatter into an existing axes."""
    model_ranks: dict[str, list[tuple]] = {}
    for r_orig, r_ver in zip(ranks_original, ranks_verified):
        for model in r_orig.index:
            model_ranks.setdefault(model, []).append((r_orig[model], r_ver[model]))

    n_models = len(model_ranks)
    fs = plt.rcParams.get("font.size", 9)

    ax.plot(
        [0.5, n_models + 0.5], [0.5, n_models + 0.5],
        color="#888888", linewidth=0.9, linestyle=":", zorder=1,
    )

    for model, pairs in model_ranks.items():
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        color = model_to_color(model)
        display = model_to_plotting_name(model)
        ax.scatter(x, y, color=color, s=18, alpha=0.18, zorder=2, linewidths=0)
        ax.scatter(
            np.mean(x), np.mean(y),
            color=color, s=70, alpha=1.0, zorder=3, label=display,
            edgecolors="white", linewidths=0.6,
        )

    ticks = list(range(1, n_models + 1))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(n_models + 0.5, 0.5)
    ax.set_ylim(n_models + 0.5, 0.5)
    ax.set_xlabel(f"Rank ({label_a})" if label_a else "Rank (original evaluation)")
    ax.set_ylabel(f"Rank ({label_b})" if label_b else "Rank (verified evaluation)")
    ax.set_title("Bootstrap Model Rankings:\nOriginal vs. Verified Evaluation")
    stat_x,  stat_y,  stat_va,  stat_ha  = (0.97, 0.03, "bottom", "right") if not swap else (0.03, 0.97, "top", "left")
    legend_loc = "upper left" if not swap else "lower right"
    ax.text(
        stat_x, stat_y,
        f"$\\bar{{\\rho}}$ = {rho_mean:.3f},  $\\bar{{\\tau}}$ = {tau_mean:.3f}",
        transform=ax.transAxes, fontsize=fs, va=stat_va, ha=stat_ha,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.legend(loc=legend_loc, markerscale=1.2, handletextpad=0.4, borderpad=0.6)
    ax.set_aspect("equal")


def plot_ranking_scatter(
    ranks_original: list,
    ranks_verified: list,
    rho_mean: float,
    tau_mean: float,
    output_path: Path,
    label_a: "str | None" = None,
    label_b: "str | None" = None,
    swap: bool = False,
):
    """Scatter of per-model bootstrap ranks: original vs. verified evaluation."""
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    _draw_ranking_scatter(ax, ranks_original, ranks_verified, rho_mean, tau_mean,
                          label_a=label_a, label_b=label_b, swap=swap)
    fig.tight_layout()
    _save_fig(fig, output_path, "bootstrap_ranking_scatter")


def plot_within_evaluation_correlations(
    rhos_orig: list,
    taus_orig: list,
    rhos_ver: list,
    taus_ver: list,
    output_path: Path,
):
    """2×2 grid: bootstrap ranking stability within each evaluation type, for both ρ and τ."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.8))
    (ax_rho_orig, ax_tau_orig), (ax_rho_ver, ax_tau_ver) = axes

    # Shared x range per metric so rows are directly comparable
    rho_min = min(rhos_orig + rhos_ver) - 0.02
    rho_max = min(max(rhos_orig + rhos_ver) + 0.02, 1.0)
    tau_min = min(taus_orig + taus_ver) - 0.02
    tau_max = min(max(taus_orig + taus_ver) + 0.02, 1.0)

    _hist_panel(
        ax_rho_orig,
        rhos_orig,
        _C_ORIG,
        xlabel="Spearman's ρ (consecutive samples)",
        title="Original — Spearman's ρ",
    )
    _hist_panel(
        ax_tau_orig,
        taus_orig,
        _C_ORIG,
        xlabel="Kendall's τ (consecutive samples)",
        title="Original — Kendall's τ",
    )
    _hist_panel(
        ax_rho_ver,
        rhos_ver,
        _C_VER,
        xlabel="Spearman's ρ (consecutive samples)",
        title="Verified — Spearman's ρ",
    )
    _hist_panel(
        ax_tau_ver,
        taus_ver,
        _C_VER,
        xlabel="Kendall's τ (consecutive samples)",
        title="Verified — Kendall's τ",
    )

    for ax in (ax_rho_orig, ax_rho_ver):
        ax.set_xlim(rho_min, rho_max)
    for ax in (ax_tau_orig, ax_tau_ver):
        ax.set_xlim(tau_min, tau_max)

    fig.tight_layout()
    _save_fig(fig, output_path, "bootstrap_within_evaluation_correlation_distributions")


def plot_ranking_distributions(
    ranks_original: list, ranks_verified: list, output_path: Path
):
    """Box plots of bootstrap rank distributions per model, for each evaluation type."""
    ranking_values_original = pd.concat(ranks_original, axis=1)
    ranking_values_verified = pd.concat(ranks_verified, axis=1)

    # Sort models by median rank in the original evaluation (best → worst)
    model_order = ranking_values_original.median(axis=1).sort_values().index.tolist()

    def _boxplot_panel(ax, ranking_values, title):
        models = model_order
        data = [ranking_values.loc[m].values for m in models]
        bxp = ax.boxplot(
            data,
            positions=range(1, len(models) + 1),
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=0.8, linestyle="--"),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markeredgewidth=0),
            boxprops=dict(linewidth=0.7),
        )
        for patch, model in zip(bxp["boxes"], models):
            patch.set_facecolor(model_to_color(model))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(
            [model_to_plotting_name(m) for m in models],
            rotation=35,
            ha="right",
            fontsize=8.5,
        )
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.6, color="#dddddd")
        ax.set_axisbelow(True)
        ax.invert_yaxis()  # rank 1 (best) at top
        ax.set_ylabel("Rank  (1 = best)")
        ax.set_title(title)

    fig, (ax_orig, ax_ver) = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True)
    _boxplot_panel(ax_orig, ranking_values_original, "Original Evaluation")
    _boxplot_panel(ax_ver, ranking_values_verified, "Verified Evaluation")
    fig.tight_layout()
    _save_fig(fig, output_path, "bootstrap_ranking_distributions")


def plot_combined_correlation_comparison(
    spearman_rhos: list,
    kendall_taus: list,
    rhos_within_orig: list,
    taus_within_orig: list,
    rhos_within_ver: list,
    taus_within_ver: list,
    output_path: Path,
    show_stats: bool = False,
):
    """Violin plot comparing cross- and within-evaluation bootstrap ranking correlations.

    Places the three comparison types (cross-evaluation, within-original,
    within-verified) side by side on a shared axis for each metric, so the
    agreement between protocols can be read against the internal stability of
    each protocol in a single glance.

    Layout: 2 panels (Spearman ρ | Kendall τ), shared y-axis.
    Each group shows: jittered individual bootstrap values, violin density, and
    a mean ± 95% CI errorbar.

    Parameters
    ----------
    show_stats:
        When True the mean and 95% CI are printed below each x-tick label so
        the exact values are readable without a separate table.  The function is
        called twice from ``run_bootstrap_analysis`` — once per value — and the
        output filename includes ``_annotated`` for the annotated version.
    """
    groups_rho = [spearman_rhos, rhos_within_orig, rhos_within_ver]
    groups_tau = [kendall_taus, taus_within_orig, taus_within_ver]
    group_colors = [_C_CROSS, _C_ORIG, _C_VER]
    x_labels = ["Cross-eval.", "Within orig.", "Within ver."]
    positions = [1, 2, 3]

    # Shared y range across both panels so ρ and τ magnitudes are comparable
    all_vals = [v for g in groups_rho + groups_tau for v in g]
    y_min = min(all_vals) - 0.04
    y_max = min(max(all_vals) + 0.04, 1.02)

    # Extra figure height when stats are shown below x-labels
    fig_height = 4.4 if show_stats else 3.4
    fig, (ax_rho, ax_tau) = plt.subplots(1, 2, figsize=(6.5, fig_height), sharey=True)

    for ax, groups, title in [
        (ax_rho, groups_rho, "Spearman's ρ"),
        (ax_tau, groups_tau, "Kendall's τ"),
    ]:
        vp = ax.violinplot(
            groups,
            positions=positions,
            showmedians=False,
            showextrema=False,
            widths=0.6,
        )
        for body, color in zip(vp["bodies"], group_colors):
            body.set_facecolor(color)
            body.set_alpha(0.45)
            body.set_edgecolor(color)
            body.set_linewidth(0.5)

        # Fixed seed so jitter positions are reproducible regardless of global RNG state
        rng = np.random.default_rng(seed=0)
        for pos, data, color in zip(positions, groups, group_colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(data))
            ax.scatter(
                pos + jitter,
                data,
                color=color,
                s=10,
                alpha=0.25,
                linewidths=0,
                zorder=2,
            )

        for pos, data, color in zip(positions, groups, group_colors):
            mean = np.mean(data)
            ci = np.percentile(data, [2.5, 97.5])
            # Clamp to zero: when τ/ρ values cluster at the discrete maximum (1.0
            # for 5 models), the 97.5th percentile can sit above the mean, making
            # ci[1] - mean negative. The violin + jitter already show the shape.
            yerr_lo = max(0.0, mean - ci[0])
            yerr_hi = max(0.0, ci[1] - mean)
            # markerfacecolor must be set explicitly; `color` only controls the line
            ax.errorbar(
                pos,
                mean,
                yerr=[[yerr_lo], [yerr_hi]],
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=4,
                capthick=1.3,
                elinewidth=1.3,
                markersize=7,
                zorder=5,
            )

        # Build x-tick labels — optionally extend with per-group stats below the name
        if show_stats:
            tick_labels = [
                f"{label}\nμ={np.mean(data):.3f}\n[{np.percentile(data, 2.5):.3f}, {np.percentile(data, 97.5):.3f}]"
                for label, data in zip(x_labels, groups)
            ]
        else:
            tick_labels = x_labels

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
        ax.set_axisbelow(True)

    ax_rho.set_ylabel("Correlation coefficient")
    fig.tight_layout()
    stem = "bootstrap_combined_correlation_comparison"
    _save_fig(fig, output_path, stem + ("_annotated" if show_stats else ""))


def run_bootstrap_analysis(
    exp_tables: list[IQTable],
    n_bootstrap: int,
    output_path: Path,
    ranking_settings: list[dict] | None = None,
    comparison_conditions: "tuple[EvalCondition, EvalCondition] | None" = None,
    run_ids: tuple[int, ...] | None = None,
    label_a: "str | None" = None,
    label_b: "str | None" = None,
) -> dict:
    """Run bootstrap resampling over model runs, produce figures, and return statistics.

    Resampling strategy
    -------------------
    Each bootstrap iteration draws run IDs (integers 1–4) with replacement for
    every scenario position.  Rows belonging to the sampled run ID are selected
    from each IQTable, so the effective resample is at the *run* level — matching
    the natural unit of independent repetition in the experiment.

    ``get_full_df()`` is used (not ``df``) so that per-run DataFrame indices are
    preserved; ``assert boot_df.index.is_unique`` guards against accidental
    row-index collisions across the concatenated run slices.

    Within-evaluation stability
    ---------------------------
    Consecutive-sample ρ and τ (between bootstrap draw i and i+1 within the same
    evaluation type) measure how quickly the ranking estimator converges rather
    than a traditional bootstrap standard error.  High values indicate that the
    ranking is robust to which scenarios happen to be up-sampled.

    Returns a dict with keys: ``spearman_rhos``, ``kendall_taus``, ``rho_mean``,
    ``tau_mean``, ``ci_rho``, ``ci_tau``, ``within_rho_original``,
    ``within_tau_original``, ``within_rho_verified``, ``within_tau_verified``,
    ``ranks_original``, ``ranks_verified``.
    """
    if ranking_settings is None:
        ranking_settings = RANKING_EVAL_SETTINGS
    if comparison_conditions is None:
        comparison_conditions = (CONDITION_ORIGINAL, CONDITION_VERIFIED)
    if run_ids is None:
        run_ids = (1, 2, 3, 4)

    comparison_keys = [
        {EVAL_KEY: c.eval_type, PROMPT_KEY: c.prompt} for c in comparison_conditions
    ]
    cond_a, cond_b = comparison_conditions

    # Each entry is an array of scenario-length run IDs sampled with replacement.
    bootstrap_sample_list = [
        np.random.choice(list(run_ids), size=len(exp_tables[0].df), replace=True)
        for _ in range(n_bootstrap)
    ]

    bootstrap_dfs = []
    for i, bootstrap_sample in enumerate(bootstrap_sample_list):
        boot_iq_tables = []
        for setting, comp in product(ranking_settings, comparison_keys):
            setting = setting.copy()
            setting.update(comp)
            series = []
            for tab in exp_tables:
                if all(tab.metadata.get(k) == v for k, v in setting.items()):
                    mask = bootstrap_sample == tab.metadata[RUN_KEY]
                    series.append(tab.get_full_df()[mask])
            boot_df = pd.concat(series)
            assert len(boot_df) == len(
                bootstrap_sample
            ), f"Bootstrap sample length {len(bootstrap_sample)} does not match boot_df length {len(boot_df)}"
            assert (
                boot_df.index.is_unique
            ), "Bootstrap sample has duplicate indices and therefore is not a valid permutation based bootstrap sample."
            bootstrap_setting = {**setting, "bootstrap_sample": i}
            boot_iq_tables.append(IQTable(boot_df, metadata=bootstrap_setting))

        bootstrap_scores_df = pd.DataFrame(
            [tab.get_output_dict() for tab in boot_iq_tables]
        )
        bootstrap_dfs.append(bootstrap_scores_df)

    ranks_original, ranks_verified = [], []
    for bootstrap_scores_df in bootstrap_dfs:
        ranking_df = (
            bootstrap_scores_df[
                [MODEL_KEY, EVAL_KEY, ORIG_SCORE_KEY, VERIFIED_SCORE_KEY]
            ]
            .groupby([MODEL_KEY, EVAL_KEY])
            .mean(numeric_only=True)
        )
        r_orig = ranking_df.xs(cond_a.eval_type, level=EVAL_KEY)
        r_ver = ranking_df.xs(cond_b.eval_type, level=EVAL_KEY)
        r_orig = r_orig.assign(rank=r_orig[cond_a.score_key].rank(ascending=False))[
            "rank"
        ]
        r_ver = r_ver.assign(rank=r_ver[cond_b.score_key].rank(ascending=False))["rank"]
        ranks_original.append(r_orig)
        ranks_verified.append(r_ver)

    spearman_rhos, kendall_taus = [], []
    for r_orig, r_ver in zip(ranks_original, ranks_verified):
        merged = r_orig.to_frame().join(
            r_ver.to_frame(), lsuffix="_original", rsuffix="_verified", how="inner"
        )
        rho, _ = stats.spearmanr(merged["rank_original"], merged["rank_verified"])
        tau, _ = stats.kendalltau(merged["rank_original"], merged["rank_verified"])
        spearman_rhos.append(rho)
        kendall_taus.append(tau)

    ci_rho = np.percentile(spearman_rhos, [2.5, 97.5])
    ci_tau = np.percentile(kendall_taus, [2.5, 97.5])
    rho_mean = np.mean(spearman_rhos)
    tau_mean = np.mean(kendall_taus)
    print(
        f"Mean Spearman's ρ: {rho_mean:.3f}, 95% CI: [{ci_rho[0]:.3f}, {ci_rho[1]:.3f}]"
    )
    print(
        f"Mean Kendall's τ: {tau_mean:.3f}, 95% CI: [{ci_tau[0]:.3f}, {ci_tau[1]:.3f}]"
    )

    # Within-evaluation ρ and τ between consecutive bootstrap samples
    pairs_orig = list(zip(ranks_original[:-1], ranks_original[1:]))
    pairs_ver = list(zip(ranks_verified[:-1], ranks_verified[1:]))
    rhos_within_orig = [stats.spearmanr(r1, r2)[0] for r1, r2 in pairs_orig]
    taus_within_orig = [stats.kendalltau(r1, r2)[0] for r1, r2 in pairs_orig]
    rhos_within_ver = [stats.spearmanr(r1, r2)[0] for r1, r2 in pairs_ver]
    taus_within_ver = [stats.kendalltau(r1, r2)[0] for r1, r2 in pairs_ver]

    plot_correlation_distributions(
        spearman_rhos, kendall_taus, rho_mean, ci_rho, tau_mean, ci_tau, output_path
    )
    plot_ranking_scatter(
        ranks_original, ranks_verified, rho_mean, tau_mean, output_path,
        label_a=label_a, label_b=label_b,
    )
    plot_within_evaluation_correlations(
        rhos_within_orig,
        taus_within_orig,
        rhos_within_ver,
        taus_within_ver,
        output_path,
    )
    for show_stats in (False, True):
        plot_combined_correlation_comparison(
            spearman_rhos,
            kendall_taus,
            rhos_within_orig,
            taus_within_orig,
            rhos_within_ver,
            taus_within_ver,
            output_path,
            show_stats=show_stats,
        )
    plot_ranking_distributions(ranks_original, ranks_verified, output_path)

    return {
        "spearman_rhos": spearman_rhos,
        "kendall_taus": kendall_taus,
        "rho_mean": rho_mean,
        "tau_mean": tau_mean,
        "ci_rho": ci_rho,
        "ci_tau": ci_tau,
        "within_rho_original": rhos_within_orig,
        "within_tau_original": taus_within_orig,
        "within_rho_verified": rhos_within_ver,
        "within_tau_verified": taus_within_ver,
        "ranks_original": ranks_original,
        "ranks_verified": ranks_verified,
    }


def plot_model_performance_bars(
    scores_df: pd.DataFrame, score_key: str, output_path: Path, mode="verified"
):
    """Bar chart of mean Physics-IQ score per model, sorted best-to-worst (left to right).

    Uses the verified evaluation (``bpp`` prompt, ``verified_full`` scoring) at each
    model's canonical FPS from ``RANKING_EVAL_SETTINGS``.  Error bars show ±1 std
    across runs; the numeric score is printed above each bar.
    """
    if mode == "verified":
        df = scores_df[
            (scores_df[EVAL_KEY] == "verified_full") & (scores_df[PROMPT_KEY] == "bpp")
        ]
        plot_title = "Physics-IQ Verified Scores"
    elif mode == "original":
        df = scores_df[
            (scores_df[EVAL_KEY] == "original") & (scores_df[PROMPT_KEY] == "op")
        ]
        plot_title = "Physics-IQ Original Scores"
    else:
        raise ValueError(
            f"Invalid mode '{mode}' for plot_model_performance_bars. Expected 'verified' or 'original'."
        )

    grouped = (
        df.groupby([MODEL_KEY])[score_key].agg(["mean", "std"]) * 100
    ).sort_values("mean", ascending=False)
    models = grouped.index.tolist()
    means = grouped["mean"].values
    # std is NaN when a model has only one run; treat as 0 for plotting
    stds = np.nan_to_num(grouped["std"].values)
    colors = [model_to_color(m) for m in models]
    display_names = [model_to_plotting_name(m) for m in models]

    n = len(models)
    fig, ax = plt.subplots(figsize=(max(4.0, n * 1.1), 3.5))
    x = np.arange(n)

    ax.bar(
        x,
        means,
        width=0.6,
        color=colors,
        alpha=0.85,
        yerr=stds,
        capsize=4,
        error_kw=dict(linewidth=0.9, capthick=0.9, ecolor="#444444"),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    # Score label sits just above the error-bar cap
    for xi, mean, std in zip(x, means, stds):
        ax.text(
            xi,
            mean + std + 0.8,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(plot_title)
    ax.set_ylim(0, max(means + stds) * 1.22)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save_fig(fig, output_path, plot_title.replace(" ", "_").lower() + "_bars")


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value (empty string when not significant)."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _plot_slope_delta(
    scores_df: pd.DataFrame,
    condition_a: "EvalCondition",
    condition_b: "EvalCondition",
    sub_cols: dict[str, str],
    output_path: Path,
    stem: str,
    title: str,
    stats_df: "pd.DataFrame | None" = None,
    delta_label: str | None = None,
    label_a: "str | None" = None,
    label_b: "str | None" = None,
):
    """Generic slope chart + per-subscore delta bars for any two EvalConditions.

    Top panel   — per-model line from condition_a mean to condition_b mean, with
                  individual run points.  The score column is taken from each
                  condition's ``score_key``.
    Bottom row  — horizontal delta bar charts (Δ = condition_b − condition_a)
                  for each subscore in *sub_cols*.  Models are ordered by their
                  condition_a mean, best first.
    """
    df_a = condition_a.filter(scores_df)
    df_b = condition_b.filter(scores_df)

    models = (
        (df_a.groupby(MODEL_KEY)[condition_a.score_key].mean() * 100)
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = plt.figure(figsize=(6.5, 5.5), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0])
    ax_main = fig.add_subplot(gs[0, :])
    ax_subs = [fig.add_subplot(gs[1, i]) for i in range(4)]
    rng = np.random.default_rng(seed=0)

    for model in models:
        color = model_to_color(model)
        x_means = []
        for x_pos, cond, df_cond in [
            (0, condition_a, df_a),
            (1, condition_b, df_b),
        ]:
            runs = df_cond[df_cond[MODEL_KEY] == model][cond.score_key].values * 100
            if len(runs) == 0:
                continue
            jitter = rng.uniform(-0.06, 0.06, size=len(runs))
            ax_main.scatter(
                x_pos + jitter,
                runs,
                color=color,
                s=18,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
            x_means.append((x_pos, np.mean(runs)))

        if len(x_means) == 2:
            xs, ys = zip(*x_means)
            ax_main.plot(
                xs,
                ys,
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            ax_main.text(
                1.04,
                ys[1],
                model_to_plotting_name(model),
                va="center",
                fontsize=8.5,
                color=color,
            )

    eff_label_a = label_a or condition_a.label
    eff_label_b = label_b or condition_b.label

    ax_main.set_xticks([0, 1])
    ax_main.set_xticklabels([eff_label_a, eff_label_b], fontsize=9)
    ax_main.set_xlim(-0.25, 1.6)
    ax_main.set_ylabel("Physics-IQ Score [%]")
    ax_main.set_title(title)
    ax_main.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
    ax_main.set_axisbelow(True)

    models_rev = list(reversed(models))
    y_pos = np.arange(len(models_rev))
    if delta_label is None:
        delta_label = f"Δ ({eff_label_b} − {eff_label_a})"
    else:
        delta_label = f"Δ ({delta_label})"
    for ax, (sub_col, sub_label) in zip(ax_subs, sub_cols.items()):
        sub_a = df_a.groupby(MODEL_KEY)[sub_col].mean() * 100
        sub_b = df_b.groupby(MODEL_KEY)[sub_col].mean() * 100
        lower_is_better = sub_col in _LOWER_IS_BETTER
        deltas = [
            # Negate lower-is-better columns so positive bar = condition_b is better
            (sub_a[m] - sub_b[m] if lower_is_better else sub_b[m] - sub_a[m])
            if m in sub_a.index and m in sub_b.index else 0.0
            for m in models_rev
        ]
        colors_ordered = [model_to_color(m) for m in models_rev]

        ax.barh(
            y_pos,
            deltas,
            height=0.6,
            color=colors_ordered,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.axvline(0, color="#444444", linewidth=0.8, zorder=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            (
                [model_to_plotting_name(m) for m in models_rev]
                if ax is ax_subs[0]
                else [""] * len(models_rev)
            ),
            fontsize=7.5,
        )
        ax.set_title(sub_label, fontsize=9)
        ax.set_xlabel(("−" if lower_is_better else "") + delta_label, fontsize=8)
        ax.xaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
        ax.set_axisbelow(True)

        if stats_df is not None:
            row = stats_df[stats_df["score"] == sub_col]
            if not row.empty:
                d   = row["cohens_d"].values[0]
                p_w = row["wilcoxon_p"].values[0]
                # Sign of d follows the visual convention: positive = condition_b better
                d_display = -d if lower_is_better else d
                stars = _sig_stars(p_w)
                annotation = (
                    f"d = {d_display:+.2f}{stars}\n"
                    f"p = {p_w:.3f}" if not np.isnan(p_w) else f"d = {d_display:+.2f}"
                )
                ax.text(
                    0.97, 0.97, annotation,
                    transform=ax.transAxes, fontsize=7.0,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.9),
                )

    _save_fig(fig, output_path, stem)



def plot_slope_delta_with_stats(
    scores_df: pd.DataFrame,
    condition_a: "EvalCondition",
    condition_b: "EvalCondition",
    sub_cols: dict[str, str],
    output_path: Path,
    stem: str,
    title: str,
    stats_alternative: str = "two-sided",
    delta_label: str | None = None,
    label_a: "str | None" = None,
    label_b: "str | None" = None,
):
    """Slope + delta chart with per-subscore statistical annotation.

    Calls ``compare_conditions_stats`` internally and annotates each delta
    panel with Cohen's d (sign-adjusted for lower-is-better columns) and the
    Wilcoxon p-value with significance stars.  Output is saved with ``_stats``
    appended to *stem* so it does not overwrite the plain version.
    """
    score_cols = [condition_a.score_key] + list(sub_cols.keys())
    stats = compare_conditions_stats(
        scores_df, condition_a, condition_b,
        score_cols=score_cols,
        alternative=stats_alternative,
    )
    _plot_slope_delta(
        scores_df, condition_a, condition_b, sub_cols,
        output_path, stem=stem + "_stats", title=title,
        stats_df=stats,
        delta_label=delta_label,
        label_a=label_a, label_b=label_b,
    )


def plot_model_performance_bars_combined(
    scores_df: pd.DataFrame,
    output_path: Path,
    condition_a: "EvalCondition | None" = None,
    condition_b: "EvalCondition | None" = None,
    stem: str = "model_performance_bars_combined",
):
    """Side-by-side bar charts for two conditions on a shared y-axis.

    Each panel is sorted independently by its own mean score so the rank order is
    visible within each protocol.  The shared y-axis makes absolute score differences
    between protocols directly readable.
    """
    if condition_a is None:
        condition_a = CONDITION_ORIGINAL
    if condition_b is None:
        condition_b = CONDITION_VERIFIED

    specs = [
        (condition_a.label, condition_a.score_key, condition_a.filter(scores_df)),
        (condition_b.label, condition_b.score_key, condition_b.filter(scores_df)),
    ]

    # Compute aggregates for both panels up-front so we can set a shared y range
    panel_data = []
    for title, score_key, df in specs:
        grouped = (
            df.groupby(MODEL_KEY)[score_key].agg(["mean", "std"]) * 100
        ).sort_values("mean", ascending=False)
        panel_data.append((title, grouped))

    all_means = np.concatenate([g["mean"].values for _, g in panel_data])
    all_stds = np.concatenate([np.nan_to_num(g["std"].values) for _, g in panel_data])
    y_max = max(all_means + all_stds) * 1.22

    n_models = max(len(g) for _, g in panel_data)
    fig, axes = plt.subplots(
        1, 2, figsize=(max(5.0, n_models * 1.1) * 2, 3.5), sharey=True
    )

    for ax, (title, grouped) in zip(axes, panel_data):
        models = grouped.index.tolist()
        means = grouped["mean"].values
        stds = np.nan_to_num(grouped["std"].values)
        colors = [model_to_color(m) for m in models]
        display_names = [model_to_plotting_name(m) for m in models]
        x = np.arange(len(models))

        ax.bar(
            x,
            means,
            width=0.6,
            color=colors,
            alpha=0.85,
            yerr=stds,
            capsize=4,
            error_kw=dict(linewidth=0.9, capthick=0.9, ecolor="#444444"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for xi, mean, std in zip(x, means, stds):
            ax.text(
                xi,
                mean + std + 0.8,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, y_max)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Physics-IQ Score")
    fig.tight_layout()
    _save_fig(fig, output_path, stem)


def _draw_bars_grouped(
    ax: plt.Axes,
    scores_df: pd.DataFrame,
    condition_a: "EvalCondition",
    condition_b: "EvalCondition",
    label_a: "str | None" = None,
    label_b: "str | None" = None,
) -> None:
    """Draw grouped bar chart comparing two conditions into an existing axes."""
    agg_a = (
        condition_a.filter(scores_df)
        .groupby(MODEL_KEY)[condition_a.score_key]
        .agg(["mean", "std"]) * 100
    )
    agg_b = (
        condition_b.filter(scores_df)
        .groupby(MODEL_KEY)[condition_b.score_key]
        .agg(["mean", "std"]) * 100
    )

    models = agg_a["mean"].sort_values(ascending=False).index.tolist()
    for m in agg_b.index:
        if m not in models:
            models.append(m)

    def _get(agg, m, col):
        return agg.loc[m, col] if m in agg.index else np.nan

    means_a = np.array([_get(agg_a, m, "mean") for m in models])
    stds_a  = np.nan_to_num(np.array([_get(agg_a, m, "std")  for m in models]))
    means_b = np.array([_get(agg_b, m, "mean") for m in models])
    stds_b  = np.nan_to_num(np.array([_get(agg_b, m, "std")  for m in models]))
    colors  = [model_to_color(m) for m in models]

    fs = plt.rcParams.get("font.size", 9)
    bar_w = 0.38
    x = np.arange(len(models))
    y_max = np.nanmax(np.concatenate([means_a + stds_a, means_b + stds_b])) * 1.22

    _HATCH_A = ""
    _HATCH_B = "///"

    for i, (color, mean_a, std_a, mean_b, std_b) in enumerate(
        zip(colors, means_a, stds_a, means_b, stds_b)
    ):
        kw = dict(width=bar_w, alpha=0.85, zorder=3, capsize=4,
                  error_kw=dict(linewidth=0.9, capthick=0.9, ecolor="#444444"))
        ax.bar(x[i] - bar_w / 2, mean_a, color=color, hatch=_HATCH_A,
               edgecolor="white", linewidth=0.5, yerr=std_a, **kw)
        ax.bar(x[i] + bar_w / 2, mean_b, color=color, hatch=_HATCH_B,
               edgecolor="#333333", linewidth=0.5, yerr=std_b, **kw)

    for xi, mean, std in zip(x - bar_w / 2, means_a, stds_a):
        if not np.isnan(mean):
            ax.text(xi, mean + std + 0.8, f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=fs * 0.85, fontweight="bold")
    for xi, mean, std in zip(x + bar_w / 2, means_b, stds_b):
        if not np.isnan(mean):
            ax.text(xi, mean + std + 0.8, f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=fs * 0.85, fontweight="bold")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#aaaaaa", hatch=_HATCH_A, edgecolor="white",
              label=label_a or condition_a.label),
        Patch(facecolor="#aaaaaa", hatch=_HATCH_B, edgecolor="#333333",
              label=label_b or condition_b.label),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [model_to_plotting_name(m) for m in models],
        rotation=30, ha="right",
    )
    ax.set_ylabel("Physics-IQ Score [%]")
    ax.set_ylim(0, y_max)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, color="#dddddd")
    ax.set_axisbelow(True)


def plot_model_performance_bars_grouped(
    scores_df: pd.DataFrame,
    output_path: Path,
    condition_a: "EvalCondition | None" = None,
    condition_b: "EvalCondition | None" = None,
    stem: str = "model_performance_bars_grouped",
    label_a: "str | None" = None,
    label_b: "str | None" = None,
):
    """Grouped bar chart comparing two conditions on a single axis.

    Bar colour encodes the model (identical to the combined plot).  Condition
    is distinguished by hatch texture: condition_a is solid, condition_b uses
    diagonal hatching.  Models are sorted by condition_a mean (best → worst).
    """
    if condition_a is None:
        condition_a = CONDITION_ORIGINAL
    if condition_b is None:
        condition_b = CONDITION_VERIFIED

    n = condition_a.filter(scores_df).groupby(MODEL_KEY).ngroups
    fig, ax = plt.subplots(figsize=(max(4.5, n * 1.2), 3.8))
    _draw_bars_grouped(ax, scores_df, condition_a, condition_b,
                       label_a=label_a, label_b=label_b)
    fig.tight_layout()
    _save_fig(fig, output_path, stem)


def _bezier_bump(
    x0: float, y0: float, x1: float, y1: float, n: int = 120
) -> "tuple[np.ndarray, np.ndarray]":
    """Cubic Hermite S-curve with zero-slope (horizontal) tangents at both ends.

    Equivalent to the "smooth step" interpolation:
        y(t) = y0·(1−t)²·(1+2t) + y1·t²·(3−2t)
    which guarantees y(0)=y0, y(1)=y1, y'(0)=0, y'(1)=0.
    x is interpolated linearly.
    """
    t = np.linspace(0.0, 1.0, n)
    x = x0 + (x1 - x0) * t
    y = y0 * (1 - t) ** 2 * (1 + 2 * t) + y1 * t ** 2 * (3 - 2 * t)
    return x, y


def _draw_ranking_bump(ax: plt.Axes, scores_df: pd.DataFrame, comp: "Comparison") -> None:
    """Draw a single bump chart for one comparison into an existing axes."""
    s_a = (
        comp.condition_a.filter(scores_df)
        .groupby(MODEL_KEY)[comp.condition_a.score_key]
        .mean()
    )
    s_b = (
        comp.condition_b.filter(scores_df)
        .groupby(MODEL_KEY)[comp.condition_b.score_key]
        .mean()
    )
    common = s_a.index.intersection(s_b.index)
    ranks_a = s_a.loc[common].rank(ascending=False)
    ranks_b = s_b.loc[common].rank(ascending=False)
    n_models = len(common)
    fs = plt.rcParams.get("font.size", 9)

    for model in common:
        color = model_to_color(model)
        ra, rb = ranks_a[model], ranks_b[model]

        bx, by = _bezier_bump(0.0, ra, 1.0, rb)
        ax.plot(bx, by, color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.scatter([0, 1], [ra, rb], color=color, s=60, zorder=3,
                   edgecolors="white", linewidths=0.8)

        ax.text(-0.04, ra, model_to_plotting_name(model),
                ha="right", va="center", fontsize=fs, color=color)
        ax.text(1.04, rb, model_to_plotting_name(model),
                ha="left", va="center", fontsize=fs, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([comp.label_a or comp.condition_a.label,
                        comp.label_b or comp.condition_b.label])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(n_models + 0.5, 0.5)
    ax.set_yticks(range(1, n_models + 1))
    ax.set_ylabel("Rank  (1 = best)")
    ax.set_title(comp.name)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#dddddd")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False)


def plot_ranking_bump_individual(
    scores_df: pd.DataFrame,
    output_path: Path,
    comparisons: "list[Comparison] | None" = None,
):
    """One bump chart per comparison, with smooth Bezier curves between ranks.

    Each figure is saved as ``<comp.stem>_bump.pdf/.png`` under *output_path*.
    The S-curve shape makes crossings visually unambiguous and keeps the chart
    readable even when several lines intersect at similar rank values.
    """
    if comparisons is None:
        comparisons = [c for c in COMPARISONS if c.run_ranking]

    for comp in comparisons:
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        _draw_ranking_bump(ax, scores_df, comp)
        fig.tight_layout()
        _save_fig(fig, output_path, f"{comp.stem}_bump")


def plot_combined_ranking_figure(
    scores_df: pd.DataFrame,
    ranks_original: list,
    ranks_verified: list,
    rho_mean: float,
    tau_mean: float,
    output_path: Path,
    comparison: "Comparison | None" = None,
    stem: str = "combined_ranking_figure",
    font_size: float = 10.0, #9.0, orig
    title_size: float = 9.0,
) -> None:
    """Three-panel figure: bar chart (a), bump chart (b), bootstrap scatter (c).

    Designed to fill the full linewidth of a NeurIPS two-column paper.  All
    font sizes are uniform across panels, controlled via *font_size* /
    *title_size* which are applied through an rc_context so only this figure
    is affected.
    """
    if comparison is None:
        comparison = next(c for c in COMPARISONS if c.save_summary)

    with mpl.rc_context({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [3, 2, 1.5]})

        for ax, label in zip(axes, ("a", "b", "c")):
            ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                    fontweight="bold", fontsize=font_size + 2)

        _draw_bars_grouped(axes[0], scores_df, comparison.condition_a, comparison.condition_b,
                           label_a=comparison.label_a, label_b=comparison.label_b)
        _draw_ranking_bump(axes[1], scores_df, comparison)
        _draw_ranking_scatter(axes[2], ranks_original, ranks_verified, rho_mean, tau_mean,
                              label_a=comparison.label_a or comparison.condition_a.label,
                              label_b=comparison.label_b or comparison.condition_b.label,
                              swap=True)

        for ax in axes:
            ax.set_title("")

        fig.tight_layout(w_pad=0.5)
        _save_fig(fig, output_path, stem)


def plot_ranking_bump(
    scores_df: pd.DataFrame,
    output_path: Path,
    comparisons: "list[Comparison] | None" = None,
    stem: str = "ranking_bump",
):
    """Bump chart showing rank changes between condition_a and condition_b.

    One panel per comparison (default: all ``run_ranking=True`` entries in
    ``COMPARISONS``).  Each model is a line connecting its rank under
    condition_a (left) to its rank under condition_b (right).  Crossings
    reveal rank reorderings; line colour identifies the model.  Model names
    are annotated at the right endpoint of the rightmost panel and at the
    left endpoint of the leftmost panel; middle panels rely on colour alone.
    """
    if comparisons is None:
        comparisons = [c for c in COMPARISONS if c.run_ranking]

    n_comp = len(comparisons)
    fig, axes = plt.subplots(
        1, n_comp,
        figsize=(n_comp * 3.2, 4.2),
        sharey=True,
    )
    if n_comp == 1:
        axes = [axes]

    # Compute ranks per comparison without triggering the print side-effects of
    # mean_score_evaluation — we only need the ranking series here.
    all_ranks = []
    for comp in comparisons:
        s_a = (
            comp.condition_a.filter(scores_df)
            .groupby(MODEL_KEY)[comp.condition_a.score_key]
            .mean()
        )
        s_b = (
            comp.condition_b.filter(scores_df)
            .groupby(MODEL_KEY)[comp.condition_b.score_key]
            .mean()
        )
        common = s_a.index.intersection(s_b.index)
        all_ranks.append((
            s_a.loc[common].rank(ascending=False),
            s_b.loc[common].rank(ascending=False),
        ))

    n_models = max(len(ra) for ra, _ in all_ranks)

    for panel_idx, (ax, comp, (ranks_a, ranks_b)) in enumerate(
        zip(axes, comparisons, all_ranks)
    ):
        models = ranks_a.index.tolist()

        for model in models:
            color = model_to_color(model)
            ra, rb = ranks_a[model], ranks_b[model]

            ax.plot([0, 1], [ra, rb], color=color, linewidth=2.0,
                    solid_capstyle="round", zorder=2)
            ax.scatter([0, 1], [ra, rb], color=color, s=55, zorder=3,
                       edgecolors="white", linewidths=0.8)

            # Model names: left side of first panel, right side of last panel
            if panel_idx == 0:
                ax.text(-0.06, ra, model_to_plotting_name(model),
                        ha="right", va="center", fontsize=8.0, color=color)
            if panel_idx == n_comp - 1:
                ax.text(1.06, rb, model_to_plotting_name(model),
                        ha="left", va="center", fontsize=8.0, color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [comp.label_a or comp.condition_a.label,
             comp.label_b or comp.condition_b.label],
            fontsize=8.5,
        )
        ax.set_xlim(-0.5 if panel_idx == 0 else -0.15,
                    1.5 if panel_idx == n_comp - 1 else 1.15)
        ax.set_title(comp.name, fontsize=8.5)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.4, color="#dddddd")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Shared y-axis: rank 1 (best) at top
    axes[0].set_ylim(n_models + 0.5, 0.5)
    axes[0].set_yticks(range(1, n_models + 1))
    axes[0].set_ylabel("Rank  (1 = best)")

    fig.tight_layout()
    _save_fig(fig, output_path, stem)


def plot_score_correlation_matrix(scores_df: pd.DataFrame, output_path: Path):
    """Side-by-side Pearson correlation heatmaps for each evaluation protocol.

    Each panel shows the 5×5 Pearson r matrix among the final score and the four
    subscores (spatial, spatiotemporal, weighted-spatial, MSE) computed from that
    protocol's runs.  The two panels share a single colorbar so magnitudes are
    directly comparable.
    """
    sub_cols = SCORES_LIST  # [score_spatial, score_spatiotemporal, score_weighted_spatial, score_mse]
    col_labels = ["Final", "Spatial", "Spatiotmp.", "Wtd. Spatial", "MSE"]

    panels = [
        (
            CONDITION_ORIGINAL.label,
            CONDITION_ORIGINAL.filter(scores_df),
            CONDITION_ORIGINAL.score_key,
        ),
        (
            CONDITION_VERIFIED.label,
            CONDITION_VERIFIED.filter(scores_df),
            CONDITION_VERIFIED.score_key,
        ),
    ]

    # constrained_layout handles the shared colorbar correctly; tight_layout does not
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    im = None
    for ax, (title, df, final_key) in zip(axes, panels):
        cols = [final_key] + sub_cols
        corr = df[cols].corr()
        n = len(cols)

        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                text_color = "white" if abs(val) > 0.65 else "#333333"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=text_color,
                )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(col_labels, fontsize=8.5, rotation=30, ha="right")
        ax.set_yticklabels(col_labels, fontsize=8.5)
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(bottom=False, left=False)

    fig.colorbar(im, ax=axes.tolist(), shrink=0.82, pad=0.02, label="Pearson r")
    _save_fig(fig, output_path, "score_correlation_matrix")


def save_results(
    output_path: Path,
    scores_df: pd.DataFrame,
    ranking_results: dict,
    bootstrap_results: dict,
    seed: int,
    n_bootstrap: int,
):
    """Write all key results to output_path.

    Files produced
    --------------
    scores_per_run.csv          Raw per-run scores for every model/eval/fps/prompt combination.
    scores_aggregated.csv       Mean ± std (×100) of each score column, one row per group.
    rankings.csv                Original vs. verified rank per model, with rank delta and
                                point-estimate correlation coefficients.
    bootstrap_rank_stats.csv    Per-model bootstrap rank statistics (mean, std, median, 95% CI)
                                for both evaluation types.
    results_summary.json        Single JSON with all scalar statistics and run metadata.
    """
    import datetime

    def _native(v):
        """Convert numpy scalars/arrays to plain Python types for JSON serialisation."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        return v

    data = _subdir(output_path, "data")

    # ── 1. Raw per-run scores ────────────────────────────────────────────────
    scores_df.to_csv(data / "scores_per_run.csv", index=False)

    # ── 2. Aggregated scores (mean ± std ×100) ───────────────────────────────
    score_cols = [ORIG_SCORE_KEY, VERIFIED_SCORE_KEY] + SCORES_LIST
    agg = (
        scores_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY])[score_cols].agg(
            ["mean", "std"]
        )
        * 100
    )
    agg.columns = ["_".join(col) for col in agg.columns]
    agg.reset_index().to_csv(data / "scores_aggregated.csv", index=False)

    # ── 3. Point-estimate rankings ────────────────────────────────────────────
    ranking_df = ranking_results["ranking_df"].copy()
    ranking_df.to_csv(data / "rankings.csv")

    # ── 4. Bootstrap rank statistics ─────────────────────────────────────────
    rank_stat_rows = []
    for eval_label, rank_list in [
        ("original", bootstrap_results["ranks_original"]),
        ("verified", bootstrap_results["ranks_verified"]),
    ]:
        rv = pd.concat(rank_list, axis=1)
        for model in rv.index:
            vals = rv.loc[model].values
            rank_stat_rows.append(
                {
                    MODEL_KEY: model,
                    EVAL_KEY: eval_label,
                    "rank_mean": float(np.mean(vals)),
                    "rank_std": float(np.std(vals)),
                    "rank_median": float(np.median(vals)),
                    "rank_ci_2_5": float(np.percentile(vals, 2.5)),
                    "rank_ci_97_5": float(np.percentile(vals, 97.5)),
                }
            )
    pd.DataFrame(rank_stat_rows).to_csv(data / "bootstrap_rank_stats.csv", index=False)

    # ── 5. JSON summary ───────────────────────────────────────────────────────
    within_rho_orig = bootstrap_results["within_rho_original"]
    within_tau_orig = bootstrap_results["within_tau_original"]
    within_rho_ver = bootstrap_results["within_rho_verified"]
    within_tau_ver = bootstrap_results["within_tau_verified"]
    summary = {
        "metadata": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "models": [s[MODEL_KEY] for s in RANKING_EVAL_SETTINGS],
        },
        "point_estimate_ranking_correlation": {
            "spearman_rho": _native(ranking_results["spearman_rho"]),
            "spearman_p": _native(ranking_results["spearman_p"]),
            "kendall_tau": _native(ranking_results["kendall_tau"]),
            "kendall_p": _native(ranking_results["kendall_p"]),
        },
        "bootstrap_cross_evaluation_correlation": {
            "spearman_rho_mean": _native(bootstrap_results["rho_mean"]),
            "spearman_rho_ci_95": _native(bootstrap_results["ci_rho"]),
            "kendall_tau_mean": _native(bootstrap_results["tau_mean"]),
            "kendall_tau_ci_95": _native(bootstrap_results["ci_tau"]),
        },
        "bootstrap_within_evaluation_stability": {
            "original_rho_mean": float(np.mean(within_rho_orig)),
            "original_rho_ci_95": _native(np.percentile(within_rho_orig, [2.5, 97.5])),
            "original_tau_mean": float(np.mean(within_tau_orig)),
            "original_tau_ci_95": _native(np.percentile(within_tau_orig, [2.5, 97.5])),
            "verified_rho_mean": float(np.mean(within_rho_ver)),
            "verified_rho_ci_95": _native(np.percentile(within_rho_ver, [2.5, 97.5])),
            "verified_tau_mean": float(np.mean(within_tau_ver)),
            "verified_tau_ci_95": _native(np.percentile(within_tau_ver, [2.5, 97.5])),
        },
        "model_rankings": {
            "original": ranking_df["rank_original"].astype(int).to_dict(),
            "verified": ranking_df["rank_verified"].astype(int).to_dict(),
            "rank_delta": ranking_df["rank_delta"].astype(int).to_dict(),
        },
    }
    with open(data / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-IQ benchmark analysis")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BASEPATH,
        help="Root directory containing experiment result CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_PATH,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    exp_tables = get_experiment_tables(args.results_dir)
    scores_df = pd.DataFrame([tab.get_output_dict() for tab in exp_tables])

    pivot_df_meanstd = (
        scores_df.groupby([MODEL_KEY, EVAL_KEY, FPS_KEY, PROMPT_KEY])[
            SCORE_COLS_FULL
        ].agg(["mean", "std"])
        * 100
    )
    for col_set_name, cols in SCORES_COLS_DICT.items():
        print(f"\nGenerating LaTeX tables for {col_set_name}...")
        if col_set_name != "full-score":
            pprint(pivot_df_meanstd[cols])
        generate_latex_table(
            args.output_dir,
            pivot_df_meanstd,
            f"full_model_table_{col_set_name}.tex",
            cols,
        )
        generate_latex_table_sora2(
            scores_df,
            args.output_dir,
            f"sora2_model_table_{col_set_name}.tex",
            cols,
        )

    scores_filtered = filter_by_settings(scores_df, RANKING_EVAL_SETTINGS)

    # FPS is omitted from the groupby because RANKING_EVAL_SETTINGS pins exactly
    # one canonical FPS per model, so it would only add a redundant index level.
    pivot_filtered = (
        scores_filtered.groupby([MODEL_KEY, EVAL_KEY, PROMPT_KEY])[
            SCORE_COLS_FULL
        ].agg(["mean", "std"])
        * 100
    )
    for col_set_name, cols in SCORES_COLS_DICT.items():
        generate_latex_table(
            args.output_dir, pivot_filtered, f"filtered_model_table_{col_set_name}.tex", cols
        )

    scenario_var_df = build_scenario_variance_df(exp_tables)
    analyze_variance_shifts(scenario_var_df, args.output_dir)
    plot_variance_distributions(scenario_var_df, args.output_dir)
    plot_variance_scenario_scatter(scenario_var_df, args.output_dir)

    plot_score_correlation_matrix(scores_filtered, args.output_dir)
    plot_model_performance_bars(
        scores_filtered, VERIFIED_SCORE_KEY, args.output_dir, mode="original"
    )
    plot_model_performance_bars(
        scores_filtered, VERIFIED_SCORE_KEY, args.output_dir, mode="verified"
    )
    plot_ranking_bump(scores_filtered, args.output_dir)

    # ── Run all comparisons declared in COMPARISONS ───────────────────────────
    ranking_results:   dict[str, dict] = {}
    bootstrap_results: dict[str, dict] = {}

    for comp in COMPARISONS:
        print(f"\n── {comp.name} ──")
        comp_out = _subdir(args.output_dir, comp.stem)

        if comp.run_stats and comp.subscores:
            score_cols = [comp.condition_a.score_key] + list(comp.subscores)
            result = compare_conditions_stats(
                scores_filtered, comp.condition_a, comp.condition_b,
                score_cols=score_cols, alternative=comp.stats_alternative,
            )
            pprint(result)
            result.to_csv(_subdir(comp_out, "data") / f"{comp.stem}_stats.csv", index=False)
            with open(_subdir(comp_out, "stats") / f"{comp.stem}_stats.txt", "w") as fh:
                fh.write(result.to_string())

        if comp.run_impact_plot and comp.subscores:
            sub_display = {col: _subscore_display(col) for col in comp.subscores}
            _plot_slope_delta(
                scores_filtered, comp.condition_a, comp.condition_b,
                sub_display, comp_out, stem=comp.stem, title=comp.name,
                delta_label=comp.delta_label,
                label_a=comp.label_a, label_b=comp.label_b,
            )
            if comp.run_stats:
                plot_slope_delta_with_stats(
                    scores_filtered, comp.condition_a, comp.condition_b,
                    sub_display, comp_out, stem=comp.stem, title=comp.name,
                    stats_alternative=comp.stats_alternative,
                    delta_label=comp.delta_label,
                    label_a=comp.label_a, label_b=comp.label_b,
                )

        if comp.run_ranking:
            plot_model_performance_bars_combined(
                scores_filtered, comp_out,
                comp.condition_a, comp.condition_b,
                stem="model_performance_bars_combined",
            )
            plot_model_performance_bars_grouped(
                scores_filtered, comp_out,
                comp.condition_a, comp.condition_b,
                stem="model_performance_bars_grouped",
                label_a=comp.label_a, label_b=comp.label_b,
            )
            plot_ranking_bump_individual(scores_filtered, comp_out, comparisons=[comp])
            result = mean_score_evaluation(
                scores_filtered, comp.condition_a, comp.condition_b,
            )
            result["ranking_df"].to_csv(_subdir(comp_out, "data") / f"{comp.stem}_rankings.csv")
            pd.DataFrame([{
                "spearman_rho": result["spearman_rho"],
                "spearman_p":   result["spearman_p"],
                "kendall_tau":  result["kendall_tau"],
                "kendall_p":    result["kendall_p"],
            }]).to_csv(_subdir(comp_out, "data") / f"{comp.stem}_ranking_corr.csv", index=False)
            ranking_results[comp.stem] = result

        if comp.run_bootstrap:
            result = run_bootstrap_analysis(
                exp_tables, args.n_bootstrap, comp_out,
                comparison_conditions=(comp.condition_a, comp.condition_b),
                label_a=comp.label_a or comp.condition_a.label,
                label_b=comp.label_b or comp.condition_b.label,
            )
            pd.DataFrame([{
                "rho_mean":    result["rho_mean"],
                "tau_mean":    result["tau_mean"],
                "rho_ci_2_5":  result["ci_rho"][0],
                "rho_ci_97_5": result["ci_rho"][1],
                "tau_ci_2_5":  result["ci_tau"][0],
                "tau_ci_97_5": result["ci_tau"][1],
            }]).to_csv(
                _subdir(comp_out, "data") / f"{comp.stem}_bootstrap_corr.csv",
                index=False,
            )
            bootstrap_results[comp.stem] = result

    primary = next((c for c in COMPARISONS if c.save_summary), None)
    primary_ranking   = ranking_results.get(primary.stem)   if primary else None
    primary_bootstrap = bootstrap_results.get(primary.stem) if primary else None

    if primary_ranking and primary_bootstrap:
        primary_out = _subdir(args.output_dir, primary.stem)
        save_results(
            primary_out,
            scores_df,
            primary_ranking,
            primary_bootstrap,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
        )
        plot_combined_ranking_figure(
            scores_filtered,
            primary_bootstrap["ranks_original"],
            primary_bootstrap["ranks_verified"],
            primary_bootstrap["rho_mean"],
            primary_bootstrap["tau_mean"],
            primary_out,
            comparison=primary,
        )


if __name__ == "__main__":
    main()
