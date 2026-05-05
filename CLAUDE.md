# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv run` instead of `python` or `python3`.**

```bash
# Run the full evaluation pipeline on generated videos
uv run code/run_physics_iq.py \
  --input_folders <model_video_dir> \
  --output_folder <output_dir> \
  --descriptions_file <descriptions.csv>

# Run statistical analysis and produce all figures/tables
uv run code/analysis.py \
  --results-dir /path/to/results \
  --output-dir ./output \
  --n-bootstrap 500 \
  --seed 1234

# Run GT experiment analysis (deterministic, no std)
uv run code/analysis_gt_runs.py \
  --results-dir /path/to/results \
  --output-dir ./output
```

There are no automated tests. Validate changes by running the analysis script against the results directory and checking that output files are produced correctly.

## Architecture

### Two distinct pipelines

**Evaluation pipeline** (run once per model, computationally heavy):
```
run_physics_iq.py
  └─ binary_mask_generator.py   — generate binary masks from videos
  └─ fps_changer.py             — resample to target FPS
  └─ calculate_and_write_metrics_to_csv.py  — compute spatial/spatiotemporal IOU + MSE per scenario
  └─ calculate_iq_score.py      — legacy score calculation, also defines VIEWS and parse_list_of_floats
```

**Analysis pipeline** (run repeatedly on existing result CSVs):
```
analysis.py / analysis_gt_runs.py
  └─ calculate_iq_score_stable.py  — IQTable class (primary scoring interface)
  └─ plot_settings.py              — model display names and hex colors
```

### IQTable (calculate_iq_score_stable.py)

The central data structure. Wraps a per-scenario metrics DataFrame and computes Physics-IQ scores.

**Key design points:**
- Each row is one test scenario; each metric has three per-perspective columns (`_perspective-left/center/right`). `__init__` collapses these into cross-view means.
- Two column types: *list columns* (spatiotemporal IOU and MSE store per-frame sequences as strings) vs *scalar columns* (spatial IOU is a single float per scenario). `get_list_keys()` / `get_scalar_keys()` separate these.
- **Scoring formula**: IOU metrics → divided by physical variance; MSE → subtracted from physical variance. Physical variance is the empirical variation of ground-truth outcomes across repeated real-world trials and serves as a difficulty-normalisation term.
- **Two final-score variants**: `final_score_orig` clips only the aggregated total; `final_score_stable` clips each component before aggregating. The stable variant (`VERIFIED_SCORE_KEY`) is the primary reported score.
- `df.copy()` in `__init__` — the class owns its data; mutating the source DataFrame after construction has no effect.

### analysis.py configuration

Key constants to edit when adding models or changing evaluation settings:

| Constant | Purpose |
|---|---|
| `MODEL_NAMES` | All model identifiers the script will recognise in filenames |
| `RANKING_EVAL_SETTINGS` | Canonical (model, FPS) pair used for each model in the ranking comparison |
| `SORA2_MODELS` | Subset of `MODEL_NAMES` for the Sora 2 variant table |
| `BASEPATH` / `OUTPUT_PATH` | Default paths, overridable with `--results-dir` / `--output-dir` |
| `COMPARISON_KEYS` | The two (eval_type, prompt) pairs compared in the bootstrap analysis |

### Output directory structure

All writes go through `_subdir(output_path, name)` which creates the folder on demand:

```
<output-dir>/
  figures/    — PDF figures (for \includegraphics in LaTeX)
  preview/    — PNG previews of the same figures
  tables/     — .tex files
  data/       — CSVs and results_summary.json
```

### Evaluation concepts

- **Prompts**: `op` = original (unverified), `bpp` = human-verified
- **Eval types**: `original` = original scoring pipeline, `verified_full` = stable scoring pipeline
- **Canonical comparison**: `original + op` vs `verified_full + bpp`
- **Physical variance**: scenario-level property (not model-dependent); used to normalise scores and compared across evaluation types in `analyze_variance_shifts()`
- **Bootstrap**: resamples at the *run level* (4 runs per model/setting), not at the scenario level

### Adding a new model

1. Add the model key to `MODEL_NAMES` in `analysis.py`
2. Add a `RANKING_EVAL_SETTINGS` entry with the canonical FPS
3. Add a `Model(...)` entry in `plot_settings.py` with `plotting_name` and `color`
4. Add the model to `latex_table.tex` with its conditioning type and resolution
