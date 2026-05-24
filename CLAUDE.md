# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv run` instead of `python` or `python3`.**

```bash
# Run the full evaluation pipeline on generated videos
uv run physiq/run_physics_iq.py \
  --input_folders <model_video_dir> \
  --output_folder <output_dir> \
  --descriptions_file <descriptions.csv>

# Run GT experiment analysis (deterministic, no std)
uv run physiq/analysis_gt_runs.py \
  --results-dir /path/to/results \
  --output-dir ./output

# Generate a model-specific descriptions CSV from a named templater
uv run physiq/generate_descriptions.py pvideo
uv run physiq/generate_descriptions.py sora2
uv run physiq/generate_descriptions.py base --no-action-suffix

# Run tests
uv run pytest physiq/tests/
```

## Architecture

### Three distinct pipelines

**Evaluation pipeline** (run once per model, computationally heavy):
```
physiq/run_physics_iq.py
  └─ binary_mask_generator.py   — generate binary masks from videos
  └─ fps_changer.py             — resample to target FPS
  └─ calculate_and_write_metrics_to_csv.py  — compute spatial/spatiotemporal IOU + MSE per scenario
  └─ calculate_iq_score.py      — legacy score calculation, also defines VIEWS and parse_list_of_floats
```

**GT analysis pipeline** (run on existing GT experiment CSVs):
```
physiq/analysis_gt_runs.py
  └─ calculate_iq_score_stable.py  — IQTable class (primary scoring interface)
  └─ plot_settings.py              — model display names and hex colors
```

**Descriptions pipeline** (run when descriptions source changes or a new templater is added):
```
physiq/generate_descriptions.py  — CLI: loads YAML, applies templater, writes descriptions_<name>.csv
  └─ dataset.py      — Benchmark.from_yaml() / to_dataframe() / build_original_descriptions()
  └─ templater/
       base.py            — REGISTRY dict, @register decorator, BaseTemplater ("base")
       physiq_verified.py — PVideoTemplater ("pvideo"), SoraTemplater ("sora2")
```

Output goes to `descriptions/model_specific/descriptions_<name>.csv`.

### IQTable (calculate_iq_score_stable.py)

The central data structure. Wraps a per-scenario metrics DataFrame and computes Physics-IQ scores.

**Key design points:**
- Each row is one test scenario; each metric has three per-perspective columns (`_perspective-left/center/right`). `__init__` collapses these into cross-view means.
- Two column types: *list columns* (spatiotemporal IOU and MSE store per-frame sequences as strings) vs *scalar columns* (spatial IOU is a single float per scenario). `get_list_keys()` / `get_scalar_keys()` separate these.
- **Scoring formula**: IOU metrics → divided by physical variance; MSE → subtracted from physical variance. Physical variance is the empirical variation of ground-truth outcomes across repeated real-world trials and serves as a difficulty-normalisation term.
- **Two final-score variants**: `final_score_orig` clips only the aggregated total; `final_score_stable` clips each component before aggregating. The stable variant (`VERIFIED_SCORE_KEY`) is the primary reported score.
- `df.copy()` in `__init__` — the class owns its data; mutating the source DataFrame after construction has no effect.

### Benchmark and Scene (dataset.py)

`Benchmark.from_yaml(path)` loads `descriptions/descriptions.yaml` into a list of `Scene` objects. `to_dataframe()` explodes scenes into one row per (perspective × take), including all template fields. `build_original_descriptions()` slices to the four columns used by the evaluation pipeline: `scenario`, `description`, `category`, `generated_video_name`.

`descriptions/descriptions.csv` is the canonical pre-generated export from `build_original_descriptions()`. The test `test_descriptions.py::test_yaml_generates_matching_descriptions_csv` asserts the two are in sync.

### Templater registry (templater/)

`REGISTRY` in `templater/base.py` maps short names to templater classes. Classes self-register with the `@register("name")` decorator. Currently registered:

| Name | Class | Format |
|---|---|---|
| `base` | `BaseTemplater` | Space-joined prose |
| `pvideo` | `PVideoTemplater` | Comma-separated, subject-action first |
| `sora2` | `SoraTemplater` | Structured sections with headers |

All templaters share the `BaseTemplater` interface: `generate_prompt(identifier)` returns the prompt string for a given scenario filename.

### Evaluation concepts

- **Prompts**: `op` = original (unverified), `bpp` = human-verified
- **Eval types**: `original` = original scoring pipeline, `verified_full` = stable scoring pipeline
- **Canonical comparison**: `original + op` vs `verified_full + bpp`
- **Physical variance**: scenario-level property (not model-dependent); used to normalise scores and compared across evaluation types in `analyze_variance_shifts()`
- **Bootstrap**: resamples at the *run level* (4 runs per model/setting), not at the scenario level

### Adding a new model

1. Add a `Model(...)` entry in `physiq/plot_settings.py` with `plotting_name` and `color`
2. Add the model to `latex_table.tex` with its conditioning type and resolution

### Adding a new templater

1. In `physiq/templater/physiq_verified.py`, add a class decorated with `@register("name")` that subclasses `BaseTemplater` and overrides `generate_prompt(identifier) -> str`
2. Run `uv run physiq/generate_descriptions.py <name>` to generate `descriptions/model_specific/descriptions_<name>.csv`
