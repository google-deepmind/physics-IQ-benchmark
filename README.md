<p align="center">
  <img src="assets/joint_duck.png" width="60%" alt="Physics-IQ and Physics-IQ Verified logos">
</p>

[Leaderboard](#leaderboard) | [Quick Start](#quick-start) | [Physics-IQ Verified Workflow](#physics-iq-verified-workflow) | [Citation](#citation) | [License](#license-and-disclaimer)

# Physics-IQ and Physics-IQ Verified: Benchmarking physical understanding in generative video models

Physics-IQ is a high-quality, realistic, and comprehensive benchmark dataset for evaluating physical understanding in generative video models.
Building on this foundation, Physics-IQ Verified contains improvements w.r.t. prompt and metric quality.

This repository contains the workflow for both Physics-IQ Verified (recommended benchmark variant) and the original Physics-IQ benchmark.

Original Physics-IQ website: [physics-iq.github.io](https://physics-iq.github.io/)<br>
Physics-IQ Verified website: [physics-iq-verified.anates.ai](https://physics-iq-verified.anates.ai)

### Key Features:
- **Real-world videos**: All videos are captured with high-quality cameras, not rendered.
- **Diverse scenarios**: Covers a wide range of physical phenomena, including collisions, fluid dynamics, gravity, material properties, light, shadows, magnetism, and more.
- **Multiple perspectives**: Each scenario is filmed from 3 different angles.
- **Variations**: Each scenario is recorded twice to capture natural physical variations.
- **High resolution and frame rate**: Videos are recorded at 3840 × 2160 resolution and 30 frames per second.

<p align="center">
  <img src="assets/teaser1.gif" width="23%" alt="Teaser 1">
  <img src="assets/teaser2.gif" width="23%" alt="Teaser 2">
  <img src="assets/teaser3.gif" width="23%" alt="Teaser 3">
  <img src="assets/teaser4.gif" width="23%" alt="Teaser 4">
  <img src="assets/teaser5.gif" width="23%" alt="Teaser 5">
  <img src="assets/teaser6.gif" width="23%" alt="Teaser 6">
  <img src="assets/teaser7.gif" width="23%" alt="Teaser 7">
  <img src="assets/teaser8.gif" width="23%" alt="Teaser 8">
</p>

---
## Leaderboard
The best possible score on Physics-IQ is 100.0%, this score would be achieved by physically realistic videos that differ only in physical randomness but adhere to all tested principles of physics.
### Physics-IQ Verified Leaderboard
If you test your model on Physics-IQ Verified and would like your score/paper/model to be featured here in this table, feel free to open a pull request that adds a row to the table and we'll be happy to include it!

The leaderboard is also hosted at: [physics-iq-verified.anates.ai](https://physics-iq-verified.anates.ai)

| # | Model | input type | Physics-IQ verified | date added (YYYY-MM-DD) |
|---|---|---|---|---|
| 1 | [Magi-1 24B + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) <small>(op)</small> | multiframe (v2v) | **58.2** <small>± 1.8</small> <br> 🥇 v2v | 2026-06-19 |
| 2 | Magi-1 24B <small>(op)</small> | multiframe (v2v) | **48.4** <small>± 1.1</small>  <br> 🥈 v2v| 2026-06-19 |
| 3 | Cosmos3-Super-Image2Video | i2v | **39.5** <small>± 0.8</small> <br> 🥇 i2v | 2026-06-18 |
| 4 | Grok Imagine Video | i2v | **34.8** <small>± 0.6</small> <br> 🥈 i2v | 2026-06-17 |
| 5 | [Magi-1 24B + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) <small>(op)</small> | i2v | **33.7** <small>± 1.4</small> <br> 🥉 i2v | 2026-06-19 |
| 6 | Hunyuan Video 1.5 | i2v | **33.4** <small>± 0.8</small> | 2026-06-17 |
| 7 | Wan 2.2 | i2v | 32.2 <small>± 0.6</small> | 2026-06-17 |
| 8 | Cosmos3-Nano | i2v | 30.3 <small>± 0.6</small> | 2026-06-18 |
| 9 | Magi-1 24B <small>(op)</small> | i2v | 30.2 <small>± 1.1</small> | 2026-06-19 |
| 10 | Sora 2 | i2v | 26.5 <small>± 0.8</small> | 2026-06-17 |
| 11 | P-Video | i2v | 25.3 <small>± 1.8</small> | 2026-06-17 |

For details on the Physics-IQ Verified metrics, see the [arXiv report](https://arxiv.org/abs/2606.18943).

Unless specified by `op` for original prompt in the entry, all reported scores use best-practice-prompts (`bpp`) based on a custom templater for each specific model.
> Rules: 
>1. One run is sufficient to be included on the verified leaderboard. In general, we recommend to use 4 runs reporting mean and standard deviation. To claim SOTA, reporting standard deviation across 4 runs is required.


### Physics-IQ Original Leaderboard

If you test your model on Physics-IQ Original and would like your score/paper/model to be featured here in this table, feel free to open a pull request that adds a row to the table and we'll be happy to include it!

| **#** | **Model** | **input type** | **Physics-IQ score** | **date added (YYYY-MM-DD)** |
| -- | --- | --- | --- | --- |
| 1 | [Magi-1 + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) reported [here](https://christianinterno.github.io/GeoPhys/) | multiframe (v2v) | **64.5 %** :1st_place_medal: v2v | 2026-06-17 |
| 2 | [Cosmos3-Super + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | multiframe (v2v) | **63.4 %** :2nd_place_medal: v2v | 2026-05-26 |
| 3 | [Magi-1 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553) | multiframe (v2v) | **62.6 %** :3rd_place_medal: v2v | 2025-10-28 |
| 4 | [Cosmos3-Super](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | multiframe (v2v) | 59.7 % | 2026-05-26 |
| 5 | [Cosmos3-Nano + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | multiframe (v2v) | 57.7 % | 2026-05-26 |
| 6 | [Magi-1](https://arxiv.org/abs/2505.13211) reported [here](https://arxiv.org/pdf/2505.13211) | multiframe (v2v) | 56.0 % | 2025-04-21 |
| 7 | [Cosmos3-Nano](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | multiframe (v2v) | 50.2 % | 2026-05-26 |
| 8 | [Cosmos3-Super + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v | 48.9 % :1st_place_medal: i2v | 2026-05-26 |
| 9 | [Sora2 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553) | i2v | 46.4 % :2nd_place_medal: i2v | 2026-04-01 |
| 10 | [Wan2.2 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553) | i2v | 44.4 % :3rd_place_medal: i2v | 2026-04-01 |
| 11 | [Cosmos3-Super](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v | 43.8 % | 2026-05-26 |
| 12 | [Cosmos3-Nano + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v | 43.8 % | 2026-05-26 |
| 13 | [Sora2](https://openai.com/index/sora-2/) reported [here](https://arxiv.org/abs/2601.10553) | i2v | 42.3 % | 2026-04-01 |
| 14 | [Cosmos3-Nano](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v | 40.2 % | 2026-05-26 |
| 15 | [Magi-1 + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) reported [here](https://christianinterno.github.io/GeoPhys/) | i2v | 38.6 % | 2026-06-17 |
| 16 | [Wan2.2](https://github.com/Wan-Video/Wan2.2) reported [here](https://arxiv.org/abs/2601.10553) | i2v | 38.3 % | 2026-04-01 |
| 17 | [Magi-1 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553) | i2v | 36.9 % | 2025-10-28 |
| 18 | [Video-GPT](https://arxiv.org/abs/2505.12489) reported [here](https://arxiv.org/abs/2505.12489) | multiframe (v2v) | 35.0 % | 2025-05-22 |
| 19 | [CogVideoX-5B + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) reported [here](https://christianinterno.github.io/GeoPhys/) | i2v | 34.1 % | 2026-06-17 |
| 20 | [Wan2.1 14B + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) reported [here](https://christianinterno.github.io/GeoPhys/) | i2v | 34.0 % | 2026-06-17 |
| 21 | [Magi-1 4.5B + GeoPhys (BoN)](https://christianinterno.github.io/GeoPhys/) reported [here](https://christianinterno.github.io/GeoPhys/) | i2v | 34.0 % | 2026-06-17 |
| 22 | [CogVideoX-5b](https://github.com/ved015/CogVideoX-5b-Physics_iq_benchmarking) reported [here](https://github.com/ved015/CogVideoX-5b-Physics_iq_benchmarking) | i2v | 32.3 % | 2026-01-06 |
| 23 | [Magi-1](https://arxiv.org/abs/2505.13211) reported [here](https://arxiv.org/pdf/2505.13211) | i2v | 30.2 % | 2025-04-21 |
| 24 | [VideoPoet](https://arxiv.org/abs/2312.14125) reported [here](https://arxiv.org/abs/2501.09038) | multiframe (v2v) | 29.5 % | 2025-02-19 |
| 25 | [Lumiere](https://arxiv.org/abs/2401.12945) reported [here](https://arxiv.org/abs/2501.09038) | multiframe (v2v) | 23.0 % | 2025-02-19 |
| 26 | [Runway Gen 3](https://runwayml.com/research/introducing-gen-3-alpha) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 22.8 % | 2025-02-19 |
| 27 | [VideoPoet](https://arxiv.org/abs/2312.14125) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 20.3 % | 2025-02-19 |
| 28 | [Lumiere](https://arxiv.org/abs/2401.12945) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 19.0 % | 2025-02-19 |
| 29 | [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 14.8 % | 2025-02-19 |
| 30 | [Pika](https://pika.art/) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 13.0 % | 2025-02-19 |
| 31 | [Sora](https://openai.com/sora/) reported [here](https://arxiv.org/abs/2501.09038) | i2v | 10.0 % | 2025-02-19 |
 

*Note to early adopters of the benchmark: results from the paper were finalized on February 19, 2025; if you used the toolbox before please re-run since we changed and improved a few aspects. Likewise, if you downloaded the dataset before that date, it is recommended to re-download it, ensuring the ground truth video masks have a duration of five seconds.*

</details>

---

## Quick Start

Choose one benchmark:

- [**Physics-IQ Verified Workflow**](#physics-iq-verified-workflow): recommended benchmark with improved prompts, masks, and scoring. This is the default when running `physiq/run_physics_iq.py`.
- [**Physics-IQ Original Workflow**](#physics-iq-original-workflow): original Physics-IQ benchmark. Use `--original_physics_iq` when evaluating.

## Physics-IQ Verified Workflow
<details>
<a id="physics-iq-verified-workflow"></a>

### A. Download Physics-IQ Verified

Download the verified benchmark from the [Physics-IQ Verified Hugging Face dataset](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified).

Note: Access requests are approved automatically.

Install the Hugging Face CLI if it is not already present:

```bash
pip install -U huggingface_hub
```

Download Physics-IQ Verified into the desired destination folder:

```bash
hf download Anates-Labs-Research/Physics-IQ-Verified \
  --repo-type dataset \
  --local-dir physics-IQ-benchmark-verified
```

Ensure you have downloaded and placed the `physics-IQ-benchmark-verified` dataset in your working directory. This dataset must include 30FPS videos and can optionally include your desired FPS. If you downloaded the dataset from the link above, it should contain all provided FPS variants (30FPS, 24FPS, 16FPS, 8FPS). If your desired FPS does not exist in the dataset already, it will be automatically generated. The folder should have the following structure:

```plaintext
physics-IQ-benchmark-verified/
├── full-videos/
│   └── take-1/
│       └── 30FPS/
│           ├── 0001_full-videos_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
│           ├── 0002_full-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
│           └── ...
├── split-videos/
│   └── testing/
│       └── 30FPS/
│           ├── 0001_testing-videos_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
│           ├── 0002_testing-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
│           └── ...
├── switch-frames/
│   ├── 0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg
│   ├── 0002_switch-frames_anyFPS_perspective-center_trimmed-ball-and-block-fall.jpg
│   └── ...
└── video-masks/
    └── real/
        └── 30FPS/
            ├── 0001_video-masks_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
            ├── 0002_video-masks_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
            └── ...
```

### B. Set Up Environment

**Option A — uv (recommended):**


```bash
uv sync
```

<details>
  <summary>Installing uv</summary>
Install uv according to [Astral documentation](https://docs.astral.sh/uv/getting-started/installation):

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

or via pip:
```bash
pip install uv
```
</details>

**Option B — pip:**

```bash
pip install .
```

To also install development tools (formatter, test runner, notebooks):

```bash
pip install ".[dev]"
```

> Contributors who need an editable install can use `pip install -e ".[dev]"` instead.

System requirements: tested on Linux; requires `ffprobe` (install with `sudo apt-get install ffmpeg`).

> **Note for pip users:** replace `uv run` with `python` in all commands below.

### C. Choose Prompt Template

**C1. Why this matters.**

Prompting conventions differ across video models. To evaluate models fairly, use the prompt template that best matches each model's expected input style instead of forcing every model into the same wording. You can either use one of the existing templates below or write your own model-specific templater. For example, OpenAI provides an excellent [Sora 2 prompting guide](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide) that can be used as a reference when designing a templater.

**C2. Prompt settings.**

Physics-IQ Verified uses two prompt settings:
- `bpp` uses a model-specific prompt (or the base version) produced by a templater stored inside `descriptions/best_practice`.
- `op` uses the original `descriptions/descriptions_original.csv` prompts.

**C3. Existing templates.**

For the bpp settings, the base templated descriptions can be found in `descriptions/best_practice/descriptions_base.csv`. For models with specific prompting guidelines, model-optimised descriptions can be generated using `uv run physiq/generate_descriptions.py {model_name}`:

| File | Optimised for |
|---|---|
| `descriptions_pvideo.csv` | P-Video (Pruna AI) |
| `descriptions_sora2.csv` | Sora 2 (OpenAI) |

**C4. Add a new templater (optional, recommended for new models).**

<details>
  <summary>Adding a new templater for your model</summary>

1. Open `physiq/templater/physiq_verified.py` and add a class decorated with `@register("name")`:

```python
from templater.base import BaseTemplater, register

@register("mymodel")
class MyModelTemplater(BaseTemplater):
    def generate_prompt(self, identifier) -> str:
        action = self.get_subjectaction_description(identifier)
        scene = self.get_scene_description(identifier)
        setup = self.get_scenesetup_description(identifier)
        # compose however your model expects it
        return f"{action} {scene} {setup}"
```

2. Generate the descriptions CSV:

```bash
uv run physiq/generate_descriptions.py mymodel
# writes descriptions/best_practice/descriptions_mymodel.csv
```

Available helper methods on `BaseTemplater`:
- `get_subjectaction_description(id)` — what happens in the scene
- `get_scene_description(id)` — static scene setup
- `get_scenesetup_description(id)` — pre-action state (optional, may be empty)
- `self.camera_description` / `self.style_description` / `self.action_description` — fixed boilerplate strings

</details>

**C5. Generate a descriptions CSV.**

To regenerate or add a new variant:

```bash
uv run physiq/generate_descriptions.py sora2   # or pvideo, base
```

This writes a model-specific descriptions CSV, for example:

```plaintext
descriptions/best_practice/descriptions_sora2.csv
```

with the same evaluation columns as the base descriptions file:

```csv
scenario,description,category,generated_video_name
0001_perspective-left_take-1_trimmed-ball-and-block-fall.mp4,"Style: ...",Solid Mechanics,0001_perspective-left_trimmed-ball-and-block-fall.mp4
```

### D. Generate Videos

**D1. Choose input mode.**

First choose the input mode used by your model.

<details open>
  <summary>Image-to-video models (I2V)</summary>

1. Use initial frames from `physics-IQ-benchmark-verified/switch-frames`.
2. If your model uses text input, use the descriptions CSV selected or generated in Step C. Only the first 198 rows marked as `take-1` are needed for generation.
3. Save generated videos with the benchmark ID prefix:

```plaintext
<model_run_folder>/0001_perspective-left_trimmed-ball-and-block-fall.mp4
```

</details>

<details>
  <summary>Multiframe-to-video models (V2V)</summary>

1. Use conditioning videos from `physics-IQ-benchmark-verified/split-videos/conditioning-videos`.
2. If your model also accepts text input, use the descriptions CSV selected or generated in Step C.
3. Ensure the frame rate matches the benchmark FPS you will evaluate at.
4. Save generated videos with the benchmark ID prefix:

```plaintext
<model_run_folder>/0001_perspective-left_trimmed-ball-and-block-fall.mp4
```

</details>

**D2. Name each model-run folder.**

Save generated videos in one directory per model run. For leaderboard-style reporting, generate four independent runs for each model and prompt setting. The aggregate leaderboard score in Step G is computed as the mean ± standard deviation across these four runs. Use the folder name to encode both the prompt setting and the run number:

```plaintext
<model_name>-<prompt_setting>-run_<run_number>
```

The prompt setting should be `bpp` for model-specific benchmark prompts or `op` for original prompts. The run number should use `run_01` through `run_04` for the standard four-run benchmark setup. Filenames may vary, but each video must keep the unique ID prefix from the benchmark (`0001_`, ..., `0198_`). Using descriptive benchmark-style names is recommended.


### E. Trim Videos

Before running evaluation, trim all generated videos to exactly 5 seconds. Videos of any other duration are incompatible with the benchmark. If you are running V2V, do not include the 3-second conditioning segment, only the generated 5 seconds.

You can use the repo-local `generated_videos_5s/` folder for trimmed outputs or store them externally and pass those folders to `--input_folders`.

Example trimmed video folder:

```plaintext
generated_videos_5s/
├── <model_name>-bpp-run_01/
│   ├── 0001_perspective-left_trimmed-ball-and-block-fall.mp4
│   ├── 0002_perspective-center_trimmed-ball-and-block-fall.mp4
│   └── ...
├── <model_name>-bpp-run_02/
│   └── ...
├── <model_name>-bpp-run_03/
│   └── ...
└── <model_name>-bpp-run_04/
    └── ...
```

<details>
  <summary>Original-prompt (`op`) trimmed folder example</summary>

```plaintext
generated_videos_5s/
├── <model_name>-op-run_01/
│   ├── 0001_perspective-left_trimmed-ball-and-block-fall.mp4
│   ├── 0002_perspective-center_trimmed-ball-and-block-fall.mp4
│   └── ...
├── <model_name>-op-run_02/
│   └── ...
├── <model_name>-op-run_03/
│   └── ...
└── <model_name>-op-run_04/
    └── ...
```

</details>

```bash
mkdir -p generated_videos_5s/<model_name>-bpp-run_01

for v in generated_videos/<model_name>-bpp-run_01/*.mp4; do
  ffmpeg -y -i "$v" \
    -t 5 \
    -r 24 \
    "generated_videos_5s/<model_name>-bpp-run_01/$(basename "$v")"
done
```

### F. Run Evaluation

Verified evaluation is the default behavior of `physiq/run_physics_iq.py`. This step reports two per-run score variants for each input folder: the original score and the verified score. For Physics-IQ Verified leaderboard reporting, use the verified score.

```bash
uv run physiq/run_physics_iq.py \
  --input_folders \
    generated_videos_5s/<model_name>-bpp-run_01 \
    generated_videos_5s/<model_name>-bpp-run_02 \
    generated_videos_5s/<model_name>-bpp-run_03 \
    generated_videos_5s/<model_name>-bpp-run_04 \
  --output_folder <output_dir> \
  --descriptions_file <descriptions_file> \
  --benchmark_base_folder <folder_containing_physics-IQ-benchmark-verified>
```

**Parameters:**
- `--input_folders`: directories containing generated `.mp4` videos, with one directory per model run.
- `--output_folder`: directory where result CSV files and plots will be saved.
- `--descriptions_file`: path to the descriptions CSV used for the benchmark.
- `--benchmark_base_folder`: parent folder containing `physics-IQ-benchmark-verified`.

The evaluator writes one result CSV and one metrics JSON per input folder, using the input folder name as the file stem:

```plaintext
<output_dir>/
└── physics-IQ-benchmark-verified/
    └── results/
        ├── <model_name>-bpp-run_01.csv
        ├── <model_name>-bpp-run_01_metrics.json
        ├── <model_name>-bpp-run_02.csv
        ├── <model_name>-bpp-run_02_metrics.json
        ├── <model_name>-bpp-run_03.csv
        ├── <model_name>-bpp-run_03_metrics.json
        ├── <model_name>-bpp-run_04.csv
        ├── <model_name>-bpp-run_04_metrics.json
        ├── physics_IQ_score_Original_barplot.pdf # return the original score
        └── physics_IQ_score_Verified_barplot.pdf # returns the verified score for the verified leaderboard. 
```

The verified score printed by the evaluator is stored as `final_score_view` in each `_metrics.json` file.

### G. Aggregate Leaderboard Scores

Step F reports per-run original and verified score variants. 
To report a Physics-IQ Verified leaderboard score, use the verified score from each run and compute the mean and standard deviation across the standard four runs.
Report this as `score ± std` in the leaderboard table.

To do this, use `aggregate_runs_from_csvs.py` can be used as follows:
```bash
uv run physiq/aggregate_runs_from_csvs.py \
  <path>/<model_name>-bpp-run_01.csv \
  <path>/<model_name>-bpp-run_02.csv \
  <path>/<model_name>-bpp-run_03.csv \
  <path>/<model_name>-bpp-run_04.csv \
  --score-type verified
```

We also accept single run results, but we do recommend using 4 runs.

</details>


## Physics-IQ Original Workflow
<details>
<a id="physics-iq-original-workflow"></a>

### A. Download Physics-IQ Original

Download the original benchmark from the [Physics-IQ Google Cloud Storage link](https://console.cloud.google.com/storage/browser/physics-iq-benchmark), or install the `gcloud` SDK and run:

```bash
uv run physiq/download_physics_iq_data.py \
  --fps 30 --original_physics_iq\
  --benchmark_base_folder <download_parent>
```

Ensure you have downloaded and placed the `physics-IQ-benchmark` dataset in your working directory. This dataset must include 30FPS videos and can optionally include your desired FPS. If you downloaded the dataset from the link above, it should contain all provided FPS variants (30FPS, 24FPS, 16FPS, 8FPS). If your desired FPS does not exist in the dataset already, it will be automatically generated. The folder should have the following structure:

```plaintext
physics-IQ-benchmark/
├── full-videos/
│   └── take-1/
│       └── 30FPS/
│           └── ...
├── split-videos/
│   ├── conditioning-videos/
│   │   └── 30FPS/
│   │       ├── 0001_conditioning-videos_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
│   │       ├── 0002_conditioning-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
│   │       └── ...
│   └── testing-videos/
│       └── 30FPS/
│           ├── 0001_testing-videos_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
│           ├── 0002_testing-videos_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
│           └── ...
├── switch-frames/
│   ├── 0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg
│   ├── 0002_switch-frames_anyFPS_perspective-center_trimmed-ball-and-block-fall.jpg
│   └── ...
└── video-masks/
    └── real/
        └── 30FPS/
            ├── 0001_video-masks_30FPS_perspective-left_take-1_trimmed-ball-and-block-fall.mp4
            ├── 0002_video-masks_30FPS_perspective-center_take-1_trimmed-ball-and-block-fall.mp4
            └── ...
```

### B. Set Up Environment

Use the same environment setup as the verified workflow.

### C. Use Original Prompts

Use `descriptions/descriptions_original.csv` for original Physics-IQ prompts.

### D. Generate Videos

Use the same generated-video folder and filename conventions as the verified workflow, but source frames and conditioning videos from `physics-IQ-benchmark/` and use the original (op) descriptions from `descriptions/descriptions_original.csv`.

### E. Trim Videos

Trim generated videos to exactly 5 seconds before evaluation.

### F. Run Evaluation

Add `--original_physics_iq` to evaluate against the original benchmark:

```bash
uv run physiq/run_physics_iq.py \
  --input_folders \
    generated_videos_5s/<model_name>
  --output_folder <output_dir> \
  --descriptions_file descriptions/descriptions_original.csv \
  --benchmark_base_folder <folder_containing_physics-IQ-benchmark> \
  --original_physics_iq
```

The evaluator writes one result CSV and one metrics JSON per input folder, using the input folder name as the file stem:

```plaintext
<output_dir>/
└── physics-IQ-benchmark-verified/
    └── results/
        ├── <model_name>.csv
        ├── <model_name>.json
        ├── physics_IQ_score_Original_barplot.pdf # score for the original leaderboard
        └── physics_IQ_score_Verified_barplot.pdf # verified score on original data 
```

The original Physics-IQ score is then plotted in `physics_IQ_score_Original_barplot.pdf` and stored inside a correspondingly named json file under: `final_score_origround`
</details>

---


## Citation
If you think this project is helpful, please feel free to leave a star ⭐️

**Original Physics-IQ:**
```latex
@article{motamed2026physics,
  title={Do generative video models understand physical principles?},
  author={Saman Motamed and Laura Culp and Kevin Swersky and Priyank Jaini and Robert Geirhos},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={948--958},
  year={2026}
}
```

**Physics-IQ Verified (which builds on the paper above):**
```latex
@article{radsch2026verified,
  author  = {Rädsch, Tim and Asano, Yuki M. and Kuehne, Hilde and Bauer, Stefan and Jaini, Priyank and Geirhos, Robert and Lüth, Carsten T.},
  title   = {Physics-IQ Verified},
  journal = {arXiv preprint arXiv:2606.18943},
  year    = {2026},
}
```


## License and disclaimer

### Physics-IQ

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
