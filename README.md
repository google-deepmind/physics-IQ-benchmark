<p align="center">
  <img src="assets/joint_duck.png" width="60%" alt="Physics-IQ and Physics-IQ Verified logos">
</p>

[Leaderboard](#leaderboard) | [Quick Start](#quick-start) | [Physics-IQ Verified Workflow](#physics-iq-verified-workflow) | [Citation](#citation) | [License](#license-and-disclaimer)

# Physics-IQ and Physics-IQ Verified: Benchmarking physical understanding in generative video models

Physics-IQ is a high-quality, realistic, and comprehensive benchmark dataset for evaluating physical understanding in generative video models.
Building on this foundation, Physics-IQ Verified improves upon the original benchmark w.r.t. prompt quality and metric improvements.


This repository contains the workflow for Physics-IQ Verified, the recommended benchmark variant.
It also retains support for the original Physics-IQ benchmark for comparison with earlier published results.



Original Physics-IQ website: [physics-iq.github.io](https://physics-iq.github.io/)<br>
Physics-IQ Verified website: TBD

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
If you test your model on Physics-IQ and would like your score/paper/model to be featured here in this table, feel free to open a pull request that adds a row to the table and we'll be happy to include it!

For details on the Physics-IQ Verified metrics, see the [metric definitions](docs/metric_definition_phys_iq_verified.pdf). The full Physics-IQ Verified report will be published soon.

| # | Model | input type | Phys-IQ verified | SP verified | ST verified | WS verified | MSE verified | date added (YYYY-MM-DD) |
|---|---|---|---|---|---|---|---|---|
| 1 🥇 | Grok Imagine Video | i2v | 34.8 <small>± 0.6</small> | 52.7 <small>± 0.9</small> | 21.4 <small>± 0.6</small> | 35.7 <small>± 1.0</small> | 29.6 <small>± 0.4</small> | 2026-05-22 |
| 2 🥈 | Hunyuan Video 1.5 | i2v | 33.4 <small>± 0.8</small> | 47.1 <small>± 1.2</small> | 26.9 <small>± 1.0</small> | 29.7 <small>± 0.6</small> | 30.0 <small>± 1.0</small> | 2026-05-22 |
| 3 🥉 | Wan 2.2 | i2v | 32.2 <small>± 0.6</small> | 51.1 <small>± 1.0</small> | 20.5 <small>± 0.7</small> | 28.5 <small>± 0.7</small> | 28.9 <small>± 0.4</small> | 2026-05-22 |
| 4 | Sora 2 | i2v | 26.5 <small>± 0.8</small> | 37.3 <small>± 0.6</small> | 27.0 <small>± 2.2</small> | 26.9 <small>± 0.7</small> | 14.8 <small>± 0.6</small> | 2026-05-22 |
| 5 | P-Video | i2v | 25.3 <small>± 1.8</small> | 38.6 <small>± 2.2</small> | 16.4 <small>± 2.4</small> | 22.9 <small>± 1.8</small> | 23.3 <small>± 1.1</small> | 2026-05-22 |

The reported scores use best-practice-prompts (`bpp`) based on a custom templater for each specific model.


### Physics-IQ Original Leaderboard

If you test your model on Physics-IQ and would like your score/paper/model to be featured here in this table, feel free to open a pull request that adds a row to the table and we'll be happy to include it!

| **#** | **Model** | **input type** | **Physics-IQ score** | **date added (YYYY-MM-DD)** |
| -- | --- | --- | --- | --- |
| 1 | [Cosmos3-Super + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | multiframe (v2v) | **63.4 %** :1st_place_medal: v2v | 2026-05-26 |
| 2 | [Magi-1 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553)                                                                                        | multiframe (v2v) | **62.6 %** :2nd_place_medal: v2v | 2025-10-28 | 
| 3 | [Cosmos3-Super](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)                  | multiframe (v2v) | **59.7 %** :3rd_place_medal: v2v | 2026-05-26 |
| 4 | [Cosmos3-Nano + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)  | multiframe (v2v) | 57.7 % | 2026-05-26 |
| 5 | [Magi-1](https://arxiv.org/abs/2505.13211) reported [here](https://arxiv.org/pdf/2505.13211)                                                                                                         | multiframe (v2v) | 56.0 % | 2025-04-21 |
| 6 | [Cosmos3-Nano](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)                   | multiframe (v2v) | 50.2 % | 2026-05-26 |
| 7 | [Cosmos3-Super + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v              | 48.9 % :1st_place_medal: i2v | 2026-05-26 |
| 8 | [Sora2 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553)                                                                                         | i2v              | 46.4 % :2nd_place_medal: i2v | 2026-04-01 |
| 9 | [Wan2.2 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553)                                                                                        | i2v              | 44.4 % :3rd_place_medal: i2v | 2026-04-01 |
| 10 | [Cosmos3-Super](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)                 | i2v              | 43.8 % | 2026-05-26 |
| 11 | [Cosmos3-Nano + WMReward (BoN)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | i2v              | 43.8 % | 2026-05-26 |
| 12 | [Sora2](https://openai.com/index/sora-2/) reported [here](https://arxiv.org/abs/2601.10553)                                                                                                         | i2v              | 42.3 % | 2026-04-01 |
| 13 | [Cosmos3-Nano](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) reported [here](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)                  | i2v              | 40.2 % | 2026-05-26 |
| 14 | [Wan2.2](https://github.com/Wan-Video/Wan2.2) reported [here](https://arxiv.org/abs/2601.10553)                                                                                                     | i2v              | 38.3 % | 2026-04-01 |
| 15 | [Magi-1 + WMReward (BoN)](https://arxiv.org/abs/2601.10553) reported [here](https://arxiv.org/abs/2601.10553)                                                                                       | i2v              | 36.9 % | 2025-10-28 |
| 16 | [Video-GPT](https://arxiv.org/abs/2505.12489) reported [here](https://arxiv.org/abs/2505.12489)                                                                                                     | multiframe (v2v) | 35.0 % | 2025-05-22 |
| 17 | [CogVideoX-5b](https://github.com/ved015/CogVideoX-5b-Physics_iq_benchmarking) reported [here](https://github.com/ved015/CogVideoX-5b-Physics_iq_benchmarking)                                      | i2v              | 32.3 % | 2026-01-06 |
| 18 | [Magi-1](https://arxiv.org/abs/2505.13211) reported [here](https://arxiv.org/pdf/2505.13211)                                                                                                        | i2v              | 30.2 % | 2025-04-21 |
| 19 | [VideoPoet](https://arxiv.org/abs/2312.14125) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                     | multiframe (v2v) | 29.5 % | 2025-02-19 |
| 20 | [Lumiere](https://arxiv.org/abs/2401.12945) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                       | multiframe (v2v) | 23.0 % | 2025-02-19 |
| 21 | [Runway Gen 3](https://runwayml.com/research/introducing-gen-3-alpha) reported [here](https://arxiv.org/abs/2501.09038)                                                                             | i2v              | 22.8 % | 2025-02-19 |
| 22 | [VideoPoet](https://arxiv.org/abs/2312.14125) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                     | i2v              | 20.3 % | 2025-02-19 |
| 23 | [Lumiere](https://arxiv.org/abs/2401.12945) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                       | i2v              | 19.0 % | 2025-02-19 |
| 24 | [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) reported [here](https://arxiv.org/abs/2501.09038)                                                                                        | i2v              | 14.8 % | 2025-02-19 |
| 25 | [Pika](https://pika.art/) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                                         | i2v              | 13.0 % | 2025-02-19 |
| 26 | [Sora](https://openai.com/sora/) reported [here](https://arxiv.org/abs/2501.09038)                                                                                                                  | i2v              | 10.0 % | 2025-02-19 |

*Note to early adopters of the benchmark: results from the paper were finalized on February 19, 2025; if you used the toolbox before please re-run since we changed and improved a few aspects. Likewise, if you downloaded the dataset before that date, it is recommended to re-download it, ensuring the ground truth video masks have a duration of five seconds.*

</details>

---

## Quick Start

Choose one benchmark:

- [**Physics-IQ Verified Workflow**](#physics-iq-verified-workflow): recommended benchmark with improved prompts, masks, and scoring. This is the default when running `physiq/run_physics_iq.py`.
- [**Physics-IQ Original Workflow**](#physics-iq-original-workflow): legacy benchmark for comparison with older published results. Use `--original_physics_iq` when evaluating.

## Physics-IQ Verified Workflow

### A. Download Physics-IQ Verified

Download the verified benchmark from the [Physics-IQ Verified Google Cloud Storage link](https://drive.google.com/file/d/1K7sRbks4VNqmpejyB9K7nIl4XcNpNWzk/view).
<!-- TODO: Add our Download link here.-->

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
- `bpp` uses a model-specific benchmark prompt produced by a templater.
- `op` uses the original `descriptions/descriptions.csv` prompts.

**C3. Existing templates.**

The base descriptions are in `descriptions/descriptions.csv`. For models with specific prompting guidelines, model-optimised descriptions are available in `descriptions/model_specific/`:

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
# writes descriptions/model_specific/descriptions_mymodel.csv
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
descriptions/model_specific/descriptions_sora2.csv
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
        ├── physics_IQ_score_Original_barplot.pdf
        └── physics_IQ_score_Verified_barplot.pdf
```

The verified score printed by the evaluator is stored as `final_score_view` in each `_metrics.json` file.

### G. Aggregate Leaderboard Scores

Step F reports per-run original and verified score variants. To report a Physics-IQ Verified leaderboard score, use the verified score from each run and compute the mean and standard deviation across the standard four runs. Report this as `score ± std` in the leaderboard table.

## Physics-IQ Original Workflow
<details>
<a id="physics-iq-original-workflow"></a>
<summary><strong><big>Physics-IQ Original Workflow</big></strong></summary>

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

Use `descriptions/descriptions.csv` for original Physics-IQ prompts.

### D. Generate Videos

Use the same generated-video folder and filename conventions as the verified workflow, but source frames and conditioning videos from `physics-IQ-benchmark/`.

### E. Trim Videos

Trim generated videos to exactly 5 seconds before evaluation.

### F. Run Evaluation

Add `--original_physics_iq` to evaluate against the original benchmark:

```bash
uv run physiq/run_physics_iq.py \
  --input_folders \
    generated_videos_5s/<model_name>-op-run_01 \
    generated_videos_5s/<model_name>-op-run_02 \
    generated_videos_5s/<model_name>-op-run_03 \
    generated_videos_5s/<model_name>-op-run_04 \
  --output_folder <output_dir> \
  --descriptions_file descriptions/descriptions.csv \
  --benchmark_base_folder <folder_containing_physics-IQ-benchmark> \
  --original_physics_iq
```

### G. Aggregate Leaderboard Scores

Use the per-run scores from Step F and compute the mean and standard deviation across the standard four runs.

</details>

---


## Citation
If you think this project is helpful, please feel free to leave a star ⭐️

Please cite both papers if you use this benchmark.
<!-- TODO: finalize our publication here. -->
```latex
@article{motamed2026physics,
  title={Do generative video models understand physical principles?},
  author={Saman Motamed and Laura Culp and Kevin Swersky and Priyank Jaini and Robert Geirhos},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={948--958},
  year={2026}
}

@article{raedsch2026physics,
  title={Physics-IQ Verified},
  author={}
  journal={arXiv preprint},
  year=2026
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
