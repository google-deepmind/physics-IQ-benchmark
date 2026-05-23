<p align="center">
  <img src="assets/duckie.png" width="20%" alt="logo">
</p>

[Step A: Generating Videos](#step-a-generating-videos-for-physics-iq-test-cases-based-on-video-model) | [Step B: Evaluating Generated Videos](#step-b-evaluating-generated-videos-on-physics-iq-to-generate-benchmark-scores) | [Leaderboard](#leaderboard) | [Citation](#citation) | [License](#license-and-disclaimer)

# Physics-IQ Verified: Benchmarking physical understanding in generative video models

The original Physics-IQ is a high-quality, realistic, and comprehensive benchmark dataset for evaluating physical understanding in generative video models.
Physics-IQ Verfied proposes key improvements with respect to data quality, prompt descriptiveness and the score computation.



Physics-IQ website: [physics-iq.github.io](https://physics-iq.github.io/)
<!-- TODO: Add Physics-IQ Verified website here -->

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
### Leaderboard Verified
If you test your model on Physics-IQ and would like your score/paper/model to be featured here in this table, feel free to open a pull request that adds a row to the table and we'll be happy to include it!

<!-- TODO: Add values from paper here in table. -->
| Model             | Prompt | Phys-IQ verified | SP verified    | ST verified    | WS verified    | MSE verified   | date added (YYYY-MM-DD) |
|-------------------|--------|------------------|----------------|----------------|----------------|----------------|-------------------------|
| Grok Imagine Video | bpp    | 34.8 <small>± 0.6</small>   | 52.7 <small>± 0.9</small> | 21.4 <small>± 0.6</small> | 35.7 <small>± 1.0</small> | 29.6 <small>± 0.4</small> | 2026-05-22              |
|                   | op     | 32.7 <small>± 0.4</small>   | 49.8 <small>± 0.7</small> | 18.8 <small>± 0.6</small> | 34.0 <small>± 0.2</small> | 28.2 <small>± 0.4</small> | 2026-05-22              |
| Hunyuan Video 1.5 | bpp    | 33.4 <small>± 0.8</small>   | 47.1 <small>± 1.2</small> | 26.9 <small>± 1.0</small> | 29.7 <small>± 0.6</small> | 30.0 <small>± 1.0</small> | 2026-05-22              |
|                   | op     | 31.7 <small>± 0.9</small>   | 43.5 <small>± 1.1</small> | 25.4 <small>± 1.0</small> | 27.4 <small>± 1.1</small> | 30.4 <small>± 0.6</small> | 2026-05-22              |
| P-Video           | bpp    | 25.3 <small>± 1.8</small>   | 38.6 <small>± 2.2</small> | 16.4 <small>± 2.4</small> | 22.9 <small>± 1.8</small> | 23.3 <small>± 1.1</small> | 2026-05-22              |
|                   | op     | 23.8 <small>± 1.7</small>   | 35.5 <small>± 1.6</small> | 16.2 <small>± 2.9</small> | 21.4 <small>± 1.3</small> | 22.2 <small>± 2.0</small> | 2026-05-22              |
| Sora 2            | bpp    | 26.5 <small>± 0.8</small>   | 37.3 <small>± 0.6</small> | 27.0 <small>± 2.2</small> | 26.9 <small>± 0.7</small> | 14.8 <small>± 0.6</small> | 2026-05-22              |
|                   | op     | 15.7 <small>± 0.7</small>   | 23.6 <small>± 1.0</small> | 16.5 <small>± 1.0</small> | 15.4 <small>± 0.5</small> | 7.4 <small>± 0.6</small>  | 2026-05-22              |
| Wan 2.2           | bpp    | 32.2 <small>± 0.6</small>   | 51.1 <small>± 1.0</small> | 20.5 <small>± 0.7</small> | 28.5 <small>± 0.7</small> | 28.9 <small>± 0.4</small> | 2026-05-22              |
|                   | op     | 34.8 <small>± 0.7</small>   | 54.3 <small>± 0.9</small> | 21.2 <small>± 1.1</small> | 31.8 <small>± 0.7</small> | 31.9 <small>± 0.2</small> | 2026-05-22              |

- bpp is using a custom templater for the specific model
- op is using the description.csv prompts
- all models are i2v model

### Leaderboard Original
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

---

## Step A: Generating Videos for Physics-IQ Test Cases Based on Video Model

### 1. Download Benchmark Dataset

For the verified Benchmark visit the [Physics-IQ Verified Google Cloud Storage link]() to download the dataset.
<!-- TODO: Add our Download link here.-->
For the original Benchmark visit the [Physics-IQ Google Cloud Storage link](https://console.cloud.google.com/storage/browser/physics-iq-benchmark) to download the dataset.

Alternatively after installing gcloud SDK from [here](https://docs.cloud.google.com/sdk/docs/install-sdk) and run the following:
```bash
uv run physiq/download_physics_iq_data.py --fps FPS
```
Whether to download the original or verified Benchmark use the following flag:
- `--original_gt`: Whether to use the verified benchmark or the original benchmark.

- If your desired FPS already exists in the dataset, it will be downloaded.
- If it does not exist, the script will download 30 FPS files and generate your desired FPS videos by downsampling the 30 FPS version.

---

### 2. Running Video Model on Test Cases from Benchmark

This section explains how to generate videos using the provided benchmark and save them in the required format. Follow the instructions below based on your model type:

#### Text Descriptions / Prompts

The base descriptions are in `descriptions/descriptions.csv`. For models with specific prompting guidelines, model-optimised descriptions are available in `descriptions/model_specific/`:

| File | Optimised for |
|---|---|
| `descriptions_pvideo.csv` | P-Video (Pruna AI) |
| `descriptions_sora2.csv` | Sora 2 (OpenAI) |

To regenerate or add a new variant:

```bash
uv run physiq/generate_descriptions.py pvideo   # or sora2, base
```

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
# → writes descriptions/model_specific/descriptions_mymodel.csv
```

Available helper methods on `BaseTemplater`:
- `get_subjectaction_description(id)` — what happens in the scene
- `get_scene_description(id)` — static scene setup
- `get_scenesetup_description(id)` — pre-action state (optional, may be empty)
- `self.camera_description` / `self.style_description` / `self.action_description` — fixed boilerplate strings

</details>

#### 2.1 Image-to-Video Models (I2V)

<details>
  <summary>I2V steps</summary>

1. **Input Requirements**:
   - **Initial Frame**: Use frames from `physics-iq-benchmark/switch-frames`.
   - **Text Input (Optional)**: If required, use descriptions from `descriptions.csv` or `descriptions_model.csv`. Only the first 198 entries (marked as`take-1`) need to be used, feel free to ignore the `take-2` entries since they're not used for sampling from models. 

2. **Steps to Run**:
   - Generate videos using the initial frame (and text condition, if applicable).
   - Save generated videos in the following structure, using any filename as long as the unique ID prefix from the test videos is kept (`0001_`, ..., `0198_`):
     ```
     .model_name/{ID}_{anything-you-like}.mp4
     ```
   - Refer to the `generated_video_name` column in `descriptions.csv` for file naming conventions.

</details>

#### 2.2 Multiframe-to-Video Models (V2V)

<details>
  <summary>V2V steps</summary>

1. **Input Requirements**:
   - **Conditioning Frames**:
     - Available in `physics-iq-benchmark/split-videos/conditioning-videos`.
     - Ensure the correct frame rate: `30FPS`, `24FPS`, `16FPS`, or `8FPS`.
   - **Text Input (Optional)**: Use `descriptions.csv` or `descriptions_model.csv`.

2. **Steps to Run**:
   - Use conditioning frames to generate videos.
   - Save generated videos in the structure:
     ```
     model_name/{ID}_{perspective}_{scenario_name}.mp4
     example: model_name/{0001}_{perspective-left}_{trimmed-ball-and-block-fall}.mp4
     ```
   - Refer to the `generated_video_name` column in `descriptions.csv` for file naming conventions.

</details>

#### 2.3 Trim Generated Videos to 5 Seconds

⚠️ **IMPORTANT**: Before running the evaluation, you must trim all generated videos to **exactly 5 seconds**. Videos of any other duration are incompatible with the benchmark. If you're running V2V (=multiframe-to-video), please make sure you're not including the 3s conditioning video, only the model-generated 5 seconds.

**Example command to trim your videos using ffmpeg:**

This example is based on cropping to the first 5 seconds which is useful for I2V; for V2V please adapt this command if the generated videos include the 3s conditioning part.
```bash
# Create output directory for trimmed videos
mkdir -p generated_videos_5s

# Trim all videos to 5 seconds at desired frame rate
# Adjust the `-r 24` parameter to match your desired FPS (e.g. 8, 16, 24, or 30)
for v in generated_video_path/*.mp4; do
  ffmpeg -y -i "$v" \
    -t 5 \
    -r 24 \
    "generated_videos_5s/$(basename "$v")"
done
```

---

## Step B: Evaluating Generated Videos on Physics-IQ to Generate Benchmark Scores

### 1. Installation

**Option A — uv (recommended):**

```bash
pip install uv
uv sync
```

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

### 2. Dataset Placement

- Ensure you have downloaded and placed the `physics-iq-benchmark-verified` or `physics-iq-benchmark` dataset in your working directory. This dataset must include 30FPS videos and optionally your desired FPS. If your desired FPS does not exist in our dataset already, it will be automatically generated. You should have the following structure:

```plaintext
physics-IQ-benchmark/
├── full-videos/
│   └── ...
|
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

- the descriptions file which includes all file names and descriptions of the scenarios should be placed in your home directory as `descriptions.csv`.
- Place your generated videos under `.model_name` directory.

⚠️ **IMPORTANT:** Note that this script evaluates the **first 5 seconds** of your generated videos. Hence, make sure these are the 5 seconds generated right after the switch frame.

### 3. Generate benchmark scores and plots

```bash
uv run physiq/run_physics_iq.py --input_folders <generated_videos_dirs> --output_folder <output_dir> --descriptions_file <descriptions_file>
```
**Parameters:**
- `--input_folders`: The path to the directories containing input videos (in `.mp4` format), with one directory per model (`/model_name/video.mp4`).
- `--output_folder`: The path to the directory where output CSV files will be saved.
- `--descriptions_file`: The path to the `descriptions.csv` file.
- `--original_gt`: Whether to use the verified benchmark or the original benchmark.

---


## Citation
If you think this project is helpful, please feel free to leave a star ⭐️
<!-- TODO: finalize our publication here. -->
```latex
@article{raedsch2026physics,
  title={Physics-IQ Verified},
  author={}
  journal={arXiv preprint},
  year=2026
}

@article{motamed2026physics,
  title={Do generative video models understand physical principles?},
  author={Saman Motamed and Laura Culp and Kevin Swersky and Priyank Jaini and Robert Geirhos},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={948--958},
  year={2026}
}
```


## License and disclaimer

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
