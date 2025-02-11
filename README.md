<p align="center">
  <img src="assets/duckie.png" width="20%" alt="logo">
</p>

[Step A: Generating Videos](#step-a-generating-videos-for-physics-iq-test-cases-based-on-video-model) | [Step B: Evaluating Generated Videos](#step-b-evaluating-generated-videos-on-physics-iq-to-generate-benchmark-scores) | [Leaderboard](#leaderboard) | [Citation](#citation) | [License](#license-and-disclaimer)

# Physics-IQ: Benchmarking physical understanding in generative video models

Physics-IQ is a high-quality, realistic, and comprehensive benchmark dataset for evaluating physical understanding in generative video models.

Project website: [physics-iq.github.io](https://physics-iq.github.io/)

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

## Step A: Generating Videos for Physics-IQ Test Cases Based on Video Model

### 1. Download Benchmark Dataset

Visit the [Google Cloud Storage link](https://console.cloud.google.com/storage/browser/physics-iq-benchmark) to download the dataset, or run the following script:

```bash
pip install gsutil
python3 ./code/download_physics_iq_data.py
```

- If your desired FPS already exists in the dataset, it will be downloaded.
- If it does not exist, the script will download 30 FPS files and generate your desired FPS videos by downsampling the 30 FPS version.

---

### 2. Running Video Model on Test Cases from Benchmark

This section explains how to generate videos using the provided benchmark and save them in the required format. Follow the instructions below based on your model type:

#### 2.1 Image-to-Video (i2v) Models

1. **Input Requirements**:
   - **Initial Frame**: Use frames from `physics-iq-benchmark/switch-frames`.
   - **Text Input (Optional)**: If required, use descriptions from `descriptions.csv`.

2. **Steps to Run**:
   - Generate videos using the initial frame (and text condition, if applicable).
   - Save generated videos in the following structure, using any filename as long as the unique ID prefix from the test videos is kept (`0001_`, ..., `0198_`):
     ```
     .model_name/{ID}_{anything-you-like}.mp4
     ```
   - Refer to the `generated_video_name` column in `descriptions.csv` for file naming conventions.

---

#### 2.2 Multiframe-to-Video Models

1. **Input Requirements**:
   - **Conditioning Frames**:
     - Available in `physics-iq-benchmark/split-videos/conditioning-videos`.
     - Ensure the correct frame rate: `30FPS`, `24FPS`, `16FPS`, or `8FPS`.
   - **Text Input (Optional)**: Use `descriptions.csv`.

2. **Steps to Run**:
   - Use conditioning frames to generate videos.
   - Save generated videos in the structure:
     ```
     model_name/{ID}_{perspective}_{scenario_name}.mp4
     example: model_name/{0001}_{perspective-left}_{trimmed-ball-and-block-fall}.mp4
     ```
   - Refer to the `generated_video_name` column in `descriptions.csv` for file naming conventions.

---

## Step B: Evaluating Generated Videos on Physics-IQ to Generate Benchmark Scores

### 1. Installation

Ensure you have Python 3 installed. Then, run the following command to install the necessary packages:

```bash
pip install -r requirements.txt
```

### 2. Dataset Placement

- Ensure you have downloaded and placed the `physics-iq-benchmark` dataset in your working directory. This dataset must include 30FPS videos and optionally your desired FPS. If your desired FPS does not exist in our dataset already, it will be automatically generated. You should have the following structure:

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
python3 code/run_physics_iq.py --input_folders <generated_videos_dirs> --output_folder <output_dir> --descriptions_file <descriptions_file>
```
**Parameters:**
- `--input_folders: The path to the directories containing input videos (in `.mp4` format), with one directory per model (/model_name/video.mp4)
- `--output_folder`: The path to the directory where output csv files will be saved
- `--descriptions_file`: The path to the descriptions.csv file

---

## Leaderboard

<p align="center">
  <img src="assets/physics_IQ_score.png" width="75%" alt="Comparison of metrics for different models">
</p>

---

## Citation

```latex
@article{motamed2025physics,
  title={Do generative video models learn physical principles from watching videos?},
  author={Saman Motamed and Laura Culp and Kevin Swersky and Priyank Jaini and Robert Geirhos},
  journal={arXiv preprint arXiv:2501.09038},
  year={2025}
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

