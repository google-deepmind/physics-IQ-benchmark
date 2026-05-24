# Trimmed Generated Videos

Optional local folder for 5-second trimmed generated videos used for evaluation.

Expected structure:

```text
generated_videos_5s/<model_name>-<prompt_setting>-run_01/*.mp4
```

The evaluator reads these folders via `--input_folders`. Do not commit generated video files.
