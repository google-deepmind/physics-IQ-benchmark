# Generated Videos

Optional local folder for raw generated videos before trimming.

Expected structure:

```text
generated_videos/<model_name>-<prompt_setting>-run_01/*.mp4
```

Use `bpp` for model-specific benchmark prompts and `op` for original benchmark prompts. You may also keep generated videos outside the repo and pass those folders directly to `--input_folders`.
