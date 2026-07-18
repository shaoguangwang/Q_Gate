# Contributing to Q-Gate

Thanks for your interest in Q-Gate! This is the research code release for our
ACM MM 2026 paper, so contributions are welcome but scoped mainly to bug fixes,
reproducibility improvements, and documentation.

## Reporting issues

Please open an issue and include, where relevant:

- **Stage** you were running (Stage 0 data prep / Stage 1 score producers /
  Stage 2 fusion / Stage 3 QA).
- **Dataset** (`longvideobench` / `videomme`), **backend**
  (`gpt4` / `qwen3_32b` / `internvl3_8b` / `internvl_26b`), and **K**.
- The exact command you ran and the full error/traceback.
- Your environment (OS, Python version, `pip freeze` of the core deps).

A quick sanity check that the code itself runs is the offline demo:

```bash
python fuse_scores.py --input_json Data/demo_raw_scores.json \
    --output_dir ./demo_out --strategy equal --temperature 0.5
```

## Pull requests

1. Keep changes focused and describe what and why.
2. Do not introduce secrets, API keys, or absolute local paths; credentials are
   read from the environment (see `.env.example`).
3. Make sure the tree still byte-compiles (this is what CI checks):

   ```bash
   python -m compileall qgate fuse_scores.py run_qa.py \
       prepare_data.py score_producers tests
   ```

4. Preserve the reported paper numbers and the documented CLI interface unless
   the change is explicitly about correcting them.

## License

By contributing, you agree that your contributions are licensed under the
repository's [MIT License](LICENSE).
