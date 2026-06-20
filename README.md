# Catan LLM Benchmark

![Catan LLM Benchmark preview](assets/preview.svg)

Text-only Catan benchmark harness for seeing how language models choose legal actions inside a real board-game simulator, with full JSONL traces instead of hand-wavy eval summaries.

## At a glance

- Uses [Catanatron](https://github.com/bcollazo/catanatron) as the simulator.
- Feeds each model a compact JSON state plus numbered legal actions.
- Expects a strict JSON answer like `{"choice": 3, "reason": "..."}`.
- Logs every prompt, model response, fallback, and game result as JSONL.
- Works without secrets in `fake` mode, then upgrades to Hugging Face router mode when `HF_TOKEN` is available.

## Why this exists

Most LLM benchmark harnesses measure static tasks. This one measures sequential game decisions inside a real simulator, which makes it useful for testing:

- action selection under structured state pressure
- provider reliability and fallback behavior
- repeated strategy differences over full games
- batch collection against a fixed family of sampled states

## Verified on 2026-06-20

- `python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`
- `.venv/bin/python bench.py --mode fake --games 2 --max-actions 80 --vps 6 --seed 42`
- `.venv/bin/python batch_collect.py --help`

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## No-secret smoke test

This path does not call any external model:

```sh
.venv/bin/python bench.py --mode fake --games 2 --max-actions 80 --vps 6 --seed 42
```

It writes JSONL logs and a `summary.json` file under `runs/<timestamp>/`.

## Hugging Face router mode

Set `HF_TOKEN` or log in with the Hugging Face CLI so the local token cache exists.

```sh
HF_TOKEN=... .venv/bin/python bench.py \
  --mode hf \
  --games 1 \
  --max-actions 120 \
  --vps 6 \
  --models zai-org/GLM-5.2:zai-org,zai-org/GLM-5.2:novita
```

Defaults:

- base URL: `https://router.huggingface.co/v1`
- model list:
  - `zai-org/GLM-5.2:zai-org`
  - `zai-org/GLM-5.2:novita`
  - `zai-org/GLM-5.2:fireworks-ai`
  - `zai-org/GLM-5.2:deepinfra`
  - `zai-org/GLM-5.2:featherless-ai`

If HF auth is missing, `--mode auto` falls back to `fake`.

## Batch collection

`batch_collect.py` samples many intermediate game states and queries one model across several strategy prompts.

```sh
HF_TOKEN=... .venv/bin/python batch_collect.py \
  --model zai-org/GLM-5.2:fireworks-ai \
  --states-per-batch 25 \
  --workers 8 \
  --max-batches 1
```

That writes:

- `runs/batch-collect-*/states.jsonl`
- `runs/batch-collect-*/decisions.jsonl`
- `runs/batch-collect-*/batch_summaries.jsonl`
- `runs/batch-collect-*/status.json`

Use `--cutoff 2026-06-20T18:00:00-04:00` if you want the collector to stop at a fixed time. Leave it unset for an open-ended run.

## Output shape

Main benchmark runs emit:

- `events.jsonl`
- `summary.json`

Each decision row includes the prompt, legal actions, selected action, fallback status, and the parsed model response when available.

## Privacy and safety

- Nothing in this repo uploads local Codex logs or browser data.
- `runs/` is ignored because it can contain provider responses and prompt traces.
- The public repo documents environment-variable names but does not ship secrets, cookies, session data, or personal machine paths.

See [SECURITY.md](SECURITY.md) for the publish checklist used for this extraction.
