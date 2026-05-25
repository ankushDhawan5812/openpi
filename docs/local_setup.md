# Local Setup Notes

Quirks, gotchas, and workarounds we hit getting pi0.5 LoRA finetuning working
on a single-GPU desktop (Ubuntu, RTX 5090). Use this as a quick reference; the
canonical README has the official flow.

---

## Installation

### 1. Install `uv` from the official script

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Do **not** use `snap install astral-uv` — it's an unofficial mirror and ships
older versions. The official installer drops `uv` into `~/.local/bin/`.

### 2. First `uv run` bootstraps everything

`uv run scripts/train.py ...` (or any script under `scripts/`) will:
- create `.venv/` if missing,
- resolve and install everything from `pyproject.toml` / `uv.lock`,
- build the local `openpi` + `openpi-client` packages,
- pull `lerobot` from its pinned git commit.

Takes a few minutes on the first invocation, then cached.

### 3. Wandb version pin

The default lockfile pins `wandb==0.19.11`, whose `apikey.write_key` hard-checks
that the API key is exactly 40 characters. Newer wandb accounts (org / SSO)
issue 86-character keys, which fail this check.

```bash
uv add "wandb>=0.27"
```

`uv add` updates `pyproject.toml` *and* `uv.lock`, which is what survives
`uv run` (a plain `uv pip install -U wandb` gets reverted on the next `uv run`
because it re-syncs from the lockfile).

---

## Dataset setup

### LeRobot dataset discovery

The training pipeline calls `LeRobotDataset(repo_id="<owner>/<name>")`, which
looks under `$HF_LEROBOT_HOME` (defaults to `~/.cache/huggingface/lerobot`).
Point it at a local dataset directory via a symlink:

```bash
mkdir -p ~/.cache/huggingface/lerobot/<owner>
ln -sfn /path/to/your/local/dataset_dir \
        ~/.cache/huggingface/lerobot/<owner>/<dataset_name>
```

The `<owner>/<dataset_name>` after the symlink must match the `repo_id` in
your TrainConfig.

### Two LeRobot column-name conventions

Datasets exported via different tools can end up with different column names
for the same content:

| Convention | Image key | State key | Action key |
|---|---|---|---|
| Standard LeRobot v2.1 | `observation.images.<cam>` | `observation.state` | `action` |
| Pre-processed for openpi | `image` | `state` | `actions` |

The existing `LeRobotXarmMugDataConfig` / `XarmMugInputs` pipeline expects the
**pre-processed** layout. If your dataset uses the v2.1 dotted-key layout, you
either need to (a) write a custom `DataConfig` with a `RepackTransform` mapping
your dotted keys to `observation/image`, `observation/state`, `actions`, or
(b) run a conversion step to produce the flat-key layout the existing pipeline
expects. Option (b) is what `xarm_single_cup_single_machine_160_10hz` /
`panda_4by4_mug_machine_10hz` use.

Also note: the default `action_sequence_keys=("actions",)` on `DataConfig`
must be overridden to `("action",)` if your column is singular.

---

## Single-GPU runs (RTX 5090)

### Batch size

The default `batch_size=32` assumes the 8-GPU slurm setup (4 per GPU on 24 GB
cards). On a single 32 GB 5090 you need to override:

```bash
uv run scripts/train.py <config-name> \
    --exp-name=<name> --fsdp-devices=1 --batch-size=4
```

Start at 4, bump to 8 if `nvidia-smi` shows headroom.

### Warnings you can ignore

- `ptxas too old. Falling back to the driver` — your system CUDA toolkit
  predates Blackwell (CC 12.0). The CUDA driver (570.x) supports the GPU and
  JAX routes PTX compilation through it. Affects first-run JIT compile time
  only.
- Long `Trying algorithm engN{...} ... is taking a while` lines — cuDNN
  convolution autotuning at first JIT. One-time, gets cached.
- `Can't reduce memory use below X GiB by rematerialization` — XLA tried to
  fit into a tighter budget but settled higher. Informational unless you
  actually OOM (look for `RESOURCE_EXHAUSTED`).

### LR schedule decay vs. num_train_steps gotcha

`CosineDecaySchedule.decay_steps` is independent from
`TrainConfig.num_train_steps`. Default is `decay_steps=30_000`. If you train
0 → 30k, the LR has decayed to ~0 by the end. Resuming with
`--num-train-steps=60000 --resume` runs steps 30k → 60k at near-zero LR — no
real learning.

Two clean options if you might want to train beyond 30k:

```bash
# Option A — decide max steps up front
uv run scripts/train.py <config> --num-train-steps=60000 \
    --lr-schedule.decay-steps=60000

# Option B — when actually resuming
uv run scripts/train.py <config> --num-train-steps=60000 \
    --lr-schedule.decay-steps=60000 --resume
```

Option B works but creates an LR discontinuity at step 30k.

---

## Wandb prompt during training

If you've never logged in, training will pause at:
```
wandb: Paste an API key from your profile and hit enter
```

Go to https://wandb.ai/authorize, paste the key. With `wandb>=0.27` installed
(see above), both 40-char and 86-char keys are accepted. The key gets saved to
`~/.netrc`, so subsequent runs don't re-prompt.

To skip wandb entirely (no online dashboard, but console logs and checkpoints
still work): `--no-wandb-enabled`.

---

## Dataset-size ablations

A `--num-episodes=N` flag is available on both `train.py` and
`compute_norm_stats.py`. When set, the data loader keeps `N` evenly-spaced
episodes via `np.linspace(0, total-1, N).round()` — important when same-type
demos were collected consecutively, since linspace preserves coverage across
the collection order.

```bash
uv run scripts/compute_norm_stats.py --config-name <config> --num-episodes 100
uv run scripts/train.py <config> --exp-name=<name> \
    --fsdp-devices=1 --batch-size=4 --num-episodes=100
```

**Caveat:** norm stats live at `assets/<config_name>/<repo_id>/`, which does
*not* include `num_episodes` in the path. Running `--num-episodes=80` after
`--num-episodes=100` overwrites the 100-ep stats. Either recompute before
each run, or pass `--data.assets.asset-id=<repo_id>_n<N>` to both commands to
namespace the norm-stats dir per subset.

---

## Workaround: LeRobot `episodes=` argument bug

`LeRobotDataset(..., episodes=[...])` has an upstream IndexError: `__getitem__`
reads the *original* `episode_index` from the parquet (e.g., 100) and indexes
into `self.episode_data_index`, which only has `len(episodes)` entries after
filtering. With a 100-ep subset of a 160-ep dataset, asking for any frame from
episode 100+ raises `IndexError: index 100 is out of bounds for dimension 0
with size 100`.

We work around it in
[`src/openpi/training/data_loader.py`](../src/openpi/training/data_loader.py)
by loading the full dataset and wrapping with `torch.utils.data.Subset` on the
explicit frame indices for the selected episodes. Don't pass `episodes=` to
`LeRobotDataset` directly until that's fixed upstream.

---

## Recent additions to the codebase

These were added during this setup; reference them when wiring up new datasets.

### Configs
- `pi05_single_cup_160ep_lora_10hz` — 160-ep xarm single-cup, uses pre-processed
  flat-key dataset (`ankushd/xarm_single_cup_single_machine_160_10hz`).
- `pi05_panda_4by4_lora_10hz` — 160-ep panda 4x4 grid, same schema.
- `pi05_single_cup_100ep_lora_10hz` — 100-ep ablation with hardcoded indices
  (now superseded by `--num-episodes=100` on the 160-ep config; left in place
  for reference).

### Data subsetting plumbing
- `DataConfig.episodes` — explicit list of episode indices.
- `DataConfig.num_episodes` — count to keep, resolved via linspace.
- `TrainConfig.num_episodes` — top-level CLI flag (`--num-episodes=N`).
- `compute_norm_stats.py --num-episodes N` — mirror flag for norm stats.

### Dependency
- `wandb >= 0.27` (was 0.19.11).

---

## Quick troubleshooting checklist

| Symptom | Likely cause / fix |
|---|---|
| `uv: command not found` | `source $HOME/.local/bin/env` after installer |
| `API key must be 40 characters long` | wandb too old — `uv add "wandb>=0.27"` |
| `KeyError: Column 'X' not in the dataset` | Column name mismatch — fix the `RepackTransform` keys or use a config that matches your dataset's columns |
| `KeyError: 'actions'` (norm stats step) | `action_sequence_keys=("actions",)` doesn't match — override to `("action",)` for v2.1-layout datasets |
| `IndexError: index N is out of bounds for dimension 0` during `_get_query_indices` | Old `data_loader.py` passed `episodes=` to LeRobot directly. Update to the frame-level subset workaround. |
| OOM during training | Drop `--batch-size`. On 32 GB GPU, bs=4 is the safe starting point. |
| LR stays at zero after step 30k on resume | `--lr-schedule.decay-steps` needs to match your extended `--num-train-steps`. |
| Norm stats look weird after switching `--num-episodes` | Different subsets overwrote each other. Recompute, or namespace via `--data.assets.asset-id`. |
