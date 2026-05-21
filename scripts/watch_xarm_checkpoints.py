"""Visualize trained xarm-mug checkpoints against a held-out LeRobot episode.

For each checkpoint, the script loads the policy, walks the episode parquet,
queries the policy at sampled timesteps, integrates the predicted action chunk
forward from the ground-truth EE pose, and writes a figure containing:

    * 3D end-effector trajectory  (GT line + predicted chunks)
    * Gripper state over time     (GT line + predicted chunks)

Modes:
    --checkpoint-dir <path>      one-shot: visualize a single checkpoint
    --run-dir <path> [--watch]   sweep all step subdirs; with --watch, poll
                                 for newly-completed checkpoints

Action / state convention (xarm_mug_to_coffee_vla_10hz):
    state[0:3]  = EE xyz                (meters)
    state[3:6]  = EE orientation        (axis-angle, rad)
    state[6:8]  = gripper width per finger
    action[0:3] = delta EE xyz          (per-step)
    action[3:6] = delta orientation
    action[6]   = absolute gripper target

A "checkpoint" here is a step directory containing the standard openpi layout
(either `params/` for JAX or `model.safetensors` for PyTorch, plus `assets/`).
"""

from __future__ import annotations

import dataclasses
import io
import logging
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import PIL.Image
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Training config name (e.g. "pi05_xarm_mug_lora_10hz").
    config: str

    # One of these two must be set:
    # Path to a single checkpoint step dir (e.g. .../<exp>/30000).
    checkpoint_dir: pathlib.Path | None = None
    # Path to a run dir containing step subdirs (e.g. .../<exp>).
    run_dir: pathlib.Path | None = None

    # Parquet episode to replay.
    dataset_root: pathlib.Path = pathlib.Path(
        "/arm/u/ankushd/openpi-workspace/xarm_mug_to_coffee_vla_10hz"
    )
    episode: int = 0
    chunk: int = 0

    # Where to write figures. In run-dir mode, one PNG per checkpoint is written
    # here as <step>.png. In single-checkpoint mode, this is the output path
    # (file or directory).
    out: pathlib.Path = pathlib.Path("plots")

    # Predict every N frames; smaller = denser overlay, slower.
    stride: int = 20

    # In --watch mode, poll interval in seconds.
    poll_seconds: float = 30.0
    # Keep watching for new checkpoints.
    watch: bool = False

    # Optional prompt override. Defaults to the episode's task string.
    prompt: str | None = None


# ---------- dataset access ---------------------------------------------------


def _decode_image(cell) -> np.ndarray:
    """Parquet image cells are dicts with 'bytes' (encoded) + 'path'."""
    if isinstance(cell, dict):
        cell = cell["bytes"]
    img = PIL.Image.open(io.BytesIO(cell)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_episode(dataset_root: pathlib.Path, episode: int, chunk: int) -> dict:
    pq_path = (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode:06d}.parquet"
    )
    df = pq.read_table(pq_path).to_pandas()

    states = np.stack(df["state"].to_numpy()).astype(np.float32)         # [T, 8]
    actions_gt = np.stack(df["actions"].to_numpy()).astype(np.float32)   # [T, 7]
    images = np.stack([_decode_image(c) for c in df["image"].to_numpy()])  # [T,H,W,3]

    # Task prompt comes from meta/tasks.jsonl via task_index.
    task_index = int(df["task_index"].iloc[0])
    prompt = _read_task(dataset_root, task_index)

    return {
        "states": states,
        "actions": actions_gt,
        "images": images,
        "prompt": prompt,
    }


def _read_task(dataset_root: pathlib.Path, task_index: int) -> str:
    import json

    with open(dataset_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if int(row["task_index"]) == task_index:
                return row["task"]
    raise KeyError(f"task_index {task_index} not found")


# ---------- rollout / integration -------------------------------------------


def integrate_chunk(start_state: np.ndarray, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a predicted action chunk forward from start_state.

    chunk: [H, 7]   action[:3] = delta xyz, action[6] = absolute gripper.
    Returns (xyz [H+1, 3], gripper [H+1]) including the start sample.
    """
    pos = np.empty((chunk.shape[0] + 1, 3), dtype=np.float32)
    grip = np.empty(chunk.shape[0] + 1, dtype=np.float32)
    pos[0] = start_state[:3]
    grip[0] = start_state[6]  # one of the two finger widths; they match
    for k in range(chunk.shape[0]):
        pos[k + 1] = pos[k] + chunk[k, :3]
        grip[k + 1] = chunk[k, 6]
    return pos, grip


def predict_chunks(policy, episode: dict, stride: int, prompt: str) -> list[dict]:
    """Query the policy every `stride` frames; integrate each prediction."""
    states = episode["states"]
    images = episode["images"]
    T = states.shape[0]

    preds = []
    for t in range(0, T, stride):
        obs = {
            "observation/state": states[t],
            "observation/image": images[t],
            "prompt": prompt,
        }
        out = policy.infer(obs)
        chunk = np.asarray(out["actions"], dtype=np.float32)  # [H, 7]
        xyz, grip = integrate_chunk(states[t], chunk)
        preds.append({"t0": t, "xyz": xyz, "gripper": grip})
    return preds


# ---------- plotting --------------------------------------------------------


def plot_episode(episode: dict, preds: list[dict], title: str, out_path: pathlib.Path) -> None:
    states = episode["states"]
    T = states.shape[0]
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_g = fig.add_subplot(1, 2, 2)

    # Ground truth.
    gt_xyz = states[:, :3]
    ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="black", lw=2, label="GT EE")
    ax3d.scatter(*gt_xyz[0], color="green", s=40, label="start")
    ax3d.scatter(*gt_xyz[-1], color="red", s=40, label="end")

    ax_g.plot(t_axis, states[:, 6], color="black", lw=2, label="GT gripper")

    # Predicted chunks, colored by their start time.
    cmap = plt.get_cmap("viridis")
    H = preds[0]["xyz"].shape[0] - 1 if preds else 0
    for p in preds:
        c = cmap(p["t0"] / max(T - 1, 1))
        ax3d.plot(p["xyz"][:, 0], p["xyz"][:, 1], p["xyz"][:, 2], color=c, lw=1.2, alpha=0.85)
        gt_t = np.arange(p["t0"], p["t0"] + p["gripper"].shape[0])
        ax_g.plot(gt_t, p["gripper"], color=c, lw=1.2, alpha=0.85)

    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("End-effector trajectory")
    ax3d.legend(loc="upper left", fontsize=8)

    ax_g.set_xlabel("frame")
    ax_g.set_ylabel("gripper width")
    ax_g.set_title("Gripper over time")
    ax_g.legend(loc="upper left", fontsize=8)
    ax_g.grid(True, alpha=0.3)

    # Colorbar for chunk start time.
    if preds:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T - 1))
        cb = fig.colorbar(sm, ax=ax_g, fraction=0.04, pad=0.04)
        cb.set_label("chunk start frame")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------- driver ----------------------------------------------------------


def _checkpoint_complete(step_dir: pathlib.Path) -> bool:
    """A checkpoint is "complete" once params/ (JAX) or model.safetensors exists."""
    return (step_dir / "params").exists() or (step_dir / "model.safetensors").exists()


def _list_checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if _checkpoint_complete(child):
            out.append(child)
    return out


def run_one(train_config, ckpt_dir: pathlib.Path, episode: dict, out_path: pathlib.Path,
            stride: int, prompt: str) -> None:
    logging.info("loading checkpoint %s", ckpt_dir)
    policy = _policy_config.create_trained_policy(train_config, str(ckpt_dir))
    preds = predict_chunks(policy, episode, stride=stride, prompt=prompt)
    title = f"{ckpt_dir.parent.name}/{ckpt_dir.name}  ep={out_path.stem}"
    plot_episode(episode, preds, title=title, out_path=out_path)
    logging.info("wrote %s (%d predicted chunks)", out_path, len(preds))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if (args.checkpoint_dir is None) == (args.run_dir is None):
        raise SystemExit("pass exactly one of --checkpoint-dir or --run-dir")

    train_config = _config.get_config(args.config)
    episode = load_episode(args.dataset_root, args.episode, args.chunk)
    prompt = args.prompt or episode["prompt"]
    logging.info("loaded episode %d (%d frames), prompt=%r",
                 args.episode, episode["states"].shape[0], prompt)

    if args.checkpoint_dir is not None:
        out = args.out
        if out.is_dir() or out.suffix == "":
            out = out / f"ep{args.episode:03d}_{args.checkpoint_dir.name}.png"
        run_one(train_config, args.checkpoint_dir, episode, out, args.stride, prompt)
        return

    run_dir = args.run_dir
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()

    while True:
        for ck in _list_checkpoints(run_dir):
            if ck.name in done:
                continue
            out_path = out_dir / f"ep{args.episode:03d}_{ck.name}.png"
            try:
                run_one(train_config, ck, episode, out_path, args.stride, prompt)
            except Exception as e:
                logging.exception("failed on %s: %s", ck, e)
            done.add(ck.name)
        if not args.watch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main(tyro.cli(Args))
