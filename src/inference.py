from __future__ import annotations

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beta_head import predictive_mean
from config import EngageNetConfig
from data_loader import iter_batches
from model import EngageNet
from read_data import ROLES, log
from train import TrainState, create_train_state
from tta import sample_filter, tta_step


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    ckpts = sorted(checkpoint_dir.glob("EngageNet_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    return ckpts[-1]


def main():
    cnfg = EngageNetConfig.from_cli()

    rng = jax.random.PRNGKey(cnfg.seed)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(cnfg, rng_init)

    ckpt_path = find_latest_checkpoint(cnfg.checkpoint_dir)
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})
    state = state.replace(params=restored["params"], batch_stats=restored["batch_stats"])

    log.info("Running TTA on test split")
    predictions = []

    for batch in iter_batches(cnfg, split="test"):
        stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement") and k != "session"}

        rng, rng_step = jax.random.split(rng)

        variables = {"params": state.params, "batch_stats": state.batch_stats}
        (alpha, beta, unimodal), _ = state.apply_fn(variables, stream_inputs, tau=cnfg.tau_min, rng=None, train=False, mutable=["batch_stats"])

        mask = sample_filter(alpha, beta, unimodal)

        if mask.any():
            rng, rng_tta = jax.random.split(rng)
            state, _loss, alpha, beta, unimodal = tta_step(state, stream_inputs, tau=cnfg.tau_min, rng=rng_tta)

        pred = predictive_mean(alpha, beta)
        predictions.append(pred)

    all_preds = jnp.concatenate(predictions, axis=0)
    log.info(f"Inference complete: {all_preds.shape[0]} predictions, mean={float(all_preds.mean()):.4f}")


if __name__ == "__main__":
    main()