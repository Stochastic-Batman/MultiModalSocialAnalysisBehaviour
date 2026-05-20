from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from flax.training import train_state

from beta_head import nll_loss, predictive_mean
from config import EngageNetConfig
from data_loader import iter_batches
from metrics import ccc
from model import EngageNet
from read_data import ROLES, log


class TrainState(train_state.TrainState):
    batch_stats: dict


def anneal_tau(tau_init: float, tau_min: float, decay: float, epoch: int) -> float:
    return max(tau_min, tau_init * (decay ** epoch))


# batch: dict{str: (B, C_i, L)}; ... -> (state, loss)
def train_step(state: TrainState, batch: dict[str, jax.Array], rng: jax.Array, tau: float):
    # Per-frame targets: average expert + novice engagement -> (B, L)
    targets = []
    for role in ROLES:
        key = f"{role}.engagement"
        if key in batch:
            targets.append(batch[key])
    target = jnp.stack(targets, axis=0).mean(axis=0)  # (B, L)

    stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement") and k != "session"}

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (alpha, beta, _unimodal), updates = state.apply_fn(variables, stream_inputs, tau=tau, rng=rng, train=True, mutable=["batch_stats"])
        loss = nll_loss(alpha, beta, target)
        return loss, updates["batch_stats"]

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_batch_stats), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)

    return state, loss


def val_ccc(state: TrainState, cnfg: EngageNetConfig, tau: float) -> float:
    all_preds = []
    all_targets = []

    for batch in iter_batches(cnfg, split="val"):
        stream_inputs = {k: v for k, v in batch.items() if not k.endswith(".engagement") and k != "session"}

        variables = {"params": state.params, "batch_stats": state.batch_stats}
        (alpha, beta, _), _ = state.apply_fn(variables, stream_inputs, tau=tau, rng=None, train=False, mutable=["batch_stats"])

        preds = predictive_mean(alpha, beta)  # (B, L')
        all_preds.append(np.array(preds.reshape(-1)))

        targets = []
        for role in ROLES:
            key = f"{role}.engagement"
            if key in batch:
                targets.append(np.array(batch[key]))
        target = np.stack(targets, axis=0).mean(axis=0)  # (B, L)
        all_targets.append(target.reshape(-1))

    return ccc(np.concatenate(all_preds), np.concatenate(all_targets))


def create_train_state(cnfg: EngageNetConfig, rng: jax.Array) -> TrainState:
    model = EngageNet(cnfg=cnfg)

    dummy = {}
    for feat in cnfg.modality_names:
        c_in = cnfg.input_dim(feat)
        for role in ROLES:
            dummy[f"{role}.{feat}"] = jnp.zeros((cnfg.batch_size, c_in, cnfg.window_len))

    rng_init, rng_gumbel = jax.random.split(rng)
    variables = model.init(rng_init, dummy, tau=1.0, rng=rng_gumbel, train=False)

    tx = optax.adamw(learning_rate=cnfg.lr, weight_decay=cnfg.weight_decay)

    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx, batch_stats=variables.get("batch_stats", {}))


def main() -> None:
    cnfg = EngageNetConfig.from_cli()

    rng = jax.random.PRNGKey(cnfg.seed)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(cnfg, rng_init)
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    log.info(f"Total params: {param_count:,}")

    cnfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    best_val_ccc = -1.0
    epochs_without_improvement = 0

    for epoch in range(cnfg.n_epochs):
        tau = anneal_tau(cnfg.tau_init, cnfg.tau_min, cnfg.tau_decay, epoch)
        rng, rng_epoch = jax.random.split(rng)

        epoch_loss = 0.0
        n_batches = 0

        for batch in iter_batches(cnfg, split="train", shuffle=True, seed=cnfg.seed + epoch):
            rng, rng_step = jax.random.split(rng)
            state, loss = train_step(state, batch, rng_step, tau)
            epoch_loss += float(loss)
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        if (epoch + 1) % cnfg.eval_every == 0:
            v_ccc = val_ccc(state, cnfg, tau)
            log.info(f"Epoch {epoch+1:3d}/{cnfg.n_epochs}  tau={tau:.4f}  loss={avg_loss:.4f}  val_ccc={v_ccc:.4f}")

            if v_ccc > best_val_ccc:
                best_val_ccc = v_ccc
                epochs_without_improvement = 0
                checkpointer.save(cnfg.checkpoint_dir / "best", {"params": state.params, "batch_stats": state.batch_stats})
                log.info(f"  New best val CCC: {best_val_ccc:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cnfg.patience:
                log.info(f"Early stopping at epoch {epoch+1} (patience={cnfg.patience})")
                break
        else:
            log.info(f"Epoch {epoch+1:3d}/{cnfg.n_epochs}  tau={tau:.4f}  loss={avg_loss:.4f}")

        if (epoch + 1) % cnfg.checkpoint_every == 0:
            ckpt_path = cnfg.checkpoint_dir / f"EngageNet_{epoch+1}"
            checkpointer.save(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})
            log.info(f"Checkpoint saved: {ckpt_path}")

    log.info(f"Training complete. Best val CCC: {best_val_ccc:.4f}")


if __name__ == "__main__":
    main()