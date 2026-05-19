from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from flax.training import train_state

from beta_head import nll_loss
from config import EngageNetConfig
from data_loader import iter_batches
from model import EngageNet
from read_data import ROLES, log


class TrainState(train_state.TrainState):
    batch_stats: dict


def anneal_tau(tau_init: float, tau_min: float, decay: float, epoch: int) -> float:
    return max(tau_min, tau_init * (decay ** epoch))


# batch: dict{str: (B, C_i, L)}; ... -> (state, loss)
def train_step(state: TrainState, batch: dict[str, jax.Array], rng: jax.Array, tau: float):
    targets = []
    for role in ROLES:
        key = f"{role}.engagement"
        if key in batch:
            targets.append(batch[key].mean(axis=-1))
    target = jnp.stack(targets, axis=0).mean(axis=0)  # (B, L) -> (B,)

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
        log.info(f"Epoch {epoch+1:3d}/{cnfg.n_epochs}  tau={tau:.4f}  loss={avg_loss:.4f}")

        if (epoch + 1) % cnfg.checkpoint_every == 0:
            ckpt_path = cnfg.checkpoint_dir / f"EngageNet_{epoch+1}"
            checkpointer.save(ckpt_path, {"params": state.params, "batch_stats": state.batch_stats})
            log.info(f"Checkpoint saved: {ckpt_path}")

    log.info("Training complete")


if __name__ == "__main__":
    main()