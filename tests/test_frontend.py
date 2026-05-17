#!/usr/bin/env python
"""Smoke-test: load one session, run ModalityFrontend on CPU, print shapes.

Usage (from repo root):
    python tests/test_frontend.py [optional/path/to/session]
"""


from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import sys

from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import EngageNetConfig
from dataset import EngageNetDataset
from init_encoder import InitEncoder
from modality_frontend import ModalityFrontend
from read_data import ROLES, STREAM_FEATURES, log


def main() -> None:
    cnfg = EngageNetConfig()

    # 1. Load a single window from the first training session
    if len(sys.argv) > 1:
        split = sys.argv[1]
    else:
        split = "train"

    ds = EngageNetDataset(cnfg, split)
    window = next(ds.iter_windows())

    # Build a fake batch of size 1
    batch = {}
    for key, val in window.items():
        if isinstance(val, np.ndarray):
            batch[key] = jnp.array(val[np.newaxis])  # (1, C, L) or (1, L)
        # skip non-array entries like "session"

    log.info("Input shapes")
    for k, v in sorted(batch.items()):
        if k.endswith(".engagement"):
            continue
        log.info(f"  {k:45s}  {v.shape}")

    # 2. Instantiate ModalityFrontend and init params on CPU
    model = ModalityFrontend(cnfg=cnfg)

    # Only pass stream tensors (not engagement / session metadata)
    stream_inputs = {
        k: v for k, v in batch.items()
        if not k.endswith(".engagement") and isinstance(v, jnp.ndarray)
    }

    rng = jax.random.PRNGKey(cnfg.seed)
    variables = model.init(rng, stream_inputs, train=False)

    log.info("Model initialised")
    param_count = sum(x.size for x in jax.tree.leaves(variables["params"]))
    log.info(f"  Total trainable parameters: {param_count:,}")

    # 3. Forward pass
    hiddens, updates = model.apply(variables, stream_inputs, train=False, mutable=["batch_stats"])

    log.info("Output shapes (h_i)")
    for k, v in sorted(hiddens.items()):
        log.info(f"  {k:45s}  {tuple(v.shape)}")

    log.info("Smoke-test passed!")


if __name__ == "__main__":
    main()
