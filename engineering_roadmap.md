# Engineering Roadmap

This markdown file describes the coding decisions we made after having an initial architecture, currently described in `documentation/EngageNet Draft.tex`. The architecture defines a multimodal BiMamba network with per-modality initial encoders, intra-modal and inter-modal BiMamba modules, Beta regression heads, test-time adaptation, and a differentiable Gumbel-Sinkhorn modality ordering mechanism.

## Implemented

### `src/config.py`

A single `@dataclass` that centralises every hyperparameter and path so nothing is hardcoded elsewhere. This is the only file you need to edit to change the corpus, window size, or per-modality encoder widths.

**`_DEFAULT_MODALITY_SPECS`** - a dictionary mapping each feature name to a 4-tuple `(C_i, C_i', kernel_size, stride)`. The `C_i` values (input channels) come directly from the dataset documentation and the `.stream` XML headers: for example, eGeMaps v2 outputs 88-dimensional acoustic descriptors, W2v-BERT 2.0 produces 1024-dimensional speech embeddings, OpenPose encodes body + 2x hand keypoints as 139 values (x, y, confidence per joint), and OpenFace2 packs 714 dimensions covering Action Units (facial muscle activation intensities), facial landmarks (2D/3D coordinates of 68 face points), gaze direction vectors, and head pose (rotation + translation). The `C_i'` values (encoder output width) are set to 64 for the two lowest-dimensional modalities (eGeMaps at 88 and OpenPose at 139) and 128 for everything else - this keeps the shallow encoder from being a bottleneck on small inputs while avoiding unnecessary width on already-compact features. Kernel size 5 and stride 1 are conservative defaults: kernel 5 gives a receptive field of 5 frames (200 ms at 25 Hz), which is enough to smooth frame-level noise without losing temporal resolution, and stride 1 preserves the original sequence length so output length equals input length.

**`target_sr: float = 25.0`** - the target sample rate in Hertz. All streams are resampled to this rate so that every modality shares the same temporal axis. 25 Hz was chosen because the engagement annotations are natively at 25 Hz (one label per video frame at 25 fps), so resampling to 25 Hz makes the labels and features align without any interpolation of the supervision signal.

**`window_len: int = 250`** - 250 frames at 25 Hz = 10 seconds. This is a common segment length in affective computing: long enough to capture meaningful engagement dynamics (a shift from engaged to disengaged typically takes several seconds) but short enough to fit comfortably in memory as a single batch element.

**`window_stride: int = 125`** - 50% overlap between consecutive windows. This doubles the effective number of training samples without loading more data from disk.

**`shared_dim: int = 128`** - the uniform projection dimension `C'` from the Order-Sensitivity Problem section of the draft. After each `InitEncoder_i` produces a hidden representation of shape `(L' by C_i')`, a learned linear projection maps it to shape `(L' by C')`. This is required by the Gumbel-Sinkhorn permutation mechanism in the inter-modal fusion module: you can only permute and soft-mix modality representations if they all live in the same vector space. 128 was chosen as a balance - large enough to not lose information from the 128-wide encoder outputs, small enough that the full concatenation across 20 modality-role pairs (20 times 128 = 2560 channels) does not blow up the inter-modal BiMamba state size.

**`active_modalities: Optional[list[str]]`** - controls which pre-extracted feature streams are actually loaded and processed. Defaults to `None` (all 10 streams). Set to `CORE_MODALITIES` to use only the 4 streams most relevant to engagement: eGeMaps v2 (prosodic features), W2v-BERT 2.0 (learned speech representations), OpenFace2 (facial AUs, landmarks, gaze, head pose), and OpenPose (body posture and hand gestures). The remaining 6 streams (XLM-RoBERTa, CLIP, DINOv2, ImageBind, Swin, VideoMAE) are either redundant with the core 4 or encode generic visual semantics that don't vary meaningfully across engagement levels. This filter propagates everywhere: the dataset skips loading unused `.stream` files from disk, and the `ModalityFrontend` only creates encoders for active streams. Note that this only affects the pre-extracted feature streams - annotation data (engagement labels, transcripts, age, gender, language) is always loaded regardless, since it comes from separate CSV files outside the stream pipeline.


### `src/dataset.py`

The lazy, memory-efficient data pipeline. The central class `EngageNetDataset` wraps a corpus split directory (e.g. `data/NoXi+J/train/`) and exposes an `iter_windows()` generator. Internally it iterates over session directories one at a time, calls `read_data.load_session()` to pull all streams for that session from disk, resamples everything to the target sample rate, clips to the shortest common temporal length, slices into fixed-length windows, and yields each window as a dictionary of NumPy arrays. Once all windows from a session are yielded, the session data falls out of scope and gets garbage-collected before the next session is loaded. This way, peak memory usage is proportional to one session (~1.5-2 GiB) rather than the full corpus (88+ GiB).

**Resampling** uses `scipy.interpolate.interp1d` with linear interpolation. This is simple and fast; for 25 Hz signals being resampled to 25 Hz it is a no-op.

**`_pad_or_trim`** ensures every window has exactly `window_len` frames. If the session's tail is shorter (i.e. the last window runs past the end of the data), we zero-pad. The `*arr.shape[1:]` unpacking in the padding constructor preserves all non-temporal dimensions (i.e. channels) so the zero-padding shape matches regardless of how many channels the modality has.

**Shuffling** uses a `jax.random.PRNGKey` to generate a permutation of session indices via `jax.random.permutation`. This keeps all randomness in the JAX PRNG system - explicit, splittable, and reproducible - rather than relying on NumPy's global random state.

**Output format** - each window's streams are transposed to channels-first shape `(C_i, L)` because the `InitEncoder` expects this layout (matching the paper's notation for input `x_i`).

### `src/data_loader.py`

A thin wrapper over `EngageNetDataset.iter_windows()` that accumulates windows into batches of size `cnfg.batch_size`, converts them to JAX arrays via `jnp.array(np.stack(...))`, and yields batched dictionaries. The `_collate` function stacks all numeric arrays along a new leading batch axis while keeping the `"session"` key as a plain Python list of strings (metadata, not a tensor). The last batch of an epoch may be smaller than `batch_size` - we yield it anyway rather than dropping data.

### `src/init_encoder.py`

The `InitEncoder` Flax module implements the shallow per-modality feature extractor described in Section 1.2.1 of the draft. It maps an input of shape `(C_i, L)` to a hidden representation of shape `(L', C_i')`.

Internally it is a single `nn.Conv` (1-D, kernel and stride from config) -> `nn.BatchNorm` -> SiLU activation. The convolution uses `padding="SAME"` so that with the default stride of 1, the output length equals the input length. The dimension transposition from channels-first input `(B, C_i, L)` to time-first output `(B, L', C_i')` is handled inside the module (Flax Conv expects channels-last).

BatchNorm is included because the draft's Section 2.4 (Surgical Fine-Tuning) specifically lists "BatchNorm layers" and "First convolution in InitEncoder" as the surgical layers updated during test-time adaptation. Having BatchNorm here makes TTA straightforward later: just unfreeze `batch_norm1` and `conv1` in the InitEncoder while keeping everything else frozen.

SiLU is the activation used throughout the Mamba architecture and is specified in the draft's BiMamba equations.

### `src/modality_frontend.py`

The top-level `ModalityFrontend` Flax module that owns all `InitEncoder` instances (one per feature type) and `nn.Dense` projection layers (one per feature type). For each feature type, it runs the same encoder on both the expert and novice inputs - weight sharing across roles. This is the right call because expert and novice streams for the same feature (e.g. both `expert.clip` and `novice.clip`) are extracted by the same pretrained model from the same type of input (video frames), so the shallow encoder should learn the same low-level patterns for both.

After each encoder produces a hidden representation of shape `(B, L', C_i')`, a per-feature `nn.Dense` projects it to the shared dimension, giving shape `(B, L', C')`. This is the uniform channel projection required so that the Gumbel-Sinkhorn permutation can treat all modality representations as interchangeable objects in the same vector space.

The output is a dictionary keyed by `"{role}.{feat}"` with all hidden representations ready for the BiMamba modules.

### `src/ssm.py`

A pure JAX module (no Flax, no `nn.Module`) implementing the selective state space scan that sits at the core of every Mamba block. `bimamba.py` owns all weight matrices and computes the input-dependent parameters; `ssm.py` is the low-level primitive it calls after those projections.

**`discretize(delta, A, B)`** converts continuous-time parameters to discrete-time ones. It computes the discretized transition matrix `A_bar` by exponentiating the element-wise product of `delta` and `A`, and the discretized input matrix `B_bar` as the element-wise product of `delta` and `B`. Both are broadcast to shape `(B, L, D, N)` via reshaping.

**`_scan_fn(left, right)`** is the associative combiner passed to `lax.associative_scan`. Each time step defines an affine map: "given the previous hidden state, produce the next one by scaling with `A_bar_t` and adding the input contribution `b_t = B_bar_t * u_t`." Two consecutive affine maps combine by multiplying their scaling factors and folding the earlier input contribution through the later scaling. The scan accumulates these combinations across all time steps so all hidden states are computed in `O(log L)` parallel steps rather than `O(L)` sequential steps. Since the initial state is zero, the accumulated input contribution at each position is directly the hidden state at that time step, which is why only the second element of the scan output is unpacked.

**`ssm(u, delta, A, B, C)`** is the main public function. It calls `discretize`, precomputes the input contribution at each step, transposes for the scan, calls `lax.associative_scan`, transposes back, and produces the output sequence by taking the element-wise product of the output projection `C` and the hidden states and summing over the state dimension.

### `src/bimamba.py`

**`BiMambaBlock`** implements the three steps from Section 1.2.1 of the draft. It takes a hidden representation of shape `(B, L', D)` and returns one of the same shape.

Step 1 is gating: a learned SiLU projection produces a gate tensor of the same shape that is later applied via element-wise multiplication to suppress noise and highlight relevant features.

Step 2 is a shared linear projection applied to the input before splitting into two branches.

Step 3 is the forward SSM branch: a depthwise Conv1D (each channel processed independently) with SiLU captures local context, then the three input-dependent selection parameters are computed - `delta` (timescale, via `softplus(s_delta + Linear(x))`), `B` (what to write to the state), and `C` (what to read from the state). `A` is a learned parameter stored as its negated log so that the discretized transition values always land in `(0, 1)` for stable state evolution. `ssm` from `ssm.py` is called with these parameters, and the result is multiplied element-wise with the gate.

Step 4 is the backward SSM branch: the shared projection is time-reversed before the depthwise conv, the same structure is applied with a separate set of weights, and the output is time-reversed back to the original order before gating.

Step 5 is the residual merge: the forward and backward outputs are averaged, projected with a linear layer, and added to the original input as a skip connection.

**`IntraModalBiMamba`** wraps a `BiMambaBlock` per feature type and applies it independently to every key in the hiddens dict. Weight sharing mirrors `ModalityFrontend`: expert and novice streams for the same feature type share one `BiMambaBlock`. Output shape is `(B, L', C')` per modality, identical to the input.

### `tests/test_frontend.py`

A smoke-test script that loads one real window from the first training session, wraps it in a batch-of-1, initialises the `ModalityFrontend` on CPU with a JAX PRNGKey, runs a forward pass, and prints all input and output shapes plus the total parameter count. This confirms end-to-end correctness (data loading -> resampling -> windowing -> encoding -> projection) without needing a GPU or the full dataset.

### `src/inter_modal.py`

Two components in one file. First, `GumbelSinkhorn`: a small Flax module that takes temporal mean-pooled summaries of all `M` modality representations `(B, M, C')`, projects them through asymmetric query and key matrices to produce an `M x M` score matrix, adds Gumbel noise scaled by temperature `tau` for stochastic exploration during training, and applies iterative Sinkhorn-Knopp normalisation to produce a doubly-stochastic permutation matrix `P`. The `sinkhorn` function is a standalone utility (not a module) that alternates row and column normalisation on the exponentiated scores for a fixed number of iterations.

Second, `InterModalBiMamba`: stacks all modality representations into `(B, M, L', C')`, calls `GumbelSinkhorn` to get `P`, applies the soft reordering via `einsum("bij,bjlc->bilc", P, stacked)`, reshapes the result into `(B, L', MC')` by concatenating modalities along the channel axis, and passes it through a `BiMambaBlock` that models cross-modal dependencies. The output `H` has shape `(B, L', MC')`. Temperature `tau` is exposed as a call argument so the training loop can anneal it from a high exploration value to near-deterministic over the course of training (exponential decay schedule from the draft).

### `tests/test_inter_modal.py`

Verifies three things on CPU with synthetic data: (1) `P` is doubly stochastic (row and column sums within 1e-3 of 1.0), (2) output shape matches `(B, L', MC')`, and (3) the model runs without errors. Uses `M=8` modalities (4 features x 2 roles) with `C'=128` giving `MC'=1024`.


### `src/beta_head.py`

`BetaHead` is a per-timestep Flax module: `Dense -> SiLU -> two Dense(1) -> softplus + 1`. It accepts arbitrary leading dimensions so it works on both `(B, D)` pooled input and `(B, L', D)` per-frame input. `MultiHeadBeta` applies one `BetaHead` to the fused inter-modal output `(B, L', MC')` for the multimodal prediction, and one per feature type (shared across roles) to each per-modality tensor `(B, L', C')`. All outputs are per-frame `(B, L')`. Standalone functions `predictive_mean`, `predictive_variance`, and `nll_loss` operate on arbitrary shapes.

### `src/model.py`

The full `EngageNet` Flax module: `ModalityFrontend -> IntraModalBiMamba -> InterModalBiMamba -> MultiHeadBeta`. All hyperparameters come from `cnfg` (no duplicated constants). Returns `(multi_alpha, multi_beta, unimodal_dict)` all at per-frame `(B, L')` resolution.

### `src/train.py`

Flax + Optax training loop with AdamW. Per-frame Beta NLL loss against engagement targets `(B, L)`. CCC-based validation every `eval_every` epochs with early stopping (`patience` epochs without improvement). Gumbel-Sinkhorn temperature annealed via exponential decay. Checkpoints saved every `checkpoint_every` epochs plus a `best/` checkpoint on val CCC improvement. All hyperparameters from `EngageNetConfig.from_cli()`.

### `src/tta.py`

Test-time adaptation library (not a script). `sample_filter`: two-level uncertainty filter keeping samples where multimodal variance is low but unimodal variance is high (adaptive percentile thresholds). `beta_kl`: closed-form KL divergence between Beta distributions. `tta_loss`: mutual information sharing (sum of KL divergences) + pseudo-label supervision weighted by `lambda`. `surgical_mask`: walks the param tree, returns True only for BatchNorm layers, `conv1` in InitEncoder, and first Dense in cross-modal BiMamba. `tta_step`: computes TTA loss, takes gradients, zeros non-surgical ones via the mask, updates state.

### `src/metrics.py`

- `ccc`: Concordance Correlation Coefficient using population covariance.
- `combined_ccc`: unweighted mean across domains. 
- `cdd_gender` and `cdd_language`: Conditional Demographic Disparity metrics using equal-frequency binning of targets.

### `src/aggregator.py`

`aggregate_windows`: overlap-add that averages per-frame predictions from overlapping windows into a full session time series. Uncovered tail frames filled with last valid value.

### `src/inference.py`

Multi-corpus TTA inference script. Loads best (or latest) checkpoint, iterates over all corpora in `SUBMISSION_CORPORA`, processes each session window-by-window with optional TTA on qualifying samples, aggregates per-frame predictions via `aggregate_windows`, and writes submission CSVs in the challenge format (`{role}.engagement.annotation.csv`, one float per line at 25 Hz).

### `src/evaluate.py`

Scoring script. Loads ground-truth and prediction CSVs, computes per-session CCC, per-corpus CCC, Combined CCC, and saves `results.json`. Accepts `--corpora` and `--split` to evaluate on val or (when released) test.

### `tests/test_model.py`

Full forward pass smoke test with synthetic data on CPU (CORE_MODALITIES, batch=2). Verifies output shapes are `(B, L')` and alpha/beta > 1.
