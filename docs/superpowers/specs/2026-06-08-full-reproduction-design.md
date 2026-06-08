# Full Paper Reproduction Design

## Goal

Reproduce the paper "LLM-Based Port Selection and Beamforming for Multiuser MISO With Fluid Antenna Systems" in a staged, verifiable way. The reproduction should move from the existing MVP to a full experiment pipeline that can train and evaluate the Proposed method, Random, Transformer, CNN, and LLM-sequential baselines, then generate the paper-style Fig.5-Fig.11 results and Table II selected port sets.

The first priority is experimental credibility, not immediate curve matching. Each stage must leave runnable code, recorded outputs, and tests before the next stage increases model size or experiment count.

## Current State

The repository already contains an MVP package under `llm_fas/` with:

- FAS geometry, Jake spatial correlation, path loss, and complex channel generation.
- Model-based beamforming from `p,q` and sum-rate calculation.
- Rectangular Gumbel-Sinkhorn relaxation and unique hard top-k helper.
- Proposed GPT-2 + PEFT LoRA MVP model.
- Deterministic dataset generation, Random baseline, training, and evaluation scripts.
- Unit and smoke tests that currently pass.
- MVP result documents showing a small positive Proposed-over-Random gain at `train=1000`, `test=200`, `epochs=20`, GPT-2 `2` layers.

The main MVP limitations are:

- Proposed evaluation still uses no-noise relaxed Sinkhorn instead of hard unique activated ports.
- Experiment metadata in `results.csv` is incomplete for later comparison.
- Only Random is implemented as a baseline.
- The current config is MVP-scale: `d_mha=128`, `mha_heads=4`, GPT-2 `2` layers, `batch_size=16`.
- LoRA is applied to GPT-2 `c_attn` as an MVP approximation, while the paper applies LoRA only to Q and V matrices.

## Paper Alignment

The full reproduction must use these paper settings unless a smaller debug config is explicitly selected:

- System: `K=3`, `Nx=Ny=4`, `N=16`, `n=4`, `W=2lambda x 2lambda`, `Pmax=20 dBm`, noise PSD `-174 dBm/Hz`, bandwidth `10 MHz`, carrier `2 GHz`, user distance `0.2 km`.
- Channel model: Jake spatial correlation, eigen-decomposed correlation matrix, CSCG small-scale fading, path loss `128.1 + 37.6 log10(d)` with `d` in kilometers.
- Preprocessing: split CSI real and imaginary parts, FC projection, real/imag multi-head attention, merge, FC to GPT-2 input embeddings, sinusoidal positional embedding.
- Proposed backbone: GPT-2 first `6` layers, `d_llm=768`, LoRA rank `4`, train LoRA matrices and LayerNorm while freezing original GPT-2 attention and FFN weights.
- Port selection: Gumbel noise, temperature `tau=max(0.1, 0.95^epoch)`, Sinkhorn iterations `10`, soft selection during training, hard selected ports during inference.
- Power factors: produce `2K` power outputs for `p` and `q`, normalize each row with softmax and scale by `Pmax`.
- Training: unsupervised negative sum-rate loss, Adam learning rate `1e-6`, up to `200` epochs, batch size `100`, early stopping patience `10`, `10000` training samples, `1000` test samples, validation set equal to `20%` of training data.

## Stage 1: Hard Evaluation And Metadata

Add a hard-inference path to the Proposed model and evaluation code.

During training, the model keeps using differentiable `selection_soft = gumbel_sinkhorn(...)`. During evaluation and test reporting, the model must convert port scores to a hard unique port set, build a one-hot selection matrix, compute `H_eff`, derive beamforming vectors, and calculate sum rate from the hard selected ports.

The forward result should expose both soft and hard-related tensors when available:

- `port_scores`: raw `[B,n,N]` logits.
- `selection_soft`: relaxed `[B,n,N]` matrix.
- `ports`: hard `[B,n]` indices during hard inference.
- `selection_hard`: one-hot `[B,n,N]` matrix during hard inference.
- `selection`: the matrix actually used for `H_eff`.
- `rate`: sum rate computed from the active selection mode.

Evaluation output must include enough metadata to compare runs:

- `method`, `selection_mode`, `K`, `Nx`, `Ny`, `N`, `n_active`, `W_lambda_x`, `W_lambda_y`, `Pmax_dBm`, `distance_km`, `seed`, `train_samples`, `val_samples`, `test_samples`, `epochs`, `batch_size`, `gpt2_layers`, `d_mha`, `mha_heads`, `test_sum_rate`.

Acceptance:

- Existing tests still pass.
- New tests verify hard inference returns unique ports and hard one-hot rows.
- Evaluation writes `selection_mode=hard` for Proposed and positive finite rates for Random and Proposed.

## Stage 2: MVP+ Stability Runs

Keep the current computational scale and run multiple seeds before increasing model size.

Recommended debug configuration:

- `train_samples=1000`
- `val_samples=200`
- `test_samples=200`
- `epochs=20`
- GPT-2 layers `2`
- `d_mha=128`
- `batch_size=16`

Add a small experiment runner that can execute the same config over a seed list and aggregate results into one CSV. It should not require manually editing `configs/mvp.yaml` between runs.

Acceptance:

- At least three seeds complete with hard Proposed evaluation.
- Aggregate CSV reports mean and standard deviation for Random and Proposed.
- The docs record whether hard Proposed is consistently better than Random. If it is not, the next work item is diagnosing port selection or training behavior instead of scaling up.

## Stage 3: Transformer Baseline

Implement the Transformer baseline before CNN and LLM-sequential because it shares the cleanest comparison with Proposed.

The Transformer baseline should reuse:

- The same FAS channel generator.
- The same preprocessing contract where practical.
- The same parallel output heads for port logits and `p,q`.
- The same Gumbel-Sinkhorn training and hard inference evaluation.
- The same model-based beamforming and sum-rate loss.

Only the backbone changes: replace GPT-2 + LoRA with a Transformer encoder-style module using 8-head attention in the full setting. MVP tests can use smaller dimensions.

Acceptance:

- A smoke training run writes a checkpoint and result row for `transformer`.
- Evaluation can produce one CSV containing `random`, `transformer`, and `proposed`.
- The result schema remains identical across methods.

## Stage 4: Paper-Scale Proposed And Fig.5

After hard inference and the Transformer baseline are stable, add full paper configs.

Create separate config files instead of overwriting MVP config:

- `configs/paper_default.yaml`
- `configs/fig5_batch_size.yaml`

`paper_default.yaml` should set `d_mha=768`, `mha_heads=8`, `gpt2_layers=6`, `lora_rank=4`, `batch_size=100`, `epochs=200`, early stopping patience `10`, `train_samples=10000`, `test_samples=1000`, and validation samples consistent with the paper's 20% training split.

Fig.5 should train Proposed under `batch_size in {50,100,200}` and save per-epoch train and validation losses.

Acceptance:

- A default full-scale Proposed run completes or produces a documented resource limit with command, hardware, and failure reason.
- Fig.5 runner writes per-batch-size training histories.
- Plot script generates a convergence figure from recorded CSVs.

## Stage 5: Remaining Baselines And Fig.6-Fig.11

Add CNN and LLM-sequential after the shared parallel baseline machinery is stable.

CNN baseline:

- Sequentially optimize port selection and beamforming-related power factors.
- Use 2D convolution over real/imag CSI features followed by FC output heads.
- Use the same model-based beamforming formula for final `C`.

LLM-sequential baseline:

- Use GPT-2/LoRA to output port selection first.
- Use CNN/FC-style power factor prediction after port selection.
- Use the same hard inference and beamforming derivation.

Sweep runner responsibilities:

- Fig.6: sum rate versus `Wx=Wy`.
- Fig.7: sum rate versus `Pmax`.
- Fig.8: sum rate versus active ports `n`.
- Table II: selected port sets for one fixed test channel and `n in {3,4,5,6,7}`.
- Fig.9: sum rate versus distance `d`.
- Fig.10: sum rate versus total ports `N`.
- Fig.11: Proposed inference time versus `N` for `n in {3,4,5}`.

Each sweep point should generate its own output directory with config snapshot, training history, test results, selected ports where relevant, and plot-ready CSV rows. When `N` or `n` changes, the model must be rebuilt because sequence length and output dimensions change.

Acceptance:

- Every paper figure has a corresponding CSV and generated plot file.
- Every result row includes full config metadata and method name.
- Table II exports 1-based port indices using the paper's top-left to bottom-right mapping.

## Architecture And Data Flow

The full codebase should keep the existing package shape and add focused modules rather than growing one large script.

Recommended boundaries:

- `llm_fas/models.py`: shared model contracts and Proposed model.
- `llm_fas/baselines.py`: Random baseline plus trainable baseline model classes or dispatch helpers.
- `llm_fas/train.py`: generic train/evaluate loops that accept method names.
- `llm_fas/experiments.py`: seed runs, sweep expansion, output-directory management.
- `llm_fas/plotting.py`: figure generation from CSVs.
- `scripts/train_mvp.py` and `scripts/evaluate_mvp.py`: keep as compatibility wrappers.
- New scripts such as `scripts/run_seeds.py`, `scripts/run_sweep.py`, and `scripts/plot_results.py`: use the generic experiment layer.

The primary tensor contract remains:

- `H`: `[B,K,N]` complex full CSI.
- `selection`: `[B,n,N]` real soft or hard selection matrix.
- `H_eff = H @ selection.transpose(-1,-2)`: `[B,K,n]`.
- `p,q`: `[B,K]`.
- `C`: `[B,n,K]`.
- `rate`: `[B]`.

## Error Handling And Reproducibility

Training and evaluation should fail early on:

- Non-finite loss or rate.
- Duplicate hard ports.
- Selection rows that do not sum to one.
- Mismatched checkpoint/config dimensions.
- Missing checkpoint during evaluation.

Every experiment output directory should contain:

- A copied config snapshot.
- `train_history.csv` for trainable methods.
- `results.csv`.
- `selected_ports.json` when port sets are reported.
- Generated figures where applicable.

Large checkpoints stay ignored by git. Small CSV and documentation artifacts can be committed when they summarize a completed milestone.

## Testing Strategy

Tests should stay fast by monkeypatching GPT-2 with tiny local configs.

Required test coverage additions:

- Proposed hard inference produces unique ports and one-hot selection.
- Evaluation CSV includes complete metadata and `selection_mode`.
- Generic evaluation handles multiple methods.
- Transformer baseline forward pass returns the same tensor contract as Proposed.
- Sweep config expansion produces expected parameter values without running full training.
- Plotting functions can consume minimal CSV fixtures and write files.

Long training runs are verified by scripts and recorded outputs rather than unit tests.

## Non-Goals For The First Full-Reproduction Pass

The first full pass does not require exact digitized matching of the paper's plotted y-values. It must reproduce the paper settings, method ordering, and qualitative trends first. If curve gaps remain after the pipeline is stable, a later calibration pass can inspect random seeds, baseline architecture details, strict Q/V-only LoRA, and possible differences in channel normalization.

The MVP PEFT `c_attn` LoRA approximation remains acceptable through Stage 4 unless it becomes a clear source of mismatch. Strict Q/V-only LoRA should be implemented as a later accuracy-improvement stage after the experiment pipeline can already reproduce all figures.

## Risks

- Full paper settings may exceed local GPU memory or take a long time on CPU. The design keeps MVP, seed, and full configs separate so smaller verification remains available.
- Hard inference may initially reduce Proposed performance versus relaxed evaluation. This is expected and should be diagnosed before scaling.
- CNN and LLM-sequential baseline details are less explicit than Proposed in the paper. Their implementations should be documented as faithful engineering interpretations using the same final beamforming formula.
- Sweep experiments are expensive because each `N` or `n` point changes model dimensions and usually requires retraining.
