# MVP Experiment Results

## Experiment Setup

Result directory: `outputs/mvp_train1000_test200_e20`

Configuration used by the current run:

| Item | Value |
|---|---:|
| Seed | `20260606` |
| Users `K` | `3` |
| Port grid `Nx x Ny` | `4 x 4` |
| Total ports `N` | `16` |
| Active ports `n` | `4` |
| Antenna area | `2.0lambda x 2.0lambda` |
| Transmit power | `20 dBm` |
| User distance | `0.2 km` |
| Train samples | `1000` |
| Validation samples | `200` |
| Test samples | `200` |
| GPT-2 layers | `2` |
| LoRA rank | `4` |
| Batch size | `16` |
| Epochs | `20` |
| Learning rate | `1e-6` |

This run is still an MVP-scale experiment. It verifies the end-to-end reproduction pipeline with a small GPT-2 backbone and limited data, not the full paper-scale setting.

## Test Sum Rate

| Method | Test sum rate |
|---|---:|
| Random | `19.177417755126953` |
| Proposed | `19.267733840942384` |

Proposed improves over Random by:

| Metric | Value |
|---|---:|
| Absolute gain | `0.09031608581543082 bps/Hz` |
| Relative gain | `0.47095019240160957%` |

The Proposed model is now ahead of the Random baseline in this run, but the margin is still small. This should be interpreted as a positive MVP sanity signal, not as a stable reproduction of the paper's claimed performance advantage.

## Training Dynamics

Because training loss and validation loss are defined as negative sum rate, a more negative loss means a higher sum rate.

Key epochs:

| Epoch | Train loss | Validation loss | Train sum rate | Validation sum rate |
|---:|---:|---:|---:|---:|
| 1 | `-13.04261279296875` | `-15.330747871398925` | `13.04261279296875` | `15.330747871398925` |
| 2 | `-16.173274185180663` | `-18.769741821289063` | `16.173274185180663` | `18.769741821289063` |
| 3 | `-18.333507705688476` | `-19.002129135131835` | `18.333507705688476` | `19.002129135131835` |
| 8 | `-18.984768493652343` | `-19.05345199584961` | `18.984768493652343` | `19.05345199584961` |
| 15 | `-19.03696110534668` | `-19.046499176025392` | `19.03696110534668` | `19.046499176025392` |
| 20 | `-19.025504440307618` | `-19.050069046020507` | `19.025504440307618` | `19.050069046020507` |

Best validation epoch:

| Metric | Value |
|---|---:|
| Best validation epoch | `8` |
| Best validation sum rate | `19.05345199584961` |

Best training epoch:

| Metric | Value |
|---|---:|
| Best training epoch | `15` |
| Best training sum rate | `19.03696110534668` |

Convergence pattern:

| Interval | Validation sum-rate change |
|---|---:|
| Epoch 1 to epoch 3 | `+3.6713812637329095` |
| Epoch 3 to epoch 20 | `+0.04793991088867244` |

The model learns most of its useful behavior in the first three epochs. After epoch 3, validation performance enters a plateau near `19.05 bps/Hz`.

Last-five-epoch stability:

| Metric | Value |
|---|---:|
| Mean validation sum rate, epochs 16-20 | `19.04725978088379` |
| Validation standard deviation, epochs 16-20 | `0.005325095882367446` |
| Mean training sum rate, epochs 16-20 | `19.023807427978515` |
| Training standard deviation, epochs 16-20 | `0.006998679426295112` |

The final five epochs are very stable. Continuing the same training setup for more epochs is unlikely to produce a large gain.

## Interpretation

The MVP result is technically healthy:

- The loss decreases quickly and then stabilizes.
- Validation does not collapse or show obvious overfitting.
- Proposed now slightly outperforms Random on the test set.
- The reported sum-rate scale is consistent across validation and test evaluation.

The result is also still limited:

- The improvement over Random is only about `0.47%`.
- The run uses GPT-2 first `2` layers, while the paper uses first `6` layers.
- The training set has `1000` samples, while the paper uses `10000`.
- The current Proposed evaluation still uses relaxed soft port selection, not strict hard unique inference.
- Only Random is implemented as a baseline; Transformer, CNN, and LLM-sequential are not included yet.
- The result is from a single seed, so there is no variance estimate.

## Comparison With Previous MVP Smoke Run

Earlier smoke-run result recorded in `docs/MVP_REPRODUCTION_PLAN.md`:

| Method | Smoke-run test sum rate |
|---|---:|
| Random | `19.097673416137695` |
| Proposed | `18.745282649993896` |

Current run:

| Method | Current test sum rate |
|---|---:|
| Random | `19.177417755126953` |
| Proposed | `19.267733840942384` |

The larger run improves the Proposed model from below Random to slightly above Random. This supports the decision to move beyond the tiny smoke setup, but it does not yet establish a robust performance hierarchy.

## Recommended Next Steps

1. Add hard unique port inference for Proposed evaluation.
2. Record `train_samples`, `val_samples`, `test_samples`, `epochs`, `gpt2_layers`, and `selection_mode` in `results.csv`.
3. Run at least three seeds with the current `train=1000`, `test=200`, `epochs=20` setup.
4. Increase GPT-2 layers from `2` to `6` after hard inference is implemented.
5. Add the Transformer baseline next, because it shares the most code with Proposed and is the cleanest architecture comparison.
6. Only after the above are stable, scale toward `train=10000`, `test=1000`, and the Fig.5-Fig.11 sweeps.

## Bottom Line

This run is a successful MVP reproduction result. It demonstrates that the implemented Proposed pipeline can train end-to-end and achieve a small positive gain over Random. The next technical priority is not more epochs, but stricter inference and stronger experimental controls.
