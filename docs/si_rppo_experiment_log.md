# SI-RPPO Experiment Log

## 2026-06-01: e20 Full-Scenario Short Matrix

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_baselines --episodes 20 --expert-episodes 20 --baseline-algorithms td3,sac
```

Scope:

- 5 full-car road scenarios.
- 20 training episodes per learned controller.
- 8000 expert transitions from `FULL_CAR_MPC_LITE`.
- PPO variants: `ppo_scratch`, `bc_ppo`, `residual_bc_ppo`, `safe_residual_bc_ppo`.
- Off-policy baselines: `td3_baseline`, `sac_baseline`.

Mean learned-controller metrics:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | Saturation | DeviationRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -13488.9463 | 56.2 | 2.9654 | 6.4882 | 3.5621 | 1384.0140 | 0.00 | 0.0000 |
| bc_ppo | PPO | -22849.8424 | 170.2 | 4.3132 | 7.3389 | 4.5073 | 1563.7225 | 0.00 | 0.0000 |
| residual_bc_ppo | PPO | -11693.9333 | 0.0 | 3.0541 | 6.2251 | 1.8300 | 852.2938 | 0.00 | 1949.7702 |
| safe_residual_bc_ppo | PPO | -8968.0470 | 0.6 | 2.7284 | 4.4785 | 0.9799 | 410.2148 | 0.00 | 912.1366 |
| td3_baseline | TD3 | -18649.9936 | 301.0 | 0.0316 | 3.5193 | 0.8706 | 307.4573 | 0.75 | 0.0000 |
| sac_baseline | SAC | -3406.5066 | 0.0 | 0.4243 | 2.7398 | 0.0339 | 0.0391 | 0.50 | 0.0000 |

Claim report status:

- `imitation_initialization`: weak or contradicted.
- `residual_prior_structure`: supported.
- `safe_residual_gate`: weak or contradicted.
- `safe_residual_ppo_vs_td3`: weak or contradicted under stricter comfort/action criteria.
- `safe_residual_ppo_vs_sac`: weak or contradicted.

Interpretation:

- The residual prior structure is the strongest current algorithmic signal. Compared with direct BC-PPO, residual BC-PPO improved return, comfort metrics, roll response, action smoothness, and removed unsafe steps.
- Direct BC-PPO is currently worse than PPO from scratch. The likely issue is that behavior cloning directly imitates a high-force expert policy without a residual constraint, then PPO fine-tuning destabilizes early online behavior.
- Safe residual PPO improves return and actuator smoothness over residual BC-PPO, but it introduced a small unsafe count on `mixed_full`. The safety gate is not yet strict enough to be a publishable safety claim.
- SAC is a strong baseline in this short run. Any paper claim must either beat SAC under the same constraints or define a safety/feasibility condition where SI-RPPO has a defensible advantage.

Next algorithm actions:

1. Add a BC-only evaluation checkpoint before PPO fine-tuning to separate imitation quality from online PPO degradation. Done.
2. Add behavior-cloning anchoring during early PPO updates so direct BC-PPO cannot drift aggressively away from the expert. Done.
3. Strengthen safe residual gating with a suspension-margin deadband and residual scale schedule. Done.
4. Add a hard residual safety shield that shrinks `delta_u_ppo` when current suspension or roll/pitch margins are near limits. Done.
5. Re-run `episodes=20` and require all SI-RPPO claim report rows to be supported before treating the approach as a thesis-grade result.

## 2026-06-01: Anchor + Shield Implementation Smoke

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_anchor_shield_smoke --episodes 1 --expert-episodes 1 --expert-max-steps 2 --train-scenario-limit 1 --eval-scenario-limit 1 --episode-seconds 0.05 --mujoco-settle-seconds 0.2 --variants safe_residual_bc_ppo
```

Result:

- Training chain completed.
- `imitation_pretrained_eval.json` was written for the BC-only checkpoint.
- PPO `training_history.json` records `bc_anchor_weight`.
- Safe residual action metrics were reduced in the smoke run, with zero unsafe steps.

This smoke run is not paper evidence. It only verifies that the anchor/shield algorithm path executes before the next `episodes=20` matrix.

## 2026-06-01: Variant-Isolated Anchor/Shield Matrix

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_variant_isolation --episodes 20 --expert-episodes 20 --baseline-algorithms td3,sac
```

Scope:

- 5 full-car road scenarios.
- 20 training episodes per learned controller.
- Direct BC-PPO kept the decaying BC anchor.
- Residual BC-PPO and safe residual BC-PPO disabled the direct BC anchor so the residual ablation isolates the prior/residual structure and safety shield.
- Off-policy baselines: `td3_baseline`, `sac_baseline`.

Mean learned-controller metrics:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | CommandDeltaRMS | TrackingRMS | Saturation | DeviationRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -13488.9463 | 56.2 | 2.9654 | 6.4882 | 3.5621 | 1384.0140 | 6059.2404 | 3182.1669 | 0.00 | 0.0000 |
| bc_ppo | PPO | -13137.0083 | 42.6 | 2.8478 | 6.8654 | 3.2268 | 1521.3790 | 6522.6256 | 3411.1139 | 0.00 | 0.0000 |
| residual_bc_ppo | PPO | -25833.8324 | 0.4 | 6.5950 | 9.1280 | 2.4667 | 779.1551 | 2279.1713 | 1175.0026 | 0.00 | 1859.2799 |
| safe_residual_bc_ppo | PPO | -28372.5205 | 0.0 | 7.5008 | 8.7428 | 1.1060 | 387.4188 | 679.5841 | 578.0809 | 0.00 | 590.8190 |
| td3_baseline | TD3 | -18649.9936 | 301.0 | 0.0316 | 3.5193 | 0.8706 | 307.4573 | 1066.7217 | 459.7862 | 0.75 | 0.0000 |
| sac_baseline | SAC | -3406.5066 | 0.0 | 0.4243 | 2.7398 | 0.0339 | 0.0391 | 0.0630 | 0.0584 | 0.50 | 0.0000 |

Interpretation:

- Direct BC anchoring helps early PPO return and unsafe steps relative to PPO scratch, but it still increases action delta. This supports imitation as a useful initialization, not yet as a smooth control method.
- Residual learning around `FULL_CAR_MPC_LITE` is not enough when the same controller is used as both prior and expert. The residual labels are near zero, so the learned policy is initialized close to a prior whose expert dataset has nonzero unsafe behavior.
- Safe residual gating reduces unsafe steps, action delta, command delta, actuator tracking error, and deviation relative to ungated residual PPO, but the return and comfort losses are too large for a CCFA-level claim.
- The next algorithm correction is therefore safe expert curation: record raw expert safety metrics, filter unsafe expert transitions, and allow an alternative conservative teacher such as `PASSIVE` while retaining `FULL_CAR_MPC_LITE` as the residual prior.

Implemented follow-up:

- `scripts/collect_expert_dataset.py` now supports `--skip-unsafe` and records `raw_transitions`, `skipped_unsafe`, `raw_unsafe_fraction`, and filtered `unsafe_fraction`.
- `scripts/run_si_rppo_ablation.py` now supports `--expert-controller` and `--skip-unsafe-expert` so the SI-RPPO pipeline can test safe-teacher imitation without changing training code by hand.

## 2026-06-01: Policy Safety Layer Matrix

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_policy_safety --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --baseline-algorithms td3,sac
```

Algorithm change:

- Added `policy_safety`, a configurable learned-policy action projection layer.
- The layer enforces maximum action fraction, maximum per-step action change, and safety-margin action shrinkage near suspension/pitch/roll/wheel-displacement limits.
- The same layer is applied during PPO/TD3/SAC training and learned-controller evaluation.
- Residual controller evaluation now uses the same residual shield used during training.

Mean learned-controller metrics:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | CommandDeltaRMS | TrackingRMS | Saturation | DeviationRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -9962.1158 | 21.0 | 2.4933 | 5.2429 | 3.8918 | 169.1610 | 179.8305 | 252.4103 | 0.00 | 0.0000 |
| bc_ppo | PPO | -9760.0900 | 19.8 | 2.3206 | 5.4087 | 3.7434 | 179.6356 | 187.1570 | 268.0398 | 0.00 | 0.0000 |
| residual_bc_ppo | PPO | -9467.0002 | 10.2 | 3.3299 | 4.6581 | 0.7102 | 123.4383 | 184.2478 | 184.1860 | 0.00 | 1784.1503 |
| safe_residual_bc_ppo | PPO | -11731.3797 | 3.0 | 4.3145 | 5.1715 | 0.2169 | 137.8060 | 157.8957 | 205.6246 | 0.00 | 1664.5975 |
| td3_baseline | TD3 | -19406.2439 | 301.0 | 1.0131 | 2.8833 | 1.1454 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| sac_baseline | SAC | -3307.5951 | 0.0 | 0.8409 | 2.9693 | 0.5187 | 151.9442 | 394.2582 | 226.7207 | 0.00 | 0.0000 |

Claim report status:

- `imitation_initialization`: weak or contradicted, but now improves return and unsafe steps; action delta is the remaining failure.
- `residual_prior_structure`: supported.
- `safe_residual_gate`: weak or contradicted; it improves unsafe steps and deviation but loses return and action smoothness.
- `safe_residual_ppo_vs_td3`: weak under comfort/action criteria, though it improves return and unsafe steps.
- `safe_residual_ppo_vs_sac`: weak; SAC remains the strongest short-horizon return/safety baseline.

Interpretation:

- The safety layer changed the failure mode in a useful direction. Compared with the previous safe-teacher run, `bc_ppo` unsafe steps dropped from 275.6 to 19.8 and action delta dropped from 999.9 N to 179.6 N.
- Residual PPO is again the strongest PPO-family variant by return and unsafe count tradeoff before applying the stricter safe gate.
- The publishable algorithm gap is now clearer: PPO needs an update objective that accounts for projected/executed actions, not only a post-policy action filter. The next revision should log action projection error and penalize advantage updates that repeatedly rely on large safety-layer corrections.

## 2026-06-01: Projection-Aware PPO Penalty

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_projection_penalty --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --baseline-algorithms td3,sac
```

Algorithm change:

- PPO trajectories now store the normalized difference between raw composed action and projected executed action.
- PPO actor loss includes `projection_penalty_weight * ratio * projection_error`, reducing probability mass on samples that require strong safety-layer correction.
- Training history records `mean_projection_error`.

Mean learned-controller metrics:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | CommandDeltaRMS | TrackingRMS | Saturation | DeviationRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -18671.3168 | 301.0 | 0.0844 | 2.8096 | 0.5472 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 |
| bc_ppo | PPO | -3213.2397 | 0.0 | 0.4466 | 2.8124 | 0.1422 | 80.2526 | 200.2387 | 119.7474 | 0.00 | 0.0000 |
| residual_bc_ppo | PPO | -8953.1051 | 8.6 | 3.3001 | 4.5031 | 0.6209 | 116.8130 | 168.5640 | 174.3002 | 0.00 | 1773.5502 |
| safe_residual_bc_ppo | PPO | -11950.7542 | 7.0 | 4.3437 | 5.2310 | 0.1478 | 136.8291 | 156.3471 | 204.1669 | 0.00 | 1725.7285 |

Interpretation:

- The projection-aware penalty is very effective when paired with offline imitation: `bc_ppo` reaches zero unsafe steps and return close to the passive safe teacher while still using nonzero active control.
- Applying the same projection penalty to PPO from scratch is not a fair baseline because it can collapse to near-zero actions while still failing safety on large suspension excursions.
- The ablation runner now disables projection penalty for `ppo_scratch`; future evidence should compare standard PPO scratch against projection-aware BC/Residual PPO.
- The strongest current thesis direction is safe-teacher BC followed by projection-aware PPO fine-tuning. Residual control remains useful for structure but is not yet the best-performing branch under the current `FULL_CAR_MPC_LITE` prior.

Follow-up focused check:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_projection_bc_focus --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --variants ppo_scratch,bc_ppo
```

This run uses standard PPO scratch with the safety layer but without projection penalty, and BC-PPO with projection-aware penalty:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | TrackingRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -9962.1158 | 21.0 | 2.4933 | 5.2429 | 3.8918 | 169.1610 | 252.4103 |
| bc_ppo | PPO | -3213.2397 | 0.0 | 0.4466 | 2.8124 | 0.1422 | 80.2526 | 119.7474 |

This is the strongest current result for the PPO-centered paper direction. It supports the claim that safe offline imitation plus projection-aware PPO fine-tuning can strongly improve early online safety, return, comfort, and action smoothness over standard PPO under the same action-safety layer.

## 2026-06-01: Core PPO Claim With Projection Metrics

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_projection_core_metrics --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --variants ppo_scratch,bc_ppo
```

Report status:

- Overall report: `incomplete`, because residual and off-policy branches were intentionally not included.
- Core PPO claim: `supported`.
- New projection metrics are written as `PolicyProjectionError` and `PolicyProjectionDeltaRMS_N`.

Core PPO comparison:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | TrackingRMS | ProjectionError | ProjectionDeltaRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -9962.1158 | 21.0 | 2.4933 | 5.2429 | 3.8918 | 169.1610 | 252.4103 | 0.4804 | 2402.1286 |
| bc_ppo | PPO | -3213.2397 | 0.0 | 0.4466 | 2.8124 | 0.1422 | 80.2526 | 119.7474 | 0.1926 | 962.8655 |

Interpretation:

- The core PPO claim is now explicitly captured by the automatic claim report as `projection_aware_imitation`.
- Safe-teacher BC plus projection-aware PPO improves all required core metrics over standard PPO: return, unsafe steps, body/pitch/roll comfort, action delta, actuator tracking, and projection error.
- This is the current best CCFA-level algorithm seed: the novelty is not PPO itself, but a safety-curated offline-to-online PPO update that penalizes reliance on action projection in an active-suspension actuator-constrained setting.

## 2026-06-01: Projection-Aware Full Matrix

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_projection_full_matrix --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --baseline-algorithms td3,sac
```

Report status:

- Overall report: `ready`.
- Core PPO claim: `supported`.
- Residual-prior and safe-residual claims: weak or contradicted.
- TD3/SAC comparisons remain useful baselines, but they do not currently displace the PPO-centered core claim.

Mean learned-controller metrics:

| Variant | Controller | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | TrackingRMS | ProjectionError | DeviationRMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -9962.1158 | 21.0 | 2.4933 | 5.2429 | 3.8918 | 169.1610 | 252.4103 | 0.4804 | 0.0000 |
| bc_ppo | PPO | -3213.2397 | 0.0 | 0.4466 | 2.8124 | 0.1422 | 80.2526 | 119.7474 | 0.1926 | 0.0000 |
| residual_bc_ppo | PPO | -8953.1051 | 8.6 | 3.3001 | 4.5031 | 0.6209 | 116.8130 | 174.3002 | 0.2013 | 1773.5502 |
| safe_residual_bc_ppo | PPO | -11950.7542 | 7.0 | 4.3437 | 5.2310 | 0.1478 | 136.8291 | 204.1669 | 0.3335 | 1725.7285 |
| td3_baseline | TD3 | -19406.2439 | 301.0 | 1.0131 | 2.8833 | 1.1454 | 0.0000 | 0.0000 | 0.0399 | 0.0000 |
| sac_baseline | SAC | -3307.5951 | 0.0 | 0.8409 | 2.9693 | 0.5187 | 151.9442 | 226.7207 | 0.2888 | 0.0000 |

Interpretation:

- The PPO-centered contribution is now the strongest thread: safe-teacher BC plus projection-aware PPO beats standard PPO on return, unsafe steps, comfort, action smoothness, actuator tracking, and projection reliance.
- SAC remains close in return and safety, but BC-PPO has lower body/pitch/roll accelerations, lower action delta, lower actuator tracking error, and lower projection error in this matrix.
- Residual learning with the current `FULL_CAR_MPC_LITE` prior is not the best main-claim route. It should be reframed as a secondary extension or postponed until the reduced MPC/LPV prior improves.
- The next CCFA-strengthening step is repeated-seed or held-out-road validation of the `projection_aware_imitation` claim.

## 2026-06-01: Projection-Aware Seed Sweep and Safety-Regularized PPO

Commands:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed_sweep_e10 --seeds 42,43,44 --episodes 10 --expert-episodes 10
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed43_e20_check --seeds 43 --episodes 20 --expert-episodes 20
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_check --seeds 44 --episodes 20 --expert-episodes 20
```

Evidence summary:

| Run | Supported Seeds | Status | Key Observation |
| --- | ---: | --- | --- |
| `projection_seed_sweep_e10` | 2 / 3 | `needs_more_evidence` | Seeds 42 and 44 supported; seed 43 was unstable under a short 10-episode budget. |
| `projection_seed43_e20_check` | 1 / 1 | `supported` | Seed 43 becomes supported at 20 episodes, improving return, unsafe steps, action smoothness, tracking, and projection error. |
| `projection_seed44_e20_check` | 0 / 1 | `needs_more_evidence` | BC-PPO improves comfort, action smoothness, tracking, and projection error, but worsens return and unsafe steps. |

Seed 44 e20 core comparison:

| Variant | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | TrackingRMS | ProjectionError |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | -13478.2070 | 114.0 | 2.8986 | 3.8833 | 3.2651 | 95.5528 | 142.5772 | 0.3261 |
| bc_ppo | -17939.6776 | 296.2 | 2.5759 | 3.3103 | 0.2617 | 3.5283 | 5.2647 | 0.0331 |

Interpretation:

- Repeated-seed evidence is not yet strong enough for a final CCFA-level robustness claim.
- The failure mode is informative: safe-teacher BC-PPO can become extremely smooth and projection-consistent, but too passive on some seeds/roads, causing safety degradation from insufficient active force.
- The algorithmic correction is to make safety violations explicit in the PPO update, not only in the environment reward. PPO trajectories now record per-step `unsafe` flags, and the actor objective supports `unsafe_penalty_weight * ratio * unsafe`.
- `ppo_scratch` keeps this weight at zero in ablation configs so it remains a standard PPO baseline; imitation-initialized PPO receives both projection and unsafe-transition regularization.

Follow-up regression:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_unsafe_regularized --seeds 44 --episodes 20 --expert-episodes 20
```

Unsafe regularization reduced seed 44 BC-PPO unsafe steps from `296.2` to `40.2` and improved return from `-17939.6776` to `-12318.1044`, beating PPO scratch on both return and unsafe steps. However, action delta, actuator tracking, and roll RMS became worse than scratch. This means the next objective must jointly penalize unsafe likelihood and actuator roughness.

Implemented correction:

- PPO trajectories now also record normalized executed-action delta.
- PPO supports `action_delta_penalty_weight * ratio * action_delta`.
- `configs/mujoco_full_car_safe_ppo.yaml` enables this smoothness penalty for the proposed method.
- `ppo_scratch` keeps projection, unsafe, and action-delta penalties at zero in ablation configs so it remains a standard PPO baseline.

Constraint-regularized regression:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_constraint_regularized --seeds 44 --episodes 20 --expert-episodes 20
```

This version further reduced seed 44 unsafe steps to `8.2` and improved return to `-11892.9002`, but it still failed the full core claim because body/pitch comfort, action delta, tracking error, and projection error were worse than PPO scratch. The active-suspension tradeoff is now explicit: the method can become safe, but not yet simultaneously safe, smooth, and projection-consistent on the hard seed.

Next algorithm step:

- Treat seed 44 as the main hard-case benchmark.
- Tune or adapt the constraint weights instead of using fixed global coefficients, e.g. increase `lambda_delta` only when unsafe rate is already below target, or use a Lagrangian update for `lambda_unsafe`, `lambda_delta`, and `lambda_proj`.
- Add held-out road-condition labels to identify whether this failure is caused by a particular road/scenario rather than seed stochasticity alone.
- Re-run the 3-seed e20 sweep after adaptive constraint weighting. The acceptance criterion is that `projection_aware_imitation` is supported on all planned seeds, or that any unsupported seed has a reproducible, diagnosable road-condition failure.

## 2026-06-01: Adaptive Constraint PPO and Feasibility-Gate Ablation

Algorithm change:

- PPO now supports adaptive Lagrangian-style constraint weights for projection error, unsafe fraction, and executed-action delta.
- After each trajectory, the weights update as `lambda <- clip(lambda + lr * (observed_constraint - target_constraint), 0, lambda_max)`.
- Training history records the current `projection_penalty_weight`, `unsafe_penalty_weight`, and `action_delta_penalty_weight` per episode, making the constraint adaptation auditable.
- PPO checkpoints store the current adaptive weights.

Hard-case command:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_adaptive_constraint --seeds 44 --episodes 20 --expert-episodes 20
```

Seed 44 adaptive-constraint result:

| Variant | Return | UnsafeSteps | BodyAccRMS | PitchAccRMS | RollAccRMS | ActionDeltaRMS | TrackingRMS | ProjectionError |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | -13478.2070 | 114.0 | 2.8986 | 3.8833 | 3.2651 | 95.5528 | 142.5772 | 0.3261 |
| bc_ppo | -11438.7318 | 18.6 | 2.5868 | 6.3926 | 3.2169 | 175.7811 | 262.2883 | 0.5808 |

Interpretation:

- Adaptive constraints improve the seed 44 return and reduce unsafe steps relative to PPO scratch.
- The full core claim is still weak because pitch, action smoothness, actuator tracking, and projection error are worse.
- Training logs show the projection multiplier reaches the configured maximum, but projection error remains high. This suggests the current actor has an action-distribution mismatch that cannot be solved by scalar penalties alone.

Additional ablation:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_feasibility_gated --seeds 44 --episodes 20 --expert-episodes 20
```

The feasibility-gated positive-advantage update was too conservative on seed 44: BC-PPO collapsed toward near-zero actions, giving excellent smoothness/projection metrics but `301.0` unsafe steps and return `-23182.5172`. The mechanism remains implemented as an optional ablation (`feasibility_advantage_weight`), but it is disabled in the default safe PPO config until redesigned.

Next algorithm correction:

- Replace scalar action outputs with a projection-aware action parameterization, e.g. train PPO to output a feasible action increment or a safety-layer preconditioned action rather than an unconstrained raw force.
- Alternatively, make the feasibility gate state-dependent: apply it only after unsafe rate is below target, or gate only unsafe-positive samples instead of all projection/action-delta violations.
- The current publishable novelty candidate is now: **adaptive constraint-regularized offline-to-online PPO**, with seed 44 serving as the hard-case evidence that motivates projection-aware policy parameterization.

## 2026-06-01: Delta-Parameterized Projection-Aware PPO

Algorithm change:

- Added `policy_action_parameterization`, a configurable layer that interprets direct PPO actor output as a bounded increment from the previous executed action.
- The parameterization is applied before the hard safety projection during PPO training and learned-policy evaluation.
- `ppo_scratch` keeps the layer disabled, so the baseline remains standard raw-force PPO under the same downstream safety layer.
- `bc_ppo` enables the delta parameterization; residual branches keep it disabled so residual-prior conclusions stay separate.

Mechanism:

```text
u_pre,t = clip(
    u_executed,t-1
  + max_delta_fraction * force_limit * tanh_like_actor_output_t,
    +/- max_action_fraction * force_limit
)
u_executed,t = Projection_U(u_pre,t)
```

This directly attacks the failure observed in adaptive-constraint PPO: scalar penalties could increase `lambda_proj`, but the actor could still generate raw actions far outside the feasible actuator-rate manifold. Delta parameterization makes the actor's action space itself closer to the executed-control manifold.

Commands:

```text
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed42_e20_delta_parameterized --seeds 42 --episodes 20 --expert-episodes 20
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed43_e20_delta_parameterized --seeds 43 --episodes 20 --expert-episodes 20
python scripts/run_projection_seed_sweep.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/projection_seed44_e20_delta_parameterized --seeds 44 --episodes 20 --expert-episodes 20
```

Core PPO repeated-seed result:

| Seed | CoreStatus | ReturnDelta | UnsafeDelta | BodyDelta | PitchDelta | RollDelta | ActionDeltaDelta | TrackingDelta | ProjectionErrorDelta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | supported | 2303.4808 | -12.2 | 0.7717 | -1.9244 | -2.0401 | -106.1663 | -158.4140 | -0.4647 |
| 43 | supported | 8329.6916 | -48.4 | -0.9999 | -2.1375 | -3.1602 | -77.6082 | -115.8016 | -0.4190 |
| 44 | supported | 8353.1791 | -109.0 | -1.2809 | -0.0299 | -0.2015 | -27.6047 | -41.1899 | -0.3065 |

Interpretation:

- The core claim is now supported on seeds 42, 43, and 44 under the e20 repeated-seed protocol.
- Seed 44, previously the hard counterexample, is repaired: unsafe steps drop from `114.0` to `5.0`, action delta from `95.5528` to `67.9480`, tracking from `142.5772` to `101.3873`, and projection error from `0.3261` to `0.0196`.
- Seed 42 still has a body-acceleration tradeoff (`+0.7717` RMS), so the publishable claim should not say every comfort metric improves on every seed. The defensible claim is that the method robustly improves return, safety, pitch/roll comfort, actuator smoothness, tracking, and projection reliance, with body acceleration requiring a comfort-weight ablation.
- This is the strongest current CCFA-level algorithm thread: **safety-curated offline imitation + adaptive constraint PPO + delta-parameterized projection-aware action space**.

## 2026-06-01: Delta-Parameterized Full Matrix

Command:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --out results/si_rppo_e20_delta_parameterized_full_matrix --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --baseline-algorithms td3,sac
```

Evidence table:

```text
python scripts/build_delta_ppo_evidence_table.py --seed-dirs results/projection_seed42_e20_delta_parameterized results/projection_seed43_e20_delta_parameterized results/projection_seed44_e20_delta_parameterized --full-matrix-dir results/si_rppo_e20_delta_parameterized_full_matrix --out results/delta_ppo_evidence_table
```

Report status:

- Overall claim report: `ready`.
- Core PPO claim: `supported`.
- Direct delta-parameterized BC-PPO vs TD3/SAC: `weak_or_contradicted`.
- Residual-prior and safe-residual claims: still weak.

Mean full-matrix learned-controller metrics:

| Variant | Controller | Return | Unsafe | Body | Pitch | Roll | ActionDelta | Tracking | ProjectionError |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_scratch | PPO | -9962.1158 | 21.0 | 2.4933 | 5.2429 | 3.8918 | 169.1610 | 252.4103 | 0.4804 |
| bc_ppo | PPO | -7658.6350 | 8.8 | 3.2650 | 3.3185 | 1.8518 | 62.9946 | 93.9962 | 0.0157 |
| residual_bc_ppo | PPO | -7776.8564 | 2.0 | 3.5037 | 3.8325 | 0.5596 | 144.3384 | 215.3717 | 0.3141 |
| safe_residual_bc_ppo | PPO | -12061.8027 | 10.0 | 4.3393 | 5.2741 | 0.1044 | 133.5292 | 199.2430 | 0.3331 |
| td3_baseline | TD3 | -3337.5338 | 0.0 | 0.4369 | 2.7373 | 0.2017 | 1.0743 | 1.6031 | 0.0001 |
| sac_baseline | SAC | -3064.9345 | 0.0 | 0.4208 | 2.7443 | 0.0174 | 0.0764 | 0.1139 | 0.0000 |

Interpretation:

- Delta-parameterized BC-PPO is now a strong improvement over standard PPO scratch in the full matrix: return, unsafe steps, pitch/roll comfort, action smoothness, actuator tracking, and projection error all improve.
- The method still does not beat TD3/SAC or passive-like behavior in this short-horizon setup. TD3/SAC learn near-passive low-action policies with zero unsafe steps and much lower actuator activity.
- This means the current paper claim should be framed as an algorithmic fix for unsafe PPO exploration under actuator projection, not yet as a full replacement for off-policy baselines.
- The next CCFA-strengthening algorithm step is a safe-teacher improvement gate: the learned controller should deviate from passive only when the policy has evidence that active force improves comfort/safety. This targets the remaining gap to passive/SAC without discarding the successful delta-parameterized PPO mechanism.
