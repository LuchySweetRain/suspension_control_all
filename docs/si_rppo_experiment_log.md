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
