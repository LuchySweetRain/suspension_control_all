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

1. Add a BC-only evaluation checkpoint before PPO fine-tuning to separate imitation quality from online PPO degradation.
2. Add KL/behavior-cloning anchoring during early PPO updates so direct BC-PPO cannot drift aggressively away from the expert.
3. Strengthen safe residual gating with a suspension-margin deadband and residual scale schedule.
4. Add a hard residual safety shield that shrinks `delta_u_ppo` when current suspension or roll/pitch margins are near limits.
5. Re-run `episodes=20` and require all SI-RPPO claim report rows to be supported before treating the approach as a thesis-grade result.
