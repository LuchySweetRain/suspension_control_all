# suspension_control_all

Pure Python comparison framework for half-car active suspension control.

Controllers:

- PID without road preview
- SPDF without road preview
- Nonlinear MPC with road preview
- Transformer-TD3 with road preview
- Transformer-DDPG with road preview
- Transformer-SAC with road preview
- Transformer-PPO with road preview

Research direction:

- The current thesis-oriented algorithm plan focuses on offline imitation initialization followed by online safe residual PPO for full-car active suspension control. See `docs/offline_imitation_online_ppo_plan.md`.
- The SI-RPPO ablation entry point is `scripts/run_si_rppo_ablation.py`, covering PPO from scratch, BC-PPO, residual BC-PPO, and safe residual BC-PPO.
- Current SI-RPPO experiment results and interpretation are logged in `docs/si_rppo_experiment_log.md`.

Environments:

- `HalfCarEnv`: fast pure-Python RK4 half-car environment for controller comparison.
- `MuJoCoHalfCarEnv`: MuJoCo rendering wrapper around `HalfCarEnv`; dynamics still come from the Python model.
- `MuJoCoVehicleEnv`: MuJoCo dynamics environment for RL. It models chassis heave/pitch, front/rear unsprung masses, passive suspension/tire spring-damper tendons, active suspension force actuators, moving road anchors, and road preview observations.
- `MuJoCoFullCarEnv`: MuJoCo full-car vertical dynamics environment. It adds chassis roll, four unsprung masses, four road anchors, correlated/independent left-right road excitation, axle-level or four-corner active control, and optional preview/perception error injection.

Road profiles:

- ISO 8608 random roads
- bump roads
- sine roads
- step roads
- potholes
- table-driven external road data
- CSV/text road files with `time,height` columns
- composite roads built from multiple components

The MuJoCo full-car corner config can train from already perceived road height profiles instead of extracting road contours inside PreScan. `road_preview` is the delayed/noisy preview seen by the controller, while `road_preview_clean` is retained in `info` for ablation and diagnostics. RL actions are treated as actuator force commands; the full-car environment can apply configurable first-order actuator lag and rate limits before the force reaches MuJoCo.

Full-car safety constraints are configured with `safety_limits`, including suspension travel, pitch, roll, wheel displacement, unsafe penalty, and optional episode termination on unsafe states. Validation reports include per-scenario safety violation counts.

Run quick checks:

```bash
python -m pip install -r requirements.txt
python -m pytest tests
```

Generate the SI-RPPO ablation configs without launching long training:

```bash
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --episodes 200 --expert-episodes 20 --dry-run
```

The runner writes `combined_metrics.csv`, `si_rppo_claim_report.json`, and `si_rppo_claim_report.md` under the selected result directory.

Include off-policy TD3/SAC baselines in the same evidence report:

```bash
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --episodes 200 --expert-episodes 20 --baseline-algorithms td3,sac
```

Run a fast SI-RPPO smoke ablation on one short scenario:

```bash
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_safe_ppo.yaml --episodes 1 --expert-episodes 1 --expert-max-steps 2 --train-scenario-limit 1 --eval-scenario-limit 1 --episode-seconds 0.05 --mujoco-settle-seconds 0.2 --variants safe_residual_bc_ppo
```

Train a small Transformer-TD3 run:

```bash
python scripts/train_rl.py --algorithm td3 --config configs/train_fast.yaml
```

Train the other RL baselines:

```bash
python scripts/train_rl.py --algorithm ddpg --config configs/train_fast.yaml
python scripts/train_rl.py --algorithm sac --config configs/train_fast.yaml
python scripts/train_rl.py --algorithm ppo --config configs/train_fast.yaml
```

Train directly in the MuJoCo dynamics environment:

```bash
python scripts/train_rl.py --algorithm sac --config configs/mujoco_vehicle.yaml
python scripts/train_rl.py --algorithm td3 --config configs/mujoco_vehicle.yaml
python scripts/train_rl.py --algorithm sac --config configs/mujoco_full_car.yaml
python scripts/train_rl.py --algorithm sac --config configs/mujoco_full_car_corner.yaml
```

The environments can also be registered with Gymnasium for external RL libraries:

```bash
python scripts/check_gym_registration.py --config configs/mujoco_full_car_corner.yaml --smoke
```

Training scenario selection is configured with `scenario_sampling`. Supported modes are `cycle`, `uniform`, and `weighted`; curriculum phases can switch the active road subset by episode. Each run writes `training_manifest.json` with the sampler settings, RL dimensions, algorithm, and engine.

Evaluate classical controllers plus trained RL checkpoints:

```bash
python scripts/evaluate_all.py --config configs/default.yaml --checkpoints td3=latest,ddpg=latest,sac=latest,ppo=latest
```

Generate a reusable road dataset and evaluate the MuJoCo full-car environment:

```bash
python scripts/generate_road_dataset.py --out datasets/mujoco_roads --duration 8
python scripts/evaluate_all.py --config datasets/mujoco_roads/mujoco_full_car_dataset.yaml --out results/mujoco_dataset_eval
python scripts/summarize_benchmark.py --result-dir results/mujoco_dataset_eval
```

Import externally generated or perceived road profiles from a directory of `time,height` CSV files:

```bash
python scripts/import_road_directory.py --road-dir path/to/road_csvs --out datasets/imported_roads --speed 20
python scripts/evaluate_all.py --config datasets/imported_roads/mujoco_full_car_imported_roads.yaml --out results/imported_road_eval
```

Validate a MuJoCo training configuration before a long RL run:

```bash
python scripts/validate_training_config.py --config configs/mujoco_full_car_corner.yaml --out results/mujoco_config_validation.json
python scripts/export_mujoco_env_spec.py --config configs/mujoco_full_car_corner.yaml --out results/mujoco_env_spec.json
python scripts/validate_mujoco_env.py --config configs/mujoco_full_car_corner.yaml --max-steps 200 --action-mode random --out results/mujoco_env_validation.json
```

Run a robustness validation matrix across nominal, random-action, sensor/actuator-error, and domain-randomized cases:

```bash
python scripts/run_mujoco_robustness_matrix.py --config configs/mujoco_full_car_corner.yaml --max-steps 200 --out results/mujoco_robustness_matrix
```

Benchmark multiple environment instances before large-scale RL collection:

```bash
python scripts/benchmark_vector_env.py --config configs/mujoco_full_car_corner.yaml --num-envs 4 --steps 200 --action-mode random --out results/mujoco_vector_env_benchmark.json
```

Collect observation/action/reward statistics for normalization and reward-scale diagnostics:

```bash
python scripts/collect_env_statistics.py --config configs/mujoco_full_car_corner.yaml --episodes 8 --max-steps 200 --action-mode random --out results/mujoco_env_statistics.json
```

Run the full preflight gate before a long training job:

```bash
python scripts/preflight_mujoco_training.py --config configs/mujoco_full_car_corner.yaml --out results/mujoco_preflight --validation-steps 200 --vector-envs 4 --vector-steps 100 --statistics-episodes 8 --statistics-steps 200
```

Run the full benchmark pipeline from one command:

```bash
python scripts/run_mujoco_benchmark.py --config configs/mujoco_full_car_corner.yaml --generate-roads --out results/mujoco_benchmark
python scripts/run_mujoco_benchmark.py --config configs/mujoco_full_car_corner.yaml --algorithms sac --episodes 20 --out results/mujoco_sac_benchmark
```

Plot an existing result directory:

```bash
python scripts/plot_results.py --result-dir results/<run>
```
