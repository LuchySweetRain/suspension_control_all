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

Run quick checks:

```bash
python -m pytest tests
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

Evaluate classical controllers plus trained RL checkpoints:

```bash
python scripts/evaluate_all.py --config configs/default.yaml --checkpoints td3=latest,ddpg=latest,sac=latest,ppo=latest
```

Plot an existing result directory:

```bash
python scripts/plot_results.py --result-dir results/<run>
```
