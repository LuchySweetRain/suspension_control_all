# suspension_control_all

Pure Python comparison framework for half-car active suspension control.

Controllers:

- PID without road preview
- SPDF without road preview
- Nonlinear MPC with road preview
- Transformer-TD3 with road preview

Run quick checks:

```bash
python -m pytest tests
```

Train a small Transformer-TD3 run:

```bash
python scripts/train_td3.py --config configs/train_fast.yaml
```

Evaluate all controllers:

```bash
python scripts/evaluate_all.py --config configs/default.yaml --checkpoint latest
```

Plot an existing result directory:

```bash
python scripts/plot_results.py --result-dir results/<run>
```
