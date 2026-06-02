# Active Suspension Model Parameter Audit

Date: 2026-06-02

## Summary

The current vehicle model is usable for algorithm development, but several parameters are weak for a CCF-A-level active-suspension paper:

- The full-car pitch inertia is too small for a 1200--1500 kg passenger vehicle.
- Safety limits are too loose: pitch/roll limits of `0.45 rad` and wheel displacement of `1.0 m` make unsafe behavior hard to detect.
- Tire damping is likely too high compared with common active-suspension benchmarks.
- Vehicle physical parameters were hard-coded in `HalfCarParams`, so paper experiments could not cleanly switch between a legacy model and a physically calibrated model.

This revision adds `vehicle_params` config overrides and a recommended config:

```text
configs/mujoco_full_car_physical.yaml
```

The original configs remain unchanged, so previous results are reproducible.

## Current Project Parameters

The default full-car safe PPO config currently resolves to approximately:

| Quantity | Current value | Comment |
| --- | ---: | --- |
| Sprung mass `mb` | `1200 kg` | plausible for a light vehicle body |
| Pitch inertia `Ip` | `600 kg m^2` | low for a full-car body; full-car references often use around `2160 kg m^2` |
| Roll inertia | `520 kg m^2` | plausible |
| Front/rear unsprung mass fields `mwf/mwr` | `100/100 kg` | in full-car corner mode this becomes `50 kg` per corner |
| Suspension stiffness | `15000 N/m` per corner | soft but plausible comfort-biased setting |
| Suspension damping | `1200--1500 Ns/m` | plausible |
| Tire stiffness | `200000 N/m` | common benchmark scale |
| Tire damping | `1500--2000 Ns/m` | likely high; many benchmarks use no tire damping or much lower values such as `~150--170 Ns/m` |
| Track width | `1.62 m` | plausible |
| Force limit | `5000 N` per controlled channel | plausible but aggressive |
| Actuator time constant | `0.02 s` | plausible fast active actuator |
| Actuator rate limit | `200000 N/s` | allows `2000 N` change per 10 ms control step |
| Max suspension travel | `0.25 m` | too loose for passenger-car safety evaluation |
| Max pitch/roll | `0.45 rad` | too loose; about 25.8 degrees |
| Max wheel displacement | `1.0 m` | not physically meaningful as a safety limit |

## Literature Anchors and Physical Reasoning

Several active-suspension papers and reference examples use quarter-car or full-car parameters in the following approximate ranges:

- Quarter-car active-suspension examples commonly use sprung mass around `290 kg`, unsprung mass around `59 kg`, suspension stiffness around `1.7e4 N/m`, tire stiffness around `1.9e5 N/m`, and damping around `1000 Ns/m`. See the parameter set used in the MATLAB active-suspension example and related benchmark descriptions.
- Full-car active-suspension examples commonly use total sprung mass near `1500 kg`, per-corner suspension stiffness around `3.5e4--3.8e4 N/m`, tire stiffness around `1.9e5--2.0e5 N/m`, roll inertia around `460--600 kg m^2`, and pitch inertia around `2000--3000 kg m^2`.
- Recent road-preview MPC active-suspension work uses road preview and RBF/MPC structures, reinforcing that preview-based active control should be evaluated with physically interpretable body, suspension, tire, and actuator metrics.
- Recent DRL-with-demonstrations active-suspension work motivates using expert initialization rather than unsafe online exploration from scratch.

Useful source anchors:

- Quarter-car benchmark scale: MATLAB active-suspension examples and many papers use approximately `m_s=290 kg`, `m_u=59 kg`, `k_s=16812 N/m`, `k_t=190000 N/m`, and `c_s=1000 Ns/m`.
- Full-car benchmark scale: 7-DOF full-car active-suspension parameter tables commonly use about `m_s=1500 kg`, `I_x=460 kg m^2`, `I_y=2160 kg m^2`, `m_u=59 kg` per wheel, `k_sf=35000 N/m`, `k_sr=38000 N/m`, `c_sf=1000 Ns/m`, `c_sr=1100 Ns/m`, and `k_t=190000 N/m`.
- Road-preview MPC reference: Papadimitrakis and Alexandridis, "Active vehicle suspension control using road preview model predictive control and radial basis function networks", Applied Soft Computing, 2022, DOI `10.1016/j.asoc.2022.108646`.
- Demonstration-based DRL reference: Tan et al., "Control of a nonlinear active suspension system based on deep reinforcement learning and expert demonstrations", Journal of Automobile Engineering, DOI `10.1177/09544070231191842`.

The most important physical check is natural frequency:

```text
heave frequency ~= sqrt(total_suspension_stiffness / sprung_mass) / (2*pi)
wheel-hop frequency ~= sqrt((tire_stiffness + suspension_stiffness) / unsprung_mass) / (2*pi)
```

Current full-car parameters give roughly:

- Heave: `~1.1 Hz`, comfort-biased and plausible.
- Wheel hop: `~10.4 Hz`, plausible.
- Pitch inertia: too low, making pitch acceleration overly sensitive to front/rear force imbalance.

The recommended physical config gives roughly:

- Heave: `~1.57 Hz`, still passenger-car plausible.
- Wheel hop: `~9.8--10.0 Hz`, still plausible.
- Pitch inertia: `2160 kg m^2`, closer to full-car parameter sets.

## Recommended Physical Config

The new `configs/mujoco_full_car_physical.yaml` uses:

| Quantity | Recommended value |
| --- | ---: |
| Sprung mass | `1500 kg` |
| Pitch inertia | `2160 kg m^2` |
| Roll inertia | `460 kg m^2` |
| Corner unsprung mass | `59 kg` |
| Front/rear suspension stiffness | `35000 / 38000 N/m` |
| Front/rear suspension damping | `1000 / 1100 Ns/m` |
| Tire stiffness | `190000 N/m` |
| Tire damping | `150 Ns/m` |
| Wheelbase split | `a=b=1.35 m` |
| Track width | `1.524 m` |
| Actuator time constant | `0.02 s` |
| Actuator rate limit | `100000 N/s` |
| Suspension safety travel | `0.12 m` |
| Pitch/roll safety limit | `0.12 rad` |
| Wheel displacement safety limit | `0.25 m` |

The calibrated config disables domain randomization by default so the nominal benchmark uses the stated physical parameters exactly. Robustness experiments should re-enable randomization explicitly after the nominal baseline is established.

## Expected Experimental Impact

This change will likely make the benchmark more demanding:

- Higher suspension stiffness and stricter safety limits should expose unsafe or rough active-force policies earlier.
- Larger pitch inertia should reduce unrealistic pitch acceleration sensitivity.
- Lower tire damping should make wheel-hop and tire-load metrics less artificially damped.
- A lower actuator rate limit makes action smoothing and actuator feasibility more important, which aligns with the paper's STG-PPO contribution.

## Recommended Next Experiment

Run the same short matrix on the calibrated config:

```text
python scripts/run_si_rppo_ablation.py --config configs/mujoco_full_car_physical.yaml --out results/si_rppo_e20_physical_params --episodes 20 --expert-episodes 20 --expert-controller PASSIVE --skip-unsafe-expert --baseline-algorithms td3,sac
```

Then regenerate the evidence table:

```text
python scripts/build_delta_ppo_evidence_table.py --seed-dirs results/projection_seed42_e20_improvement_gate results/projection_seed43_e20_improvement_gate results/projection_seed44_e20_improvement_gate --full-matrix-dir results/si_rppo_e20_physical_params --out results/physical_param_evidence_table
```

For the paper, report both:

- **Legacy benchmark:** preserves continuity with the current experiment log.
- **Physically calibrated benchmark:** stronger evidence for real active-suspension relevance.
