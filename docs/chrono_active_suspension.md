# Project Chrono Active-Suspension Smoke Environment

This repository now includes a minimal quarter-car active-suspension benchmark with an optional Project Chrono backend:

```powershell
python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_smoke --backend rk4
```

The same script is prepared for PyChrono:

```powershell
conda install projectchrono::pychrono -c conda-forge
python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_chrono --backend chrono
```

On the current Windows base environment, `pychrono` was available on conda-forge but installation did not finish within the local 20 minute command budget. The `rk4` backend therefore serves as a deterministic smoke backend using the same quarter-car force equations and controller interface. Once PyChrono is installed, the `chrono` backend uses `ChSystemNSC` bodies and `ChLinkTSDA` suspension/tire force elements with `SetActuatorForce` for the active suspension input.

## What This Adds

- A compact external-physics validation path for active suspension.
- Passive vs active skyhook/groundhook comparison under a smooth bump road.
- Output CSV trajectories and a metrics manifest with body acceleration RMS, suspension deflection, actuator force, action delta, and unsafe steps.

This is not yet a full-car Chrono::Vehicle benchmark. It is a deliberately small bridge so the project can later validate the MuJoCo full-car PPO controller against Chrono vehicle dynamics.

## Local Smoke Result

Command:

```powershell
python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_smoke --backend rk4
```

Result on the deterministic RK4 backend:

| Controller | BodyAccRMS_mps2 | BodyDispRMS_m | MaxSuspensionDeflection_m | ActiveForceRMS_N | ActionDeltaRMS_N | UnsafeSteps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 1.5615 | 0.0194 | 0.0407 | 0.0000 | 0.0000 | 0 |
| active | 0.7235 | 0.0096 | 0.0286 | 201.7319 | 2.1084 | 0 |

The active skyhook/groundhook controller reduces body acceleration RMS by `0.8381 m/s^2`, body displacement RMS by `0.0098 m`, and maximum suspension deflection by `0.0121 m` in the smooth-bump smoke case.
