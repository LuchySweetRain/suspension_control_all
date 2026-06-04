# Project Chrono Active-Suspension Smoke Environment

This repository now includes a minimal quarter-car active-suspension benchmark with an optional Project Chrono backend:

```powershell
python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_smoke --backend rk4
```

The same script can run the Project Chrono backend:

```powershell
conda create -y -n chrono-suspension --override-channels -c conda-forge python=3.12 pychrono=10.0.0 numpy pyyaml pandas
conda run -n chrono-suspension python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_chrono --backend chrono
```

On the current Windows machine, `pychrono=10.0.0` is installed in the dedicated `chrono-suspension` conda environment. The `chrono` backend uses `ChSystemNSC` bodies and `ChLinkTSDA` suspension/tire force elements with `SetActuatorForce` for the active suspension input. The `rk4` backend remains as a deterministic fallback using the same quarter-car force equations and controller interface.

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

Project Chrono backend command:

```powershell
conda run -n chrono-suspension python scripts/run_chrono_active_suspension.py --config configs/chrono_quarter_car.yaml --out results/chrono_quarter_car_chrono --backend chrono
```

Result:

| Controller | BodyAccRMS_mps2 | BodyDispRMS_m | MaxSuspensionDeflection_m | ActiveForceRMS_N | ActionDeltaRMS_N | UnsafeSteps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| passive | 1.5647 | 0.0194 | 0.0408 | 0.0000 | 0.0000 | 0 |
| active | 0.7274 | 0.0096 | 0.0286 | 202.2941 | 2.1209 | 0 |

The Chrono backend matches the RK4 smoke result closely and confirms that the active controller reduces body acceleration, body displacement, and suspension deflection under the same smooth-bump road profile.
