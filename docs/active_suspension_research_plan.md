# Active Suspension Research and Optimization Plan

Date: 2026-05-28

## Current Project Baseline

The repository already contains a useful experimental base for active suspension control:

- Environments: fast half-car RK4, MuJoCo half-car visualization, MuJoCo half-car dynamics, and MuJoCo full-car vertical dynamics with heave, pitch, roll, four unsprung masses, four road anchors, axle or corner actions, and axle or corner preview.
- Controllers: PID, SPDF, nonlinear MPC, Transformer-TD3, Transformer-DDPG, Transformer-SAC, Transformer-PPO.
- Road inputs: ISO random roads, bump, sine, step, pothole, table, CSV/file, and composite roads.
- Robustness features: preview delay/noise/bias/dropout/scale error, actuator first-order lag and rate limit, safety limits, domain randomization, weighted scenario sampling, imported road profiles, preflight validation, vector benchmark, and statistics collection.

Key local files:

- `envs/mujoco_full_car_env.py`: four-corner dynamics, preview error, actuator lag/rate limit, safety and reward.
- `rl/networks.py`: transformer preview encoder with mean pooling over future road tokens.
- `controllers/mpc.py`: half-car nonlinear MPC using road preview.
- `configs/mujoco_full_car_corner.yaml`: current most realistic config with four-corner control, independent road excitation, domain randomization, actuator dynamics, and preview error.

## Literature Signals

### 1. Delay-aware DRL is now central for realistic active suspension

Recent full-vehicle work embeds LSTM into TD3 for active suspension under mass uncertainty, varying driving conditions, and actuator delays. The reported motivation is direct: actuator delays create temporal misalignment between commanded and applied force, and historical state-action context improves robustness under distributed asynchronous delays.

Implication for this project:

- Current environment models actuator lag and rate limits, but the RL networks are feedforward with preview-only temporal encoding. They do not consume historical state-action sequences.
- Add recurrent or history-window variants first for TD3 and SAC, because the current `mujoco_full_car_corner.yaml` already exposes the right failure mode: four independent actuators with delay and noisy preview.

Source:

- Zhao and Yang, "Delay-Resilient Robust Control of Automobile Active Suspensions via Deep Reinforcement Learning", Results in Engineering, 2026. https://doi.org/10.1016/j.rineng.2026.109381

### 2. Expert-guided soft-hard constraints are a better fit than pure reward shaping

TD3-SH work argues that active suspension observations mix displacement, velocity, acceleration, tire load, and control force at different scales. It uses expert-guided soft rewards plus hard actuator constraints, action delay mechanisms, and practical force limitations. Reported gains are over DDPG, TD3, and MPC baselines, with sustained improvement under delay and actuator dynamics.

Implication for this project:

- The current reward already penalizes body acceleration, pitch/roll acceleration, suspension travel, tire load, and action magnitude.
- The gap is not raw terms; it is structured normalization, explicit hard constraint projection, force-velocity or bandwidth feasibility, and reporting separate objective components instead of only scalar return.
- Add metric decomposition and hard action feasibility before adding more algorithms.

Source:

- Wang et al., "Enhancing vehicle ride comfort through deep reinforcement learning with expert-guided soft-hard constraints and system characteristic considerations", Advanced Engineering Informatics, 2024. https://doi.org/10.1016/j.aei.2023.102328

### 3. Preview MPC remains a strong benchmark, especially with speed and delay modeling

Preview MPC/LPV work for speed-dependent active suspension models front/rear wheel time delays, uses adaptive Kalman filtering under sensor noise, and precomputes explicit control laws with road preview and speed to reduce online computation.

Implication for this project:

- The current MPC is half-car only and online scipy optimization only. It cannot benchmark the full-car corner environment.
- Add a practical baseline: LPV/LQR or reduced full-car preview MPC with speed-aware delay, then use it as an expert prior or residual base.
- State estimation should be a first-class component, not an assumption of perfect simulator state.

Source:

- Li et al., "Model Predictive Control for Speed-Dependent Active Suspension System with Road Preview Information", Sensors, 2024. https://doi.org/10.3390/s24072255

### 4. Sim-to-real evidence favors system identification, action delay, smoothness penalties, and actuator modeling

A real heavy-vehicle study found that policies trained with domain randomization, action delays, and reward penalties for smooth control transferred much better. Policies without smooth-action penalties showed fast switching behavior that looked good in simulation but transferred poorly.

Implication for this project:

- Current domain randomization and actuator lag are aligned with the literature.
- The action penalty should be strengthened from simple action magnitude to include action delta, jerk, saturation dwell time, and actuator energy proxy.
- Preflight should include control smoothness and actuator feasibility gates.

Source:

- Wiberg et al., "Sim-to-real transfer of active suspension control using deep reinforcement learning", arXiv:2306.11171, v3 2024. https://arxiv.org/abs/2306.11171

### 5. Residual RL and prior-guided control are a low-risk next algorithmic step

Recent residual RL work combines an LQR prior with enhanced TD3 plus LSTM/residual connections, aiming to keep baseline stability while learning adaptive residual control.

Implication for this project:

- This is more practical than training a four-corner policy from scratch.
- Use PID/SPDF or a new LQR/MPC-lite controller as the prior. The RL actor outputs bounded residual forces.
- Evaluate residual policies under actuator delay and preview dropout before considering them "better".

Source:

- "Prior-Guided Residual Reinforcement Learning for Active Suspension Control", Machines, 2025. https://www.mdpi.com/2075-1702/13/11/983

## Main Gaps Against the Literature

1. No history-aware RL policy:
   - Preview tokens encode future road, but not historical actuator-state misalignment.
   - Add history windows for state, action, and applied actuator force.

2. No full-car model-based expert baseline:
   - MPC exists only for Python half-car.
   - Full-car MuJoCo evaluation defaults to PASSIVE for classical control.

3. Reward and reporting are under-instrumented:
   - Scalar reward hides tradeoffs.
   - Need per-term metrics: RMS heave acceleration, RMS pitch/roll acceleration, suspension travel RMS/max, tire load RMS/max, action RMS, action delta RMS, saturation ratio, unsafe fraction.

4. Preview perception is synthetic but not estimated:
   - Preview error injection exists, but there is no state/road estimator.
   - Literature emphasizes Kalman filtering and road preview/state uncertainty.

5. Scenario coverage lacks explicit frequency-domain and real-road evaluation:
   - ISO, sine, bump, pothole, composite are present.
   - Need PSD/frequency metrics and imported real/precomputed road datasets as standard benchmark cases.

## Optimization Plan

### Phase 1: Measurement and Benchmark Hygiene

Priority: highest. This turns the project from "can train" into "can prove improvement".

Tasks:

- Add metric decomposition to rollout results:
  - body vertical acceleration RMS and peak
  - pitch acceleration RMS
  - roll acceleration RMS
  - suspension travel RMS and max per corner
  - tire load RMS and peak per corner
  - control force RMS
  - action delta RMS
  - actuator saturation ratio
  - unsafe fraction
- Add reward component logging in `info`, not only scalar reward.
- Extend `collect_env_statistics.py` and benchmark summaries to report these metrics.
- Add ablations:
  - clean preview vs delayed/noisy preview
  - actuator model off vs on
  - axle action vs four-corner action
  - correlated vs independent left/right road
  - nominal vs domain randomized parameters

Acceptance:

- One benchmark command produces a table where every controller/scenario has comfort, handling, safety, and actuation metrics.
- Results are comparable across PID/SPDF/MPC/RL where applicable.

### Phase 2: Add Smoothness and Feasibility Constraints

Priority: high. This directly addresses the sim-to-real failure mode.

Tasks:

- Add configurable reward terms:
  - `action_delta`
  - `action_jerk`
  - `saturation`
  - `applied_vs_commanded_error`
  - optional energy proxy: sum(abs(force * suspension_velocity))
- Add hard action projection:
  - keep current force clipping
  - expose rate-limit-aware projected command as an optional policy wrapper
  - log projected vs raw action difference
- Add safety/constraint penalties separately from comfort reward.

Acceptance:

- Policies trained with constraints reduce action delta RMS and saturation ratio without increasing unsafe fraction.
- Benchmark includes unconstrained vs constrained SAC/TD3 comparison.

### Phase 3: History-Aware TD3/SAC

Priority: high after metrics are in place.

Tasks:

- Add `HistoryBuffer` inside environments or rollout wrappers.
- Extend observation spec to include a fixed history window:
  - base state history
  - commanded action history
  - applied actuator force history
  - optional preview history
- Implement `RecurrentActorCritic` or `HistoryTransformerActor/Critic`.
- Start with TD3 because the current deterministic continuous-action stack is simpler.
- Then port to SAC for entropy-regularized robustness.

Recommended default:

- history length: 5 control steps
- include previous command and applied action
- keep preview horizon unchanged initially

Acceptance:

- Under random 10-30 ms actuator delay and preview delay/noise, history-aware TD3 beats feedforward TD3 on RMS body acceleration, roll acceleration, and unsafe fraction.
- Under burst delay tests, performance degrades gracefully rather than failing abruptly.

### Phase 4: Residual RL With a Prior Controller

Priority: medium-high. It improves stability and sample efficiency.

Tasks:

- Implement a controller composition:
  - `u = clip(u_prior + alpha * u_rl_residual)`
- Candidate priors:
  - SPDF for immediate use
  - LQR for linearized half/full-car model
  - reduced preview MPC for half-car/full-car if implemented
- Train residual TD3/SAC with smaller action limits.
- Add alpha scheduling or learned residual scale.

Acceptance:

- Residual policy starts from stable baseline behavior.
- It improves comfort metrics over the prior without violating suspension travel, roll, or saturation constraints.

### Phase 5: Full-Car Reduced MPC or LPV Baseline

Priority: medium. Needed for credible comparison and expert data.

Tasks:

- Build a reduced full-car linear model for heave, pitch, roll, and four unsprung masses.
- Add speed-dependent front/rear delay and corner preview.
- Implement an offline or fast online baseline:
  - LQR with preview feedforward as first version
  - constrained QP MPC as second version
- Use it both as:
  - a benchmark controller
  - an expert dataset source for behavior cloning or residual RL

Acceptance:

- Full-car controller works in `MuJoCoFullCarEnv`.
- It is slower than SPDF but stable enough for evaluation and expert rollout generation.

### Phase 6: State and Road Estimation

Priority: medium. Required if the target is real deployment or PreScan-like perceived road input.

Tasks:

- Add observation modes:
  - privileged simulator state
  - estimated state
  - noisy sensor-only state
- Add Kalman/EKF baseline for heave, pitch, roll, wheel states.
- Add road preview uncertainty model tied to speed and sensor range.

Acceptance:

- RL and MPC can be evaluated with estimated/noisy states.
- Metrics report estimator error alongside control metrics.

### Phase 7: Dataset and Generalization Protocol

Priority: medium.

Tasks:

- Generate fixed road train/validation/test splits.
- Include:
  - ISO A-D
  - sine sweeps by wavelength
  - bump height/length grid
  - pothole depth/length grid
  - composite mixed roads
  - imported CSV roads
- Keep test roads and seeds unseen during training.
- Add frequency-domain diagnostics for chassis acceleration and tire load.

Acceptance:

- Every paper-style result distinguishes train roads, validation roads, and unseen test roads.
- Generalization claims are based on held-out scenarios, not training scenarios.

## Recommended Execution Order

1. Implement metric decomposition and benchmark tables.
2. Add action smoothness, saturation, and applied-force logging.
3. Run baseline matrix: PASSIVE, PID, SPDF, current TD3/SAC/PPO if checkpoints exist.
4. Implement history-aware TD3.
5. Train feedforward TD3 vs history-aware TD3 on `mujoco_full_car_corner.yaml`.
6. Add residual RL with SPDF prior.
7. Add reduced full-car MPC/LQR baseline.
8. Add estimator/noisy-state evaluation.

## First Concrete Implementation Slice

The first slice should be small and verifiable:

- Add reward component dictionary to `MuJoCoFullCarEnv._info`.
- Add action delta and saturation metrics to `collect_env_statistics.py`.
- Add a benchmark summary table with comfort, handling, safety, and actuator metrics.
- Add tests that verify metric keys exist for `MuJoCoFullCarEnv`.

This slice does not change training behavior, but it creates the evidence base needed for every later optimization.

