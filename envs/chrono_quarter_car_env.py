"""Quarter-car active-suspension environment with optional Project Chrono backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BackendName = Literal["chrono", "rk4"]


@dataclass
class QuarterCarParams:
    sprung_mass_kg: float = 375.0
    unsprung_mass_kg: float = 59.0
    suspension_stiffness_npm: float = 35_000.0
    suspension_damping_nspm: float = 1_000.0
    tire_stiffness_npm: float = 190_000.0
    tire_damping_nspm: float = 100.0
    force_limit_n: float = 3_500.0
    force_rate_limit_nps: float = 20_000.0


def sinusoidal_bump(t: float, start: float = 0.6, duration: float = 0.35, height: float = 0.04) -> tuple[float, float]:
    """Smooth bump road profile and its velocity."""
    if t < start or t > start + duration:
        return 0.0, 0.0
    phase = (t - start) / duration
    road = 0.5 * height * (1.0 - np.cos(2.0 * np.pi * phase))
    road_dot = height * np.pi / duration * np.sin(2.0 * np.pi * phase)
    return float(road), float(road_dot)


class QuarterCarActiveController:
    """Simple skyhook-plus-groundhook active controller with actuator limits."""

    def __init__(self, params: QuarterCarParams, skyhook_gain: float = -3_000.0, groundhook_gain: float = 300.0):
        self.params = params
        self.skyhook_gain = float(skyhook_gain)
        self.groundhook_gain = float(groundhook_gain)
        self.force = 0.0

    def reset(self) -> None:
        self.force = 0.0

    def __call__(self, state: np.ndarray, dt: float) -> float:
        z_s, z_s_dot, z_u, z_u_dot = state
        raw = -self.skyhook_gain * z_s_dot + self.groundhook_gain * z_u_dot
        raw = float(np.clip(raw, -self.params.force_limit_n, self.params.force_limit_n))
        max_delta = self.params.force_rate_limit_nps * dt
        self.force += float(np.clip(raw - self.force, -max_delta, max_delta))
        return self.force


class QuarterCarRK4Backend:
    """Deterministic RK4 backend matching the Chrono quarter-car force model."""

    def __init__(self, params: QuarterCarParams, dt: float):
        self.params = params
        self.dt = float(dt)
        self.state = np.zeros(4, dtype=np.float64)

    def reset(self) -> np.ndarray:
        self.state[:] = 0.0
        return self.state.copy()

    def _deriv(self, state: np.ndarray, active_force: float, road: float, road_dot: float) -> np.ndarray:
        p = self.params
        z_s, z_s_dot, z_u, z_u_dot = state
        susp_force = p.suspension_stiffness_npm * (z_s - z_u) + p.suspension_damping_nspm * (z_s_dot - z_u_dot)
        tire_force = p.tire_stiffness_npm * (z_u - road) + p.tire_damping_nspm * (z_u_dot - road_dot)
        z_s_ddot = (-susp_force - active_force) / p.sprung_mass_kg
        z_u_ddot = (susp_force + active_force - tire_force) / p.unsprung_mass_kg
        return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot], dtype=np.float64)

    def step(self, active_force: float, road: float, road_dot: float) -> np.ndarray:
        h = self.dt
        y = self.state
        k1 = self._deriv(y, active_force, road, road_dot)
        k2 = self._deriv(y + 0.5 * h * k1, active_force, road, road_dot)
        k3 = self._deriv(y + 0.5 * h * k2, active_force, road, road_dot)
        k4 = self._deriv(y + h * k3, active_force, road, road_dot)
        self.state = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.state.copy()


class QuarterCarChronoBackend:
    """Project Chrono backend.

    This backend is intentionally small and mirrors the RK4 state definition.
    It requires the conda-forge `pychrono` package.
    """

    def __init__(self, params: QuarterCarParams, dt: float):
        try:
            import pychrono as chrono
        except ImportError as exc:
            raise ImportError(
                "Project Chrono Python bindings are not installed. "
                "Install with: conda install projectchrono::pychrono -c conda-forge"
            ) from exc
        self.chrono = chrono
        self.params = params
        self.dt = float(dt)
        self._build_system()

    def _build_system(self) -> None:
        chrono = self.chrono
        p = self.params
        self.z_s_eq = 0.6
        self.z_u_eq = 0.15
        self.suspension_free_length = self.z_s_eq - self.z_u_eq
        self.tire_free_length = self.z_u_eq
        self.system = chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))

        self.ground = chrono.ChBody()
        self.ground.SetFixed(True)
        self.system.AddBody(self.ground)

        self.sprung = chrono.ChBody()
        self.sprung.SetMass(p.sprung_mass_kg)
        self.sprung.SetPos(chrono.ChVector3d(0.0, 0.0, self.z_s_eq))
        self.system.AddBody(self.sprung)

        self.unsprung = chrono.ChBody()
        self.unsprung.SetMass(p.unsprung_mass_kg)
        self.unsprung.SetPos(chrono.ChVector3d(0.0, 0.0, self.z_u_eq))
        self.system.AddBody(self.unsprung)

        self.suspension = chrono.ChLinkTSDA()
        self.suspension.Initialize(
            self.sprung,
            self.unsprung,
            False,
            chrono.ChVector3d(0.0, 0.0, self.z_s_eq),
            chrono.ChVector3d(0.0, 0.0, self.z_u_eq),
        )
        self.suspension.SetSpringCoefficient(p.suspension_stiffness_npm)
        self.suspension.SetDampingCoefficient(p.suspension_damping_nspm)
        # TSDA compression force is created when current length is below rest length.
        # Set static preload so the nominal initial pose balances gravity.
        suspension_static_compression = p.sprung_mass_kg * 9.81 / p.suspension_stiffness_npm
        self.suspension.SetRestLength(self.suspension_free_length + suspension_static_compression)
        self.system.AddLink(self.suspension)

        self.tire = chrono.ChLinkTSDA()
        self.tire.Initialize(
            self.unsprung,
            self.ground,
            False,
            chrono.ChVector3d(0.0, 0.0, self.z_u_eq),
            chrono.ChVector3d(0.0, 0.0, 0.0),
        )
        self.tire.SetSpringCoefficient(p.tire_stiffness_npm)
        self.tire.SetDampingCoefficient(p.tire_damping_nspm)
        tire_static_compression = (p.sprung_mass_kg + p.unsprung_mass_kg) * 9.81 / p.tire_stiffness_npm
        self.tire.SetRestLength(self.tire_free_length + tire_static_compression)
        self.system.AddLink(self.tire)

    def reset(self) -> np.ndarray:
        self._build_system()
        return self._state(0.0)

    def _state(self, road: float) -> np.ndarray:
        z_s = self.sprung.GetPos().z - self.z_s_eq
        z_u = self.unsprung.GetPos().z - self.z_u_eq
        return np.array(
            [
                z_s,
                self.sprung.GetPosDt().z,
                z_u,
                self.unsprung.GetPosDt().z,
            ],
            dtype=np.float64,
        )

    def step(self, active_force: float, road: float, road_dot: float) -> np.ndarray:
        # Chrono TSDA actuator force is positive in tension along the link.
        # The project convention treats positive active_force as the RK4 force
        # added to the unsprung-mass equation and subtracted from the sprung
        # equation, so the Chrono command must be sign-flipped.
        self.suspension.SetActuatorForce(float(-active_force))
        self.ground.SetPos(self.chrono.ChVector3d(0.0, 0.0, float(road)))
        self.system.DoStepDynamics(self.dt)
        return self._state(road)


class ChronoQuarterCarEnv:
    """Minimal active-suspension rollout wrapper for Chrono or RK4 backends."""

    def __init__(self, params: QuarterCarParams | None = None, dt: float = 0.001, backend: BackendName = "rk4"):
        self.params = params or QuarterCarParams()
        self.dt = float(dt)
        self.backend_name = backend
        self.backend = QuarterCarChronoBackend(self.params, self.dt) if backend == "chrono" else QuarterCarRK4Backend(self.params, self.dt)
        self.controller = QuarterCarActiveController(self.params)

    def reset(self) -> np.ndarray:
        self.controller.reset()
        return self.backend.reset()

    def rollout(self, duration: float = 2.0, controller: Literal["passive", "active"] = "passive") -> dict:
        steps = int(round(duration / self.dt))
        state = self.reset()
        rows = []
        prev_force = 0.0
        for step in range(steps):
            t = step * self.dt
            road, road_dot = sinusoidal_bump(t)
            active_force = 0.0 if controller == "passive" else self.controller(state, self.dt)
            state = self.backend.step(active_force, road, road_dot)
            z_s, z_s_dot, z_u, z_u_dot = state
            susp_def = z_s - z_u
            force_delta = active_force - prev_force
            prev_force = active_force
            rows.append(
                {
                    "time_s": t,
                    "road_m": road,
                    "body_z_m": z_s,
                    "body_v_mps": z_s_dot,
                    "wheel_z_m": z_u,
                    "wheel_v_mps": z_u_dot,
                    "suspension_deflection_m": susp_def,
                    "active_force_n": active_force,
                    "active_force_delta_n": force_delta,
                }
            )
        return {"backend": self.backend_name, "controller": controller, "rows": rows, "metrics": self._metrics(rows)}

    def _metrics(self, rows: list[dict]) -> dict:
        body_v = np.asarray([r["body_v_mps"] for r in rows], dtype=np.float64)
        body_z = np.asarray([r["body_z_m"] for r in rows], dtype=np.float64)
        susp = np.asarray([r["suspension_deflection_m"] for r in rows], dtype=np.float64)
        force = np.asarray([r["active_force_n"] for r in rows], dtype=np.float64)
        force_delta = np.asarray([r["active_force_delta_n"] for r in rows], dtype=np.float64)
        body_acc = np.gradient(body_v, self.dt)
        unsafe = np.abs(susp) > 0.12
        return {
            "BodyAccRMS_mps2": float(np.sqrt(np.mean(body_acc**2))),
            "BodyDispRMS_m": float(np.sqrt(np.mean(body_z**2))),
            "MaxSuspensionDeflection_m": float(np.max(np.abs(susp))),
            "ActiveForceRMS_N": float(np.sqrt(np.mean(force**2))),
            "ActionDeltaRMS_N": float(np.sqrt(np.mean(force_delta**2))),
            "UnsafeSteps": int(np.sum(unsafe)),
        }
