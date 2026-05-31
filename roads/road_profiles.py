from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BaseRoadProfile:
    def value(self, t: float) -> float:
        raise NotImplementedError

    def derivative(self, t: float) -> float:
        eps = 1e-3
        return (self.value(t + eps) - self.value(max(0.0, t - eps))) / (2 * eps)

    def pair(self, t: float, speed: float, wheelbase: float) -> tuple[float, float, float, float]:
        delay = wheelbase / max(speed, 1e-6)
        zf = self.value(t)
        dzf = self.derivative(t)
        tr = max(0.0, t - delay)
        zr = self.value(tr)
        dzr = self.derivative(tr)
        return zf, dzf, zr, dzr

    def preview(self, t: float, speed: float, wheelbase: float, steps: int, dt: float) -> np.ndarray:
        rows = []
        for k in range(1, steps + 1):
            tk = t + k * dt
            zf, _, zr, _ = self.pair(tk, speed, wheelbase)
            rows.append([zf, zr])
        return np.asarray(rows, dtype=np.float64)


@dataclass
class IsoRoadProfile(BaseRoadProfile):
    level: str = "B"
    duration: float = 20.0
    dt: float = 0.001
    speed: float = 20.0
    seed: int = 42

    def __post_init__(self):
        gdn0_table = {"A": 32e-6, "B": 128e-6, "C": 512e-6, "D": 2048e-6}
        gdn0 = gdn0_table[self.level.upper()]
        rng = np.random.default_rng(self.seed)
        t = np.arange(0.0, self.duration + self.dt, self.dt)
        n0 = 0.1
        n1 = 0.01
        a = -2.0 * np.pi * n1 * self.speed
        b = 2.0 * np.pi * n0 * np.sqrt(gdn0 * self.speed)
        w = rng.standard_normal(t.shape[0])
        z = np.zeros_like(t)
        for i in range(1, len(t)):
            z[i] = z[i - 1] + (a * z[i - 1] + b * w[i - 1]) * self.dt
        z -= z[0]
        self._t = t
        self._z = z
        self._dz = np.gradient(z, self.dt)

    def value(self, t: float) -> float:
        return float(np.interp(t, self._t, self._z, left=self._z[0], right=self._z[-1]))

    def derivative(self, t: float) -> float:
        return float(np.interp(t, self._t, self._dz, left=0.0, right=0.0))


@dataclass
class BumpRoadProfile(BaseRoadProfile):
    speed: float = 20.0
    bump_length: float = 0.25
    bump_height: float = 0.1
    start_time: float = 0.6

    def value(self, t: float) -> float:
        duration = self.bump_length / max(self.speed, 1e-6)
        if self.start_time <= t <= self.start_time + duration:
            omega = 2.0 * np.pi * self.speed / self.bump_length
            return float(0.5 * self.bump_height * (1.0 - np.cos(omega * (t - self.start_time))))
        return 0.0

    def derivative(self, t: float) -> float:
        duration = self.bump_length / max(self.speed, 1e-6)
        if self.start_time <= t <= self.start_time + duration:
            omega = 2.0 * np.pi * self.speed / self.bump_length
            return float(0.5 * self.bump_height * omega * np.sin(omega * (t - self.start_time)))
        return 0.0


@dataclass
class SineRoadProfile(BaseRoadProfile):
    speed: float = 20.0
    amplitude: float = 0.03
    wavelength: float = 4.0
    phase: float = 0.0

    def value(self, t: float) -> float:
        omega = 2.0 * np.pi * self.speed / max(self.wavelength, 1e-6)
        return float(self.amplitude * np.sin(omega * t + self.phase))

    def derivative(self, t: float) -> float:
        omega = 2.0 * np.pi * self.speed / max(self.wavelength, 1e-6)
        return float(self.amplitude * omega * np.cos(omega * t + self.phase))


@dataclass
class StepRoadProfile(BaseRoadProfile):
    height: float = 0.05
    start_time: float = 0.6

    def value(self, t: float) -> float:
        return float(self.height if t >= self.start_time else 0.0)

    def derivative(self, t: float) -> float:
        return 0.0


@dataclass
class PotholeRoadProfile(BaseRoadProfile):
    speed: float = 20.0
    length: float = 0.5
    depth: float = 0.06
    start_time: float = 0.6

    def value(self, t: float) -> float:
        duration = self.length / max(self.speed, 1e-6)
        if self.start_time <= t <= self.start_time + duration:
            omega = 2.0 * np.pi * self.speed / self.length
            return float(-0.5 * self.depth * (1.0 - np.cos(omega * (t - self.start_time))))
        return 0.0

    def derivative(self, t: float) -> float:
        duration = self.length / max(self.speed, 1e-6)
        if self.start_time <= t <= self.start_time + duration:
            omega = 2.0 * np.pi * self.speed / self.length
            return float(-0.5 * self.depth * omega * np.sin(omega * (t - self.start_time)))
        return 0.0


@dataclass
class TableRoadProfile(BaseRoadProfile):
    times: np.ndarray
    heights: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.heights = np.asarray(self.heights, dtype=np.float64)
        if self.times.ndim != 1 or self.heights.ndim != 1 or self.times.shape != self.heights.shape:
            raise ValueError("times and heights must be one-dimensional arrays with the same shape")
        if self.times.size < 2:
            raise ValueError("table road profiles require at least two samples")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("table road times must be strictly increasing")
        self._dz = np.gradient(self.heights, self.times)

    def value(self, t: float) -> float:
        return float(np.interp(t, self.times, self.heights, left=self.heights[0], right=self.heights[-1]))

    def derivative(self, t: float) -> float:
        return float(np.interp(t, self.times, self._dz, left=0.0, right=0.0))


@dataclass
class CompositeRoadProfile(BaseRoadProfile):
    profiles: list[BaseRoadProfile]

    def value(self, t: float) -> float:
        return float(sum(profile.value(t) for profile in self.profiles))

    def derivative(self, t: float) -> float:
        return float(sum(profile.derivative(t) for profile in self.profiles))


class RoadProfileFactory:
    @staticmethod
    def create(scenario: dict, duration: float, dt: float) -> BaseRoadProfile:
        road_type = scenario.get("type", "iso").lower()
        speed = float(scenario.get("speed", 20.0))
        seed = int(scenario.get("seed", 42))
        if road_type == "bump":
            return BumpRoadProfile(
                speed=speed,
                bump_length=float(scenario.get("bump_length", scenario.get("length", 0.25))),
                bump_height=float(scenario.get("bump_height", scenario.get("height", 0.1))),
                start_time=float(scenario.get("start_time", 0.6)),
            )
        if road_type == "sine":
            return SineRoadProfile(
                speed=speed,
                amplitude=float(scenario.get("amplitude", 0.03)),
                wavelength=float(scenario.get("wavelength", 4.0)),
                phase=float(scenario.get("phase", 0.0)),
            )
        if road_type == "step":
            return StepRoadProfile(
                height=float(scenario.get("height", 0.05)),
                start_time=float(scenario.get("start_time", 0.6)),
            )
        if road_type == "pothole":
            return PotholeRoadProfile(
                speed=speed,
                length=float(scenario.get("length", 0.5)),
                depth=float(scenario.get("depth", 0.06)),
                start_time=float(scenario.get("start_time", 0.6)),
            )
        if road_type == "table":
            return TableRoadProfile(times=np.asarray(scenario["times"]), heights=np.asarray(scenario["heights"]))
        if road_type in {"csv", "file"}:
            path = Path(str(scenario["path"]))
            delimiter = str(scenario.get("delimiter", ","))
            skiprows = int(scenario.get("skiprows", 0))
            try:
                data = np.loadtxt(path, delimiter=delimiter, comments="#", skiprows=skiprows)
            except ValueError:
                data = np.loadtxt(path, delimiter=delimiter, comments="#", skiprows=max(1, skiprows))
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("CSV road profiles require at least two columns: time,height")
            scale = float(scenario.get("scale", 1.0))
            return TableRoadProfile(times=data[:, 0], heights=data[:, 1] * scale)
        if road_type == "composite":
            profiles = [
                RoadProfileFactory.create({**part, "speed": part.get("speed", speed)}, duration, dt)
                for part in scenario.get("components", [])
            ]
            if not profiles:
                raise ValueError("composite road profiles require at least one component")
            return CompositeRoadProfile(profiles)
        return IsoRoadProfile(
            level=str(scenario.get("level", "B")),
            duration=duration + 5.0,
            dt=dt,
            speed=speed,
            seed=seed,
        )
