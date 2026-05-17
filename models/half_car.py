from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HalfCarParams:
    mb: float = 1200.0
    mb_max: float = 1300.0
    mb_min: float = 1100.0
    Ip: float = 600.0
    Ip_max: float = 700.0
    Ip_min: float = 500.0
    mwf: float = 100.0
    mwr: float = 100.0
    kf1: float = 15000.0
    kr1: float = 15000.0
    knf1: float = 1000.0
    knr1: float = 1000.0
    be: float = 1500.0
    bc: float = 1200.0
    delta_kf1: float = 1000.0
    delta_kr1: float = 1000.0
    delta_knf1: float = 100.0
    delta_knr1: float = 100.0
    delta_be: float = 100.0
    delta_bc: float = 100.0
    kf2: float = 200000.0
    kr2: float = 200000.0
    bf2: float = 1500.0
    br2: float = 2000.0
    a: float = 1.2
    b: float = 1.5
    delta_ab: float = 0.15
    fmax: float = 5000.0
    mb_real: float = 1200.0
    Ip_real: float = 600.0
    kf1_real: float = 15000.0
    kr1_real: float = 15000.0
    knf1_real: float = 1000.0
    knr1_real: float = 1000.0
    be_real: float = 1500.0
    bc_real: float = 1200.0
    a_real: float = 1.2
    b_real: float = 1.5

    @classmethod
    def from_seed(cls, seed: int = 42) -> "HalfCarParams":
        p = cls()
        rng = np.random.default_rng(seed)

        def bounded(nominal: float, spread: float) -> float:
            value = nominal + spread / 3.0 * rng.standard_normal()
            return float(np.clip(value, nominal - spread, nominal + spread))

        p.mb_real = bounded(p.mb, p.mb_max - p.mb)
        p.Ip_real = bounded(p.Ip, p.Ip_max - p.Ip)
        p.kf1_real = bounded(p.kf1, p.delta_kf1)
        p.kr1_real = bounded(p.kr1, p.delta_kr1)
        p.knf1_real = bounded(p.knf1, p.delta_knf1)
        p.knr1_real = bounded(p.knr1, p.delta_knr1)
        p.be_real = bounded(p.be, p.delta_be)
        p.bc_real = bounded(p.bc, p.delta_bc)
        p.a_real = bounded(p.a, p.delta_ab)
        p.b_real = p.b - (p.a_real - p.a)
        return p


class HalfCarModel:
    """Nonlinear uncertain half-car active suspension model."""

    def __init__(self, params: HalfCarParams):
        self.p = params

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=np.float64), -self.p.fmax, self.p.fmax)

    def derived(self, x: np.ndarray, road: np.ndarray, action: np.ndarray | None = None) -> dict:
        p = self.p
        x = np.asarray(x, dtype=np.float64)
        road = np.asarray(road, dtype=np.float64)
        if action is None:
            action = np.zeros(2, dtype=np.float64)
        action = self.clip_action(action)
        zb, dzb, theta, dtheta, zwf, dzwf, zwr, dzwr = x
        zdf, dzdf, zdr, dzdr = road
        uaf, uar = action

        delta_yf = zb + p.a_real * np.sin(theta) - zwf
        delta_dyf = dzb + p.a_real * dtheta * np.cos(theta) - dzwf
        delta_yr = zb - p.b_real * np.sin(theta) - zwr
        delta_dyr = dzb - p.b_real * dtheta * np.cos(theta) - dzwr

        fsf = p.kf1_real * delta_yf + p.knf1_real * delta_yf**3
        fsr = p.kr1_real * delta_yr + p.knr1_real * delta_yr**3
        fdf = 0.5 * ((p.be_real + p.bc_real) + (p.be_real - p.bc_real) * np.sign(delta_dyf)) * delta_dyf
        fdr = 0.5 * ((p.be_real + p.bc_real) + (p.be_real - p.bc_real) * np.sign(delta_dyr)) * delta_dyr
        ftf = p.kf2 * (zwf - zdf)
        ftr = p.kr2 * (zwr - zdr)
        fbf = p.bf2 * (dzwf - dzdf)
        fbr = p.br2 * (dzwr - dzdr)

        ub = uaf + uar
        utheta = p.a_real * uaf - p.b_real * uar
        ddzb = -(fsf + fdf + fsr + fdr) / p.mb_real + ub / p.mb_real
        ddtheta = (-(p.a_real * (fsf + fdf)) + p.b_real * (fsr + fdr)) / p.Ip_real + utheta / p.Ip_real
        ddzwf = (fsf + fdf - ftf - fbf) / p.mwf - uaf / p.mwf
        ddzwr = (fsr + fdr - ftr - fbr) / p.mwr - uar / p.mwr

        return {
            "delta_yf": float(delta_yf),
            "delta_dyf": float(delta_dyf),
            "delta_yr": float(delta_yr),
            "delta_dyr": float(delta_dyr),
            "Fpf": float(ftf + fbf),
            "Fpr": float(ftr + fbr),
            "ddzb": float(ddzb),
            "ddtheta": float(ddtheta),
            "ddzwf": float(ddzwf),
            "ddzwr": float(ddzwr),
        }

    def derivative(self, x: np.ndarray, action: np.ndarray, road: np.ndarray) -> np.ndarray:
        d = self.derived(x, road, action)
        return np.asarray(
            [x[1], d["ddzb"], x[3], d["ddtheta"], x[5], d["ddzwf"], x[7], d["ddzwr"]],
            dtype=np.float64,
        )

    def rk4_step(self, x: np.ndarray, action: np.ndarray, road: np.ndarray, dt: float) -> np.ndarray:
        action = self.clip_action(action)
        k1 = self.derivative(x, action, road)
        k2 = self.derivative(x + 0.5 * dt * k1, action, road)
        k3 = self.derivative(x + 0.5 * dt * k2, action, road)
        k4 = self.derivative(x + dt * k3, action, road)
        return np.asarray(x, dtype=np.float64) + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

