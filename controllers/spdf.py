from __future__ import annotations

from collections import deque

import numpy as np

from .base import Controller, split_generalized_force
from models.half_car import HalfCarParams


class SPDFController(Controller):
    name = "SPDF"

    def __init__(self, params: HalfCarParams, dt: float, period: float = 1.0):
        self.p = params
        self.dt = dt
        self.period = period
        self.h = period / 4.0
        self.delay_steps = max(1, int(round(self.h / dt)))
        self.a1 = 40.0
        self.a2 = 40.0
        self.tau1 = 0.75
        self.tau2 = 0.75
        self.eps = 1e-3
        self.reset()

    def reset(self):
        self.time = 0.0
        self.history = deque(maxlen=2 * self.delay_steps + 2)
        for _ in range(2 * self.delay_steps + 2):
            # Simulink delays the mux [s1, s2, x1, x2, x3, x4] by T/4.
            self.history.append(np.zeros(6, dtype=np.float64))
        self.prev_k1 = 0.0
        self.prev_k2 = 0.0

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self.p
        x = np.asarray(info["state"], dtype=np.float64)
        d = info["derived"]
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        delayed = self.history[-self.delay_steps]
        s1_h, s2_h, x1_h, x2_h, x3_h, x4_h = delayed
        t = self.time % self.period

        k1 = self._k_a(t, self.a1)
        k2 = self._k_a(t, self.a2)
        dk1 = (k1 - self.prev_k1) / self.dt
        dk2 = (k2 - self.prev_k2) / self.dt
        self.prev_k1 = k1
        self.prev_k2 = k2

        s1 = x2 + self.a1 * x1 + k1 * x1_h
        s2 = x4 + self.a2 * x3 + k2 * x3_h

        ro1 = self._ro1(d)
        ro2 = self._ro2(d)
        f1 = self._f1(d)
        f2 = self._f2(d)
        delta_g1 = (p.mb_max - p.mb) / p.mb_min
        delta_g2 = (p.Ip_max - p.Ip) / p.Ip_min

        ub0 = -self.a1 / (2 * (1 - self.tau1)) * s1 - self._k_ah(t, self.a1) / (2 * (1 - self.tau1)) * np.sign(s1) * abs(s1) ** (2 * self.tau1 - 1) * abs(s1_h) ** (2 - 2 * self.tau1)
        gamma1 = 1 / (1 - delta_g1) * delta_g1 * abs(-2 / np.pi * ro1 * np.arctan(ro1 * s1 / self.eps) - self.a1 * x2 - dk1 * x1_h - k1 * x2_h + ub0)
        vb0 = -2 / np.pi * gamma1 * np.arctan(gamma1 * s1 / self.eps)
        vb = p.mb * (-2 / np.pi * ro1 * np.arctan(ro1 * s1 / self.eps) - self.a1 * x2 - dk1 * x1_h - k1 * x2_h + ub0 + vb0)
        ub = vb + f1

        utheta0 = -self.a2 / (2 * (1 - self.tau2)) * s2 - self._k_ah(t, self.a2) / (2 * (1 - self.tau2)) * np.sign(s2) * abs(s2) ** (2 * self.tau2 - 1) * abs(s2_h) ** (2 - 2 * self.tau2)
        gamma2 = 1 / (1 - delta_g2) * delta_g2 * abs(-2 / np.pi * ro2 * np.arctan(ro2 * s2 / self.eps) - self.a2 * x4 - dk2 * x3_h - k2 * x4_h + utheta0)
        vtheta0 = -2 / np.pi * gamma2 * np.arctan(gamma2 * s2 / self.eps)
        vtheta = p.Ip * (-2 / np.pi * ro2 * np.arctan(ro2 * s2 / self.eps) - self.a2 * x4 - dk2 * x3_h - k2 * x4_h + utheta0 + vtheta0)
        utheta = vtheta + f2

        self.history.append(np.asarray([s1, s2, x1, x2, x3, x4], dtype=np.float64))
        self.time += self.dt
        return split_generalized_force(ub, utheta, p.a, p.b, p.fmax)

    def _k_a(self, t: float, a: float) -> float:
        if (t % (4 * self.h)) < 2 * self.h:
            return 0.0
        return self._k_ah(t, a)

    def _k_ah(self, t: float, a: float) -> float:
        t1 = t % (2 * self.h)
        if t1 < self.h:
            return 0.0
        return float((4 * a * np.sin(np.pi * t1 / self.h) ** 2 * np.exp(-a * (3 * self.h - 2 * t1)) * (a**2 * self.h**2 + np.pi**2)) / (np.pi**2 * (np.exp(2 * a * self.h) - 1)))

    def _f1(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return p.kf1 * dyf + p.knf1 * dyf**3 + p.kr1 * dyr + p.knr1 * dyr**3 + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyf)) * ddyf + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyr)) * ddyr

    def _f2(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return p.a * (p.kf1 * dyf + p.knf1 * dyf**3 + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyf)) * ddyf) - p.b * (p.kr1 * dyr + p.knr1 * dyr**3 + 0.5 * ((p.be + p.bc) + (p.be - p.bc) * np.sign(ddyr)) * ddyr)

    def _ro1(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return 1 / p.mb_min * (abs(p.delta_kf1 * dyf) + abs(p.delta_knf1 * dyf**3) + abs(max(p.delta_be, p.delta_bc) * ddyf) + abs(p.delta_kr1 * dyr) + abs(p.delta_knr1 * dyr**3) + abs(max(p.delta_be, p.delta_bc) * ddyr))

    def _ro2(self, d: dict) -> float:
        p = self.p
        dyf, ddyf, dyr, ddyr = d["delta_yf"], d["delta_dyf"], d["delta_yr"], d["delta_dyr"]
        return 1 / p.Ip_min * ((p.a + p.delta_ab) * (abs(p.delta_kf1 * dyf) + abs(p.delta_knf1 * dyf**3) + abs(max(p.delta_be, p.delta_bc) * ddyf)) + (p.b + p.delta_ab) * (abs(p.delta_kr1 * dyr) + abs(p.delta_knr1 * dyr**3) + abs(max(p.delta_be, p.delta_bc) * ddyr)) + p.delta_ab * abs(self._f1(d)))
