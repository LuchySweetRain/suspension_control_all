from __future__ import annotations

import numpy as np


class Controller:
    name = "controller"

    def reset(self):
        pass

    def compute_action(self, obs: np.ndarray, info: dict) -> np.ndarray:
        raise NotImplementedError


def split_generalized_force(ub: float, utheta: float, a: float, b: float, fmax: float) -> np.ndarray:
    uaf = (b * ub + utheta) / (a + b)
    uar = (a * ub - utheta) / (a + b)
    return np.clip(np.asarray([uaf, uar], dtype=np.float64), -fmax, fmax)

