"""Synthetic motor data: temperature and current, one reading per minute.

The motor follows a daily load cycle plus random load changes. Temperature
follows load with a lag. At some point a bearing starts to wear. Friction
then adds heat that grows over time, the readings get noisier, and near the
end the motor overheats. A few single-sample sensor glitches are mixed in.
They are not faults, and a good detector should ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Run:
    data: pd.DataFrame     # columns: temperature, current, fault
    fault_start: int       # index of the first degraded reading


def simulate(minutes: int = 3 * 24 * 60, seed: int = 0, fault: bool = True) -> Run:
    rng = np.random.default_rng(seed)
    t = np.arange(minutes)

    # Load: daily shift pattern plus a random walk of set-point changes.
    daily = 0.5 + 0.3 * np.sin(2 * np.pi * (t / 1440 - 0.3))
    steps = np.repeat(rng.normal(0, 0.08, minutes // 60 + 1), 60)[:minutes]
    load = np.clip(daily + steps, 0.1, 1.0)
    current = 12 * load + rng.normal(0, 0.3, minutes)          # amps

    # Temperature lags load: first-order response with a ~40 minute time constant.
    target = 45 + 30 * load
    temp = np.empty(minutes)
    temp[0] = target[0]
    alpha = 1 / 40
    for i in range(1, minutes):
        temp[i] = temp[i - 1] + alpha * (target[i] - temp[i - 1])
    noise_sd = np.full(minutes, 0.6)

    fault_start = minutes
    if fault:
        fault_start = int(rng.uniform(0.55, 0.75) * minutes)
        n = minutes - fault_start
        wear = np.linspace(0, 1, n) ** 2                       # slow at first, then faster
        temp[fault_start:] += 9 * wear
        noise_sd[fault_start:] += 1.5 * wear
        final = int(0.97 * minutes)
        temp[final:] += np.linspace(0, 20, minutes - final)    # overheating

    temp = temp + rng.normal(0, 1, minutes) * noise_sd
    glitches = rng.choice(minutes, size=6, replace=False)
    temp[glitches] += rng.choice([-1, 1], size=6) * rng.uniform(12, 20, size=6)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=minutes, freq="min"),
        "temperature": temp,
        "current": current,
        "fault": t >= fault_start,
    })
    return Run(df, fault_start)
