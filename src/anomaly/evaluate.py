"""Compare detectors over many simulated runs.

    python -m anomaly.evaluate            # table over 40 runs
    python -m anomaly.evaluate --plot     # also writes docs/example_run.png

For each detector and each input (raw temperature or residual), we measure:
  * false alarms per day in the healthy stretch after the training day
  * how often the fault is caught before the motor starts to overheat
  * lead time: hours between the alarm and the start of overheating
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .detectors import DETECTORS, ExpectedTemperature, debounce
from .simulate import Run, simulate

TRAIN = 1440          # the first day is assumed healthy


@dataclass
class Result:
    false_alarms_per_day: float
    caught: bool
    lead_hours: float | None


def alarms(run: Run, detector_name: str, use_residual: bool) -> np.ndarray:
    df = run.data
    x = df["temperature"]
    if use_residual:
        x = ExpectedTemperature().fit(df.iloc[:TRAIN]).residual(df)
    det = DETECTORS[detector_name]().fit(x.iloc[:TRAIN])
    return debounce(det.flags(x))


def score(run: Run, alarm: np.ndarray) -> Result:
    overheat = int(0.97 * len(alarm))
    healthy = alarm[TRAIN:run.fault_start]
    episodes = np.count_nonzero(healthy[1:] & ~healthy[:-1]) + int(healthy[:1].any())
    days = len(healthy) / 1440
    hits = np.flatnonzero(alarm[run.fault_start:overheat])
    caught = hits.size > 0
    lead = (overheat - run.fault_start - hits[0]) / 60 if caught else None
    return Result(episodes / days, caught, lead)


def compare(n_runs: int = 40) -> pd.DataFrame:
    runs = [simulate(seed=s) for s in range(n_runs)]
    rows = []
    for name in DETECTORS:
        for resid in (False, True):
            res = [score(r, alarms(r, name, resid)) for r in runs]
            leads = [r.lead_hours for r in res if r.caught]
            rows.append({
                "detector": name,
                "input": "residual" if resid else "raw",
                "false alarms / day": np.mean([r.false_alarms_per_day for r in res]),
                "caught": np.mean([r.caught for r in res]),
                "median lead (h)": np.median(leads) if leads else np.nan,
            })
    return pd.DataFrame(rows)


def plot(path: str = "docs/example_run.png", seed: int = 3) -> None:
    import matplotlib.pyplot as plt

    run = simulate(seed=seed)
    df = run.data
    resid = ExpectedTemperature().fit(df.iloc[:TRAIN]).residual(df)
    hours = np.arange(len(df)) / 60

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.6]})
    axes[0].plot(hours, df["temperature"], lw=0.6, color="0.3")
    axes[0].set_ylabel("temperature (°C)")
    axes[1].plot(hours, resid, lw=0.6, color="0.3")
    axes[1].set_ylabel("residual (°C)")
    for ax in axes[:2]:
        ax.axvline(run.fault_start / 60, color="tab:red", ls="--", lw=1)
        ax.axvspan(0, TRAIN / 60, color="0.92")
    axes[0].text(TRAIN / 120, axes[0].get_ylim()[1], "training day", ha="center", va="top", fontsize=9)
    axes[0].text(run.fault_start / 60, axes[0].get_ylim()[1], " bearing wear starts",
                 color="tab:red", va="top", fontsize=9)

    for row, name in enumerate(DETECTORS):
        a = alarms(run, name, use_residual=True)
        axes[2].fill_between(hours, row - 0.35, row + 0.35, where=a, step="mid", color="tab:orange", lw=0)
    axes[2].axvline(run.fault_start / 60, color="tab:red", ls="--", lw=1)
    axes[2].set_yticks(range(len(DETECTORS)), list(DETECTORS))
    axes[2].set_ylim(-0.6, len(DETECTORS) - 0.4)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("hours")
    axes[2].set_title("alarms on the residual", fontsize=10, loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    table = compare(args.runs)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
