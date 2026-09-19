import numpy as np
import pandas as pd
import pytest

from anomaly.detectors import CUSUM, DETECTORS, EWMAChart, ExpectedTemperature, RollingZ, Threshold, debounce
from anomaly.evaluate import TRAIN, alarms, score
from anomaly.simulate import simulate

rng = np.random.default_rng(1)
NOISE = pd.Series(rng.normal(0, 1, 3000))


def test_simulation_is_reproducible():
    a, b = simulate(seed=7), simulate(seed=7)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert a.fault_start == b.fault_start


def test_healthy_run_has_no_fault():
    run = simulate(seed=1, fault=False)
    assert not run.data["fault"].any()


def test_debounce_needs_consecutive_flags():
    flags = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    assert debounce(flags, 3).tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1]


@pytest.mark.parametrize("name", list(DETECTORS))
def test_single_glitch_raises_no_alarm(name):
    x = NOISE.copy()
    x[2000] += 25
    det = DETECTORS[name]().fit(x[:TRAIN])
    assert not debounce(det.flags(x))[1500:].any()


@pytest.mark.parametrize("detector", [Threshold(), EWMAChart(), CUSUM()])
def test_step_change_is_caught(detector):
    x = NOISE.copy()
    x[2000:] += 5
    alarm = debounce(detector.fit(x[:TRAIN]).flags(x))
    assert alarm[2000:2100].any()


def test_rolling_z_gets_used_to_slow_drift():
    # The reason it is in the comparison: it misses exactly the fault we care about.
    x = NOISE.copy()
    x[1500:] += np.linspace(0, 8, 1500)
    assert not debounce(RollingZ().flags(x)).any()
    assert debounce(CUSUM().fit(x[:TRAIN]).flags(x)).any()


def test_residual_removes_load_swings():
    df = simulate(seed=2, fault=False).data
    resid = ExpectedTemperature().fit(df.iloc[:TRAIN]).residual(df)
    healthy = df["temperature"].iloc[TRAIN:]
    assert resid.iloc[TRAIN:].std() < 0.25 * healthy.std()


def test_score_counts_lead_time():
    run = simulate(seed=0)
    result = score(run, alarms(run, "CUSUM", use_residual=True))
    assert result.caught
    assert result.false_alarms_per_day == 0
    assert 0 < result.lead_hours < 72
