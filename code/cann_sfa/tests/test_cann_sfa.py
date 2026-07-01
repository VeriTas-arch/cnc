import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cann_sfa_triple_regimes import RegimeConfig, simulate_regime, wrapped_difference


def test_tracking_smoke():
    config = RegimeConfig(name="tracking_smoke", num=64, total_duration=20.0, dt=0.2, alpha=0.2, v_ext=0.004)
    result = simulate_regime(config)
    assert result["u"].shape == (int(config.total_duration / config.dt), config.num)
    assert result["center"].shape[0] == result["u"].shape[0]
    assert np.all(np.isfinite(result["u"]))


def test_wrapped_difference_range():
    values = wrapped_difference(np.asarray([-4 * np.pi, -np.pi, 0.0, np.pi, 4 * np.pi]))
    assert np.all(values >= -np.pi)
    assert np.all(values <= np.pi)
