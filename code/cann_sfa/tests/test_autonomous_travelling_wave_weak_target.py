import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cann_sfa_triple_regimes import RegimeConfig, simulate_regime


def test_autonomous_travelling_wave_weak_target_smoke():
    config = RegimeConfig(name="weak_target_smoke", num=64, m=0.9, alpha=0.03, v_ext=0.001, total_duration=20.0, dt=0.2, stim_duration=20.0)
    result = simulate_regime(config)
    assert result["u"].shape == (int(config.total_duration / config.dt), config.num)
    assert np.all(np.isfinite(result["center"]))
