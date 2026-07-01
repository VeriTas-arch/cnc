import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cann_sfa_triple_regimes import RegimeConfig, simulate_regime


def test_noise_triggered_travelling_wave_smoke():
    config = RegimeConfig(name="noise_triggered_smoke", num=64, m=0.9, alpha=0.7, v_ext=0.0, total_duration=20.0, dt=0.2, stim_duration=4.0)
    result = simulate_regime(config)
    assert result["u"].shape == (int(config.total_duration / config.dt), config.num)
    assert np.all(np.isfinite(result["u"]))
