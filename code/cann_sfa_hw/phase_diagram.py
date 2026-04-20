"""
实验2：追踪状态的相图
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cann_sfa_common as common


@dataclass(frozen=True)
class Config(common.Config):
    v_ext: float = 4.36 / 3.0 * 1e-3
    total_time: float = 5000.0
    stable_time: float = 4500.0
    loc0: float = -(np.pi / 2.0 + np.pi / 8.0)
    num_samples: int = 10
    alpha_min: float = 0.14
    alpha_max: float = 0.35
    m_min: float = 80.0 / 48.0
    m_max: float = 200.0 / 48.0
    save_dir: Path = common.OUT_DIR / "phase_diagram"
    cache_path: Path = common.OUT_DIR / "phase_diagram" / "state_map.npz"
    recompute: bool = False
    show: bool = True

    @property
    def alpha(self):
        return np.linspace(self.alpha_min, self.alpha_max, self.num_samples)

    @property
    def m(self):
        return np.linspace(self.m_min, self.m_max, self.num_samples)


LABELS = {1: "Smooth", 2: "Oscillatory", 3: "Traveling"}
COLORS = ["#72B7B2", "#F58518", "#E45756"]


class Model(common.CANN1DSFA):
    def __init__(self, cfg: Config, alpha: float, m: float):
        super().__init__(cfg, m=m)
        self.cfg = cfg
        self.alpha = alpha
        self.center_u = bm.Variable(0.0)
        self.center_i = bm.Variable(0.0)

    def update(self):
        t = bp.share["t"]
        loc = common.wrap(self.cfg.loc0 + self.cfg.v_ext * (t + self.dt))
        delta = common.circ_diff(self.x - loc, self.z_range)
        self.input[:] = self.alpha * bm.exp(-(delta**2) / (4.0 * self.a**2))
        self.step()
        self.center_i.value = loc
        self.center_u.value = common.fix_center(self.center, loc, self.a)


def run_point(alpha: float, m: float, cfg: Config):
    bm.set_dt(cfg.dt)
    model = Model(cfg, alpha=alpha, m=m)
    runner = bp.DSRunner(model, monitors=["center_u", "center_i"], progress_bar=False)
    runner.predict(cfg.total_time)

    u = np.asarray(runner.mon.center_u)
    i = np.asarray(runner.mon.center_i)
    start = int(round(cfg.stable_time / cfg.dt))
    delta = u - i
    delta = delta[start:] - np.mean(delta[start:])
    amp = float(np.max(delta) - np.min(delta))

    if amp > 1.0:
        return 3
    if amp > 0.1:
        return 2
    return 1


def scan(cfg: Config):
    state = np.zeros((len(cfg.alpha), len(cfg.m)), dtype=np.int8)
    total = len(cfg.alpha)
    for i, alpha in enumerate(cfg.alpha):
        for j, m in enumerate(cfg.m):
            state[i, j] = run_point(alpha, m, cfg)
        print(f"[scan] alpha row {i + 1}/{total} finished")
    return state


def plot(cfg: Config, state):
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.matplotlib.colors.ListedColormap(COLORS)
    norm = plt.matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    image = ax.imshow(
        state.T,
        origin="lower",
        aspect="auto",
        extent=[
            float(cfg.alpha[0]),
            float(cfg.alpha[-1]),
            float(cfg.m[0]),
            float(cfg.m[-1]),
        ],
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels([LABELS[k] for k in [1, 2, 3]])
    ax.set_xlabel(r"input strength $\alpha$")
    ax.set_ylabel("feedback inhibition strength m")
    ax.set_title("tracking-states phase diagram")
    fig.tight_layout()

    path = cfg.save_dir / "phase_diagram.png"
    fig.savefig(path, dpi=300)
    if cfg.show:
        plt.show()
    else:
        plt.close(fig)


def load_or_scan(cfg: Config):
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.recompute and cfg.cache_path.exists():
        data = np.load(cfg.cache_path)
        return data["state_map"]

    state = scan(cfg)
    np.savez(cfg.cache_path, state_map=state, alpha=cfg.alpha, m=cfg.m)
    return state


def main():
    cfg = Config()
    plot(cfg, load_or_scan(cfg))


if __name__ == "__main__":
    main()
