"""
实验1：波包的固有速度随 SFA 强度的变化
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cann_sfa_common as common


@dataclass(frozen=True)
class Config(common.Config):
    k: float = 0.7
    cue_speed: float = 4.36 / 3.0 * 1e-3
    cue_amp: float = 0.5
    cue_time: float = 10.0
    total_time: float = 2.0e4
    cue_pos: float = -(np.pi / 2.0 + np.pi / 8.0)
    scaled_m_start: float = 0.0
    scaled_m_end: float = 4.0
    n_samples: int = 81
    save_path: Path = common.OUT_DIR / "intrinsic_speed" / "intrinsic_speed.png"
    cache_path: Path = common.OUT_DIR / "intrinsic_speed" / "intrinsic_speed.npz"
    recompute: bool = False
    show: bool = True

    def __post_init__(self):
        object.__setattr__(self, "scaled_m", np.linspace(self.scaled_m_start, self.scaled_m_end, self.n_samples))

    @property
    def m(self) -> np.ndarray:
        return self.scaled_m * self.tau / self.tau_v


def theory_speed(m, tau, tau_v, a):
    s = m * tau_v / tau
    if s <= 1.0:
        return 0.0
    return 2.0 * a / tau_v * np.sqrt(max(s - np.sqrt(s), 0.0))


def fit_speed(s, c, lam):
    s = np.asarray(s, dtype=float)
    inner = np.maximum(s - lam * np.sqrt(np.maximum(s, 0.0)), 0.0)
    return c * np.sqrt(inner)


def make_inputs(cfg: Config):
    steps = int(round(cfg.total_time / cfg.dt))
    cue_steps = int(round(cfg.cue_time / cfg.dt))
    ts = np.arange(steps) * cfg.dt
    x = common.grid(cfg)
    inputs = np.zeros((steps, cfg.num))

    pos = cfg.cue_pos
    for i in range(min(steps, cue_steps)):
        if ts[i] >= cfg.cue_time:
            break
        pos = common.wrap(pos + cfg.cue_speed * cfg.dt)
        inputs[i] = common.gauss(x=x, pos=pos, amplitude=cfg.cue_amp, a=cfg.a)
    return inputs


def run_m(cfg: Config, m, inputs, conn_mat):
    model = common.CANN1DSFA(cfg, m=m, conn_mat=conn_mat)
    center = np.zeros(inputs.shape[0])
    for i, current_input in enumerate(inputs):
        model.input[:] = current_input
        model.step()
        center[i] = model.center
    return center


def measure_speed(center, m, cfg: Config):
    if m <= cfg.tau / cfg.tau_v or (center.max() - center.min()) < 0.1:
        return 0.0

    v = theory_speed(m=m, tau=cfg.tau, tau_v=cfg.tau_v, a=cfg.a)
    if v <= 0.0:
        return 0.0

    dt = np.pi / v
    lag = int(np.floor(dt / cfg.dt))
    if lag < 1:
        return 0.0

    peaks = np.flatnonzero(center == center.max())
    if len(peaks) == 0:
        return 0.0

    i = int(peaks[-1])
    if i < lag:
        return 0.0
    return float((center[i] - center[i - lag]) / dt)


def save_cache(cfg: Config, speed):
    np.savez(cfg.cache_path, speed=speed, scaled_m=cfg.scaled_m, m=cfg.m)


def scan(cfg: Config, speed=None):
    if speed is None or np.asarray(speed).shape != cfg.m.shape:
        speed = np.full_like(cfg.m, np.nan, dtype=float)
    else:
        speed = np.asarray(speed, dtype=float)
    total = len(cfg.m)
    inputs = make_inputs(cfg)
    conn_mat = common.conn(cfg)

    for i, m in enumerate(cfg.m):
        if np.isfinite(speed[i]):
            print(
                f"[{i + 1}/{total}] scaled={cfg.scaled_m[i]:.2f}, "
                f"m={cfg.m[i]:.6f}, speed={speed[i]:.6f} (cached)"
            )
            continue

        center = run_m(cfg, m, inputs, conn_mat)
        speed[i] = measure_speed(center, m, cfg)
        save_cache(cfg, speed)
        print(
            f"[{i + 1}/{total}] scaled={cfg.scaled_m[i]:.2f}, "
            f"m={cfg.m[i]:.6f}, speed={speed[i]:.6f}"
        )
    return speed


def load_or_scan(cfg: Config):
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.recompute and cfg.cache_path.exists():
        data = np.load(cfg.cache_path)
        speed = np.asarray(data["speed"], dtype=float)
        if speed.shape == cfg.m.shape and np.all(np.isfinite(speed)):
            return speed

        # Try to reuse cache points when an old run used a different sample count.
        aligned = np.full_like(cfg.m, np.nan, dtype=float)
        if "scaled_m" in data and data["scaled_m"].shape == speed.shape:
            cached_scaled = np.asarray(data["scaled_m"], dtype=float)
            for i, s in enumerate(cfg.scaled_m):
                idx = np.flatnonzero(np.isclose(cached_scaled, s, atol=1e-12, rtol=0.0))
                if idx.size > 0 and np.isfinite(speed[idx[0]]):
                    aligned[i] = float(speed[idx[0]])
            speed = aligned

        return scan(cfg, speed=speed)

    speed = scan(cfg)
    save_cache(cfg, speed)
    return speed


def fit_curve(cfg: Config, speed):
    s = np.asarray(cfg.scaled_m, dtype=float)
    y = np.asarray(speed, dtype=float)
    lam_grid = np.linspace(0.8, 1.4, 1201)

    best_c = 0.0
    best_lam = 1.0
    best_err = np.inf

    for lam in lam_grid:
        basis = np.sqrt(np.maximum(s - lam * np.sqrt(np.maximum(s, 0.0)), 0.0))
        denom = float(np.dot(basis, basis))
        if denom <= 0.0:
            continue
        c = float(np.dot(basis, y) / denom)
        err = float(np.dot(y - c * basis, y - c * basis))
        if err < best_err:
            best_err = err
            best_c = c
            best_lam = float(lam)

    return {
        "c": best_c,
        "lam": best_lam,
        "sc": best_lam**2,
        "speed": fit_speed(s, best_c, best_lam),
    }


def plot(cfg: Config, speed):
    theory = np.array([theory_speed(m, cfg.tau, cfg.tau_v, cfg.a) for m in cfg.m])
    fit = fit_curve(cfg, speed)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        cfg.scaled_m,
        theory * 1e3 / cfg.tau,
        color="black",
        linewidth=2.0,
        label="Theory",
    )
    ax.plot(
        cfg.scaled_m,
        speed * 1e3 / cfg.tau,
        "o-",
        color="tab:blue",
        linewidth=1.4,
        markersize=3.5,
        label="Simulation",
    )
    ax.plot(
        cfg.scaled_m,
        fit["speed"] * 1e3 / cfg.tau,
        "--",
        color="tab:red",
        linewidth=2.0,
        label=f"Fit: c={fit['c']:.5f}, lambda={fit['lam']:.3f}",
    )
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2)

    ax.set_xlim(0.0, float(cfg.scaled_m.max()))
    ax.set_ylim(bottom=-0.5)
    ax.set_xlabel("Scaled SFA strength m")
    ax.set_ylabel("Intrinsic speed")
    ax.set_title("Intrinsic speed of the network bump")
    ax.legend()
    fig.tight_layout()

    print(
        f"Fit: v(s) = {fit['c']:.8f} * sqrt(max(s - {fit['lam']:.8f} * sqrt(s), 0)), "
        f"s_c = {fit['sc']:.6f}"
    )

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.save_path, dpi=300)
    if cfg.show:
        plt.show()
    else:
        plt.close(fig)


def main():
    cfg = Config()
    plot(cfg, load_or_scan(cfg))


if __name__ == "__main__":
    main()
