"""
实验1：波包的固有速度随 SFA 强度的变化

作业要求：
1. 补全 `measure_speed()`，根据论文/参考代码从 bump 中心轨迹估计 intrinsic speed。
2. 补全 `fit_curve()`，用与理论曲线同型的经验函数拟合仿真结果。
3. 运行脚本，得到理论曲线、仿真曲线和拟合曲线。
4. 思考为什么仿真曲线会与理论曲线产生系统性偏差。
"""

from dataclasses import dataclass, field
from pathlib import Path
import sys

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cann_sfa_common as common


# fmt: off
@dataclass(frozen=True)
class Config(common.Config):
    """实验 1 的参数配置。"""

    # Model / protocol parameters.
    k: float = 0.7
    cue_speed: float = 4.36 / 3.0 * 1e-3            # 初始短暂输入的移动速度
    cue_amp: float = 0.5                            # 初始短暂输入强度
    cue_time: float = 10.0                          # 初始短暂输入持续时间
    total_time: float = 2.0e4                       # 单次仿真总时长
    cue_pos: float = -(np.pi / 2.0 + np.pi / 8.0)   # 初始输入位置

    # Scan parameters.
    scaled_m: np.ndarray = field(default_factory=lambda: np.arange(0.0, 4.05, 0.05))
    # scaled_m 是横轴，对应 m * tau_v / tau。

    # Output / execution controls.
    save_path: Path = common.OUT_DIR / "intrinsic_speed" / "intrinsic_speed.png"    # 图片输出路径
    cache_path: Path = common.OUT_DIR / "intrinsic_speed" / "intrinsic_speed.npz"   # 缓存路径
    recompute: bool = True  # True: 忽略缓存并重新扫描；False: 优先复用缓存
    show: bool = True       # 是否弹出图窗

    @property
    def m(self) -> np.ndarray:
        """将 scaled_m 换算成模型中实际使用的 m。"""
        return self.scaled_m * self.tau / self.tau_v
# fmt: on


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
    bm.set_dt(cfg.dt)
    model = common.CANN1DSFA(cfg, m=m, conn_mat=conn_mat)
    runner = bp.DSRunner(
        model, inputs=("input", inputs, "iter"), monitors=["center"], progress_bar=False
    )
    runner.predict(cfg.total_time)
    return np.asarray(runner.mon.center)


def measure_speed(center, m, cfg: Config):
    """
    TODO: 根据中心轨迹 `center` 测量 intrinsic speed。

    1. 当 m <= tau / tau_v 时，速度记为 0。
    2. 当 bump 几乎不动时，速度记为 0。
    3. 用理论速度估计一个时间窗口 T_interval = pi / v_theory。
    4. 在该时间窗口上计算平均速度。
    """
    raise NotImplementedError("TODO: implement measure_speed()")


def save_cache(cfg: Config, speed):
    np.savez(cfg.cache_path, speed=speed, scaled_m=cfg.scaled_m, m=cfg.m)


def scan(cfg: Config, speed=None):
    speed = np.full_like(cfg.m, np.nan, dtype=float) if speed is None else speed
    total = len(cfg.m)
    inputs = bm.asarray(make_inputs(cfg))
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
        return scan(cfg, speed=speed)

    speed = scan(cfg)
    save_cache(cfg, speed)
    return speed


def fit_curve(cfg: Config, speed):
    """
    TODO: 用经验函数拟合仿真曲线。我们假定实际曲线与理论曲线同型，你也可以尝试其他函数形式。

        v_fit(s) = c * sqrt(max(s - lambda * sqrt(s), 0))

    其中 s = m * tau_v / tau。

    函数需要返回一个 dict，包含：
    - "c"
    - "lam"
    - "sc"
    - "speed"

    其中 "speed" 是拟合曲线在 cfg.scaled_m 上的取值。
    """
    raise NotImplementedError("TODO: implement fit_curve()")


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
