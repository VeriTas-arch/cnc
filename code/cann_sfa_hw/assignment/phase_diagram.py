"""
实验2：追踪状态的相图

作业要求：
1. 补全 `run_point()`，对单个 (alpha, m) 参数点进行分类。
2. 运行脚本，得到 smooth / oscillatory / traveling 三态相图。
3. 定性解释为什么随着参数变化会出现不同的动力学区域。
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


# fmt: off
@dataclass(frozen=True)
class Config(common.Config):
    """实验 2 的参数配置。"""

    # Tracking protocol parameters.
    v_ext: float = 4.36 / 3.0 * 1e-3            # 外部输入移动速度
    total_time: float = 5000.0                  # 单个参数点的仿真总时长
    stable_time: float = 4500.0                 # 用于分类的稳态分析起点
    loc0: float = -(np.pi / 2.0 + np.pi / 8.0)  # 外部输入初始位置

    # Scan range.
    num_samples: int = 10           # alpha 和 m 两个方向的采样点数
    alpha_min: float = 0.14         # alpha 扫描下界
    alpha_max: float = 0.35         # alpha 扫描上界
    m_min: float = 80.0 / 48.0      # m 扫描下界
    m_max: float = 200.0 / 48.0     # m 扫描上界

    # Output / execution controls.
    save_dir: Path = common.OUT_DIR / "phase_diagram"                       # 图片输出目录
    cache_path: Path = common.OUT_DIR / "phase_diagram" / "state_map.npz"   # 状态图缓存路径
    recompute: bool = True  # True: 忽略缓存并重新扫描；False: 优先复用缓存
    show: bool = True       # 是否弹出图窗

    @property
    def alpha(self):
        """相图纵轴参数：输入强度 alpha。"""
        return np.linspace(self.alpha_min, self.alpha_max, self.num_samples)

    @property
    def m(self):
        """相图横轴参数：反馈抑制强度 m。"""
        return np.linspace(self.m_min, self.m_max, self.num_samples)
# fmt: on


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
    """
    TODO(student):
    对单个参数点 (alpha, m) 运行 tracking 仿真，并返回状态类别：

    - 1: smooth
    - 2: oscillatory
    - 3: traveling

    1. 运行模型，记录 center_u 和 center_i。
    2. 取稳态时间窗后的相对位移 L = center_u - center_i。
    3. 去掉均值后计算振幅。
    4. 用振幅阈值分类：
       - amp > 1.0  -> traveling
       - amp > 0.1  -> oscillatory
       - else       -> smooth
    """
    raise NotImplementedError("TODO: implement run_point()")


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
