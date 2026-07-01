"""
1D CANN+SFA demo script containing three representative regimes:

- Smooth Tracking
- Oscillatory Tracking
- Traveling Wave

References
----------
- <https://papers.nips.cc/paper_files/paper/2022/file/d6797a91df19b768409b5178642dcb26-Paper-Conference.pdf>
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration of one CANN+SFA simulation regime.

    Attributes
    ----------
    name : str
        Regime name used in titles and output naming.
    num : int
        Number of neurons (spatial grid points on the ring).
    tau : float
        Time constant of neural activity variable u.
    tau_v : float
        Time constant of adaptation variable v.
    g : float
        Gain of the firing-rate nonlinearity.
    k : float
        Strength of global divisive inhibition.
    a : float
        Spatial width of recurrent kernel and external drive.
    m : float
        Coupling strength from u to adaptation v.
    dt : float
        Simulation time step in ms.
    alpha : float
        Amplitude of external Gaussian stimulus.
    v_ext : float
        External target velocity along the ring (rad/ms).
    target_pos0 : float
        Initial target position on the ring (rad).
    stim_duration : float | None
        Stimulus duration in ms. None means full trial duration.
    total_duration : float
        Total simulation duration in ms.
    """
    name: str
    num: int = 256
    tau: float = 1.0
    tau_v: float = 50.0
    g: float = 1.0
    k: float = 1.0
    a: float = 0.3
    m: float = 0.05
    dt: float = 0.1
    alpha: float = 0.2
    v_ext: float = 0.004
    target_pos0: float = 0.0
    stim_duration: float | None = None
    total_duration: float = 1400.0


# parameters for the three regimes
REGIMES = {
    "smooth": RegimeConfig(name="Smooth Tracking", m=0.05, alpha=0.2, v_ext=0.004),
    "oscillatory": RegimeConfig(
        name="Oscillatory Tracking", m=16.7 / 50.0, alpha=0.2, v_ext=0.20 * 0.3 / 50.0
    ),
    "travelling_wave": RegimeConfig(
        name="Traveling Wave", m=0.90, alpha=0.03, v_ext=0.0, stim_duration=30.0
    ),
}

# task type
SELECTED_REGIME = "all"  # "smooth", "oscillatory", "travelling_wave", "all"

# gif configs, you don't need to change them most of the time
MAKE_GIF = False
FRAME_STRIDE = 60
FPS = 12
SAVE_DIR = Path(__file__).parent / "outputs" / "triple_regimes"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = True


class CANN1DSFA:
    def __init__(self, num, m=0.1, tau=1.0, tau_v=10.0, g=4.0, k=0.1, a=0.5):
        self.num = num
        self.tau = tau
        self.tau_v = tau_v
        self.g = g
        self.k = k
        self.a = a
        self.m = m
        self.z_range = 2.0 * np.pi
        self.x = np.arange(num) * self.z_range / num - np.pi

        self.u = np.zeros(num)
        self.v = np.zeros(num)
        self.input = np.zeros(num)
        self.center = 0.0

        self.conn_mat = self.make_conn(self.x)
        self.phase_kernel = np.exp(1j * self.x)

    def _exprel(self, x):
        if abs(x) < 1e-8:
            return 1.0 + x / 2.0
        return np.expm1(x) / x

    def make_conn(self, x):
        d = wrapped_difference(x - x[:, None], z_range=self.z_range)
        return np.exp(-0.5 * np.square(d / self.a)) / (np.sqrt(2.0 * np.pi) * self.a)

    def update(self, dt):
        u2 = np.square(self.u)
        r = self.g * u2 / (1.0 + self.k * np.sum(u2))
        irec = np.dot(self.conn_mat, r)
        du = (-self.u + irec + self.input - self.v) / self.tau
        dv = (-self.v + self.m * self.u) / self.tau_v
        self.u[:] = np.maximum(
            self.u + dt * self._exprel(-dt / self.tau) * du, 0.0
        )
        self.v[:] = self.v + dt * self._exprel(-dt / self.tau_v) * dv
        self.input[:] = 0.0
        self.center = np.angle(np.sum(self.u * self.phase_kernel))


def wrapped_difference(x, z_range=2.0 * np.pi):
    return ((x + 0.5 * z_range) % z_range) - 0.5 * z_range


def gaussian_drive(x, pos, amplitude, a, z_range=2.0 * np.pi):
    delta = wrapped_difference(x - pos, z_range=z_range)
    return amplitude * np.exp(-(delta**2) / (4.0 * a**2))


def wrap_trajectory_for_display(values, z_range=2.0 * np.pi):
    wrapped = np.asarray(wrapped_difference(np.asarray(values), z_range=z_range))
    jumps = np.abs(np.diff(wrapped)) > (0.5 * z_range)
    wrapped = wrapped.copy()
    wrapped[1:][jumps] = np.nan
    return wrapped


def regime_title(config: RegimeConfig) -> str:
    return (
        f"{config.name} | "
        f"m={config.m:.3f}, alpha={config.alpha:.3f}, v_ext={config.v_ext:.4f}"
    )


def setup_position_axis(ax):
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_yticklabels(["-π", "0", "π"])


def file_stem(config: RegimeConfig) -> str:
    return config.name.lower().replace(" ", "_")


def build_inputs(config: RegimeConfig, x):
    total_steps = int(round(config.total_duration / config.dt))
    ts = np.arange(total_steps) * config.dt
    x = np.asarray(x)
    stim_duration = (
        config.total_duration if config.stim_duration is None else config.stim_duration
    )
    stim_steps = max(0, min(total_steps, int(round(stim_duration / config.dt))))

    inputs = np.zeros((total_steps, config.num))
    target_visible = np.full(total_steps, np.nan)
    if stim_steps > 0:
        target_visible[:stim_steps] = wrapped_difference(
            config.target_pos0 + config.v_ext * ts[:stim_steps]
        )
        inputs[:stim_steps] = np.asarray(
            gaussian_drive(
                x[None, :],
                target_visible[:stim_steps, None],
                amplitude=config.alpha,
                a=config.a,
                z_range=2.0 * np.pi,
            )
        )
    return inputs, target_visible


def simulate_regime(config: RegimeConfig):
    model = CANN1DSFA(
        num=config.num,
        tau=config.tau,
        tau_v=config.tau_v,
        g=config.g,
        k=config.k,
        a=config.a,
        m=config.m,
    )
    inputs, target_visible = build_inputs(config, model.x)
    ts = np.arange(inputs.shape[0]) * config.dt
    u = np.zeros((inputs.shape[0], config.num))
    center = np.zeros(inputs.shape[0])
    for i, current_input in enumerate(inputs):
        model.input[:] = current_input
        model.update(config.dt)
        u[i] = model.u
        center[i] = model.center
    center = np.unwrap(center)
    return {
        "config": config,
        "ts": ts,
        "x": np.asarray(model.x),
        "u": u,
        "input": inputs,
        "center": center,
        "target_visible": target_visible,
        "relative": center - target_visible,
    }


def plot_regime(result, save_path: Path | None = None, show: bool = False):
    config = result["config"]
    ts = result["ts"]
    center_display = wrap_trajectory_for_display(result["center"])
    target_display = wrap_trajectory_for_display(result["target_visible"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(ts, center_display, linewidth=1.8, label="bump")
    axes[0].plot(ts, target_display, linewidth=1.6, label="target")
    axes[0].set_ylabel("position")
    setup_position_axis(axes[0])
    axes[0].legend(loc="upper right")
    axes[0].set_title(regime_title(config))

    axes[1].plot(ts, result["relative"], color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("z(t)-vt")
    axes[1].set_title("relative position")

    extent = [ts[0], ts[-1], result["x"][0], result["x"][-1]]
    axes[2].imshow(
        result["u"].T, aspect="auto", origin="lower", extent=extent, cmap="magma"
    )
    axes[2].plot(ts, target_display, color="cyan", linewidth=1.0)
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("network position")
    setup_position_axis(axes[2])
    axes[2].set_title("activity heatmap")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_regime(
    result,
    save_path: Path | None = None,
    show: bool = False,
    frame_stride: int = 60,
    fps: int = 12,
):
    config = result["config"]
    ts = result["ts"]
    x = result["x"]
    u = result["u"]
    inputs = result["input"]
    center = wrap_trajectory_for_display(result["center"])
    target = wrap_trajectory_for_display(result["target_visible"])

    frames = list(range(0, len(ts), frame_stride))
    if frames[-1] != len(ts) - 1:
        frames.append(len(ts) - 1)

    max_y = float(max(u.max(), inputs.max()) * 1.15)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    profile_ax, center_ax = axes

    profile_ax.set_xlim(float(x[0]), float(x[-1]))
    profile_ax.set_ylim(float(min(inputs.min(), 0.0) * 1.1), max_y)
    profile_ax.set_ylabel("activity")
    profile_ax.set_title(regime_title(config))
    (input_line,) = profile_ax.plot([], [], linewidth=1.8, label="external input")
    (bump_line,) = profile_ax.plot([], [], linewidth=2.0, linestyle="--", label="bump")
    profile_ax.legend(loc="upper right")

    center_ax.set_xlim(float(ts[0]), float(ts[-1]))
    center_ax.set_xlabel("time (ms)")
    center_ax.set_ylabel("position")
    setup_position_axis(center_ax)
    (target_line,) = center_ax.plot([], [], linewidth=1.8, label="target")
    (center_line,) = center_ax.plot([], [], linewidth=1.8, label="bump")
    cursor = center_ax.axvline(
        float(ts[0]), color="black", linestyle=":", linewidth=1.0
    )
    status = center_ax.text(
        0.02,
        0.96,
        "",
        transform=center_ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"},
    )
    center_ax.legend(loc="upper right")

    def init():
        input_line.set_data([], [])
        bump_line.set_data([], [])
        target_line.set_data([], [])
        center_line.set_data([], [])
        cursor.set_xdata([float(ts[0]), float(ts[0])])
        status.set_text("")
        return (input_line, bump_line, target_line, center_line, cursor, status)

    def update(frame_id):
        idx = frames[frame_id]
        input_line.set_data(x, inputs[idx])
        bump_line.set_data(x, u[idx])
        target_line.set_data(ts[: idx + 1], target[: idx + 1])
        center_line.set_data(ts[: idx + 1], center[: idx + 1])
        cursor.set_xdata([float(ts[idx]), float(ts[idx])])
        status.set_text(
            f"t={ts[idx]:.1f} ms\n"
            f"bump={center[idx]:.3f}\n"
            f"target={target[idx]:.3f}\n"
            f"delta={wrapped_difference(center[idx]-target[idx]):.3f}"
        )
        return (input_line, bump_line, target_line, center_line, cursor, status)

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()

    if save_path is not None:
        animation.save(save_path, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_regime(
    config: RegimeConfig,
    save_dir: Path,
    make_gif: bool,
    show: bool,
    frame_stride: int,
    fps: int,
):
    result = simulate_regime(config)
    plot_regime(result, save_path=save_dir / f"{file_stem(config)}.png", show=show)
    if make_gif or show:
        animate_regime(
            result,
            save_path=save_dir / f"{file_stem(config)}.gif" if make_gif else None,
            show=show,
            frame_stride=frame_stride,
            fps=fps,
        )
    print(f"[{config.name}] finished")


def main():
    if SELECTED_REGIME == "all":
        targets = list(REGIMES.values())
        mode_label = "all regimes"
    else:
        if SELECTED_REGIME not in REGIMES:
            raise ValueError(f"Unknown SELECTED_REGIME: {SELECTED_REGIME}")
        targets = [REGIMES[SELECTED_REGIME]]
        mode_label = targets[0].name

    print(
        f"Exporting {mode_label} to {SAVE_DIR} | "
        f"gif={'on' if MAKE_GIF else 'off'}, frame_stride={FRAME_STRIDE}, fps={FPS}"
    )
    for config in targets:
        run_regime(
            config,
            save_dir=SAVE_DIR,
            make_gif=MAKE_GIF,
            show=SHOW_PLOTS,
            frame_stride=FRAME_STRIDE,
            fps=FPS,
        )


if __name__ == "__main__":
    main()
