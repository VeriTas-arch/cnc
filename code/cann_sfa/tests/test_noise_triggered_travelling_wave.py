import argparse
from dataclasses import dataclass
from pathlib import Path

import brainpy as bp
import brainpy.math as bm
import brainstate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class NoiseWaveConfig:
    num: int = 512
    tau: float = 1.0
    tau_v: float = 50.0
    k: float = 1.0
    a: float = 0.3
    A: float = 0.7
    J0: float = 1.0
    m: float = 0.9
    dt: float = 0.1
    cue_duration: float = 40.0
    free_duration: float = 1960.0
    cue_pos: float = 0.0
    noise_sigma: float = 0.03
    seed: int = 7
    display_start: float = 200.0
    z_min: float = -np.pi
    z_max: float = np.pi

    @property
    def total_duration(self) -> float:
        return self.cue_duration + self.free_duration

    @property
    def mbar(self) -> float:
        return self.m * self.tau_v / self.tau


class CANN1D_SFA(bp.dyn.NeuDyn):
    def __init__(
        self,
        num,
        m=0.1,
        tau=1.0,
        tau_v=10.0,
        k=0.1,
        a=0.5,
        J0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
        **kwargs,
    ):
        super().__init__(size=num, **kwargs)
        self.tau = tau
        self.tau_v = tau_v
        self.k = k
        self.a = a
        self.J0 = J0
        self.m = m

        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = bm.arange(num) * self.z_range / num + z_min

        self.u = bm.Variable(bm.zeros(num))
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(0.0)

        self.conn_mat = self.make_conn(self.x)
        self.phase_kernel = bm.exp(1j * self.x)
        self.integral = bp.odeint(self.derivative)

    @property
    def derivative(self):
        def du(u, t, v, irec, iext):
            return (-u + irec + iext - v) / self.tau

        def dv(v, t, u):
            return (-v + self.m * u) / self.tau_v

        return bp.JointEq([du, dv])

    def dist(self, d):
        d = bm.remainder(d, self.z_range)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self, x):
        d = self.dist(x - x[:, None])
        return (
            self.J0
            * bm.exp(-0.5 * bm.square(d / self.a))
            / (bm.sqrt(2.0 * bm.pi) * self.a)
        )

    def gaussian_input(self, pos, amplitude):
        return amplitude * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

    def update(self, x=None):
        u2 = bm.square(self.u)
        r = u2 / (1.0 + self.k * bm.sum(u2))
        irec = bm.dot(self.conn_mat, r)
        u, v = self.integral(self.u, self.v, bp.share["t"], irec, self.input)
        self.u[:] = bm.maximum(u, 0.0)
        self.v[:] = v
        self.input[:] = 0.0
        self.center.value = bm.angle(bm.sum(self.u * self.phase_kernel))


def build_noise_triggered_input(config: NoiseWaveConfig, model: CANN1D_SFA):
    total_steps = int(round(config.total_duration / config.dt))
    cue_steps = int(round(config.cue_duration / config.dt))

    rng = np.random.default_rng(config.seed)
    noise_scale = config.noise_sigma * np.sqrt(config.tau / config.dt)
    noise = rng.normal(0.0, noise_scale, size=(total_steps, config.num)).astype(np.float32)

    cue = np.zeros((total_steps, config.num), dtype=np.float32)
    cue_profile = np.asarray(model.gaussian_input(config.cue_pos, config.A), dtype=np.float32)
    cue[:cue_steps] = cue_profile

    return cue + noise


def simulate_noise_wave(config: NoiseWaveConfig):
    brainstate.environ.set(dt=config.dt)
    bm.set_dt(config.dt)

    model = CANN1D_SFA(
        num=config.num,
        tau=config.tau,
        tau_v=config.tau_v,
        k=config.k,
        a=config.a,
        J0=config.J0,
        m=config.m,
        z_min=config.z_min,
        z_max=config.z_max,
    )
    inputs = build_noise_triggered_input(config, model)

    runner = bp.DSRunner(
        model,
        inputs=("input", bm.asarray(inputs), "iter"),
        monitors=["u", "v", "center"],
        progress_bar=False,
    )
    runner.predict(config.total_duration)

    ts = np.asarray(runner.mon.ts)
    u = np.asarray(runner.mon.u)
    v = np.asarray(runner.mon.v)
    center = np.unwrap(np.asarray(runner.mon.center))
    common_len = min(len(ts), len(center), len(inputs))

    ts = ts[:common_len]
    u = u[:common_len]
    v = v[:common_len]
    center = center[:common_len]
    inputs = inputs[:common_len]

    display_start_idx = int(round(config.display_start / config.dt))
    drift_window = center[display_start_idx:]
    drift_speed = 0.0 if len(drift_window) < 2 else float((drift_window[-1] - drift_window[0]) / (ts[-1] - ts[display_start_idx]))

    return {
        "config": config,
        "ts": ts,
        "x": np.asarray(model.x),
        "u": u,
        "v": v,
        "input": inputs,
        "center": center,
        "drift_speed": drift_speed,
    }


def plot_noise_wave(result, save_path: Path | None = None, show: bool = False):
    config = result["config"]
    start_idx = int(round(config.display_start / config.dt))
    ts = result["ts"][start_idx:]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(result["ts"], result["center"], linewidth=1.8)
    axes[0].axvline(config.cue_duration, color="black", linestyle="--", linewidth=1.0)
    axes[0].axvline(config.display_start, color="gray", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("center")
    axes[0].set_title(
        f"Noise-Triggered Travelling Wave | m={config.m:.2f} (mbar={config.mbar:.1f}), "
        f"cue A={config.A:.2f}, noise sigma={config.noise_sigma:.3f}, "
        f"drift={result['drift_speed']:.5f} rad/ms"
    )

    axes[1].plot(result["ts"], result["input"].mean(axis=1), linewidth=1.2, color="tab:orange")
    axes[1].axvline(config.cue_duration, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("mean input")

    extent = [ts[0], ts[-1], result["x"][0], result["x"][-1]]
    axes[2].imshow(
        result["u"][start_idx:].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("network position")
    axes[2].set_title("Activity heatmap after transient")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_noise_wave(
    result,
    save_path: Path | None = None,
    show: bool = False,
    frame_stride: int = 80,
    fps: int = 18,
):
    config = result["config"]
    start_idx = int(round(config.display_start / config.dt))
    ts = result["ts"][start_idx:]
    x = result["x"]
    u = result["u"][start_idx:]
    inputs = result["input"][start_idx:]
    center = result["center"][start_idx:]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    frames = list(range(0, len(ts), frame_stride))
    if frames[-1] != len(ts) - 1:
        frames.append(len(ts) - 1)

    max_y = float(max(u.max(), inputs.max()) * 1.15)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    profile_ax, center_ax = axes

    profile_ax.set_xlim(float(x[0]), float(x[-1]))
    profile_ax.set_ylim(float(min(inputs.min(), 0.0) * 1.1), max_y)
    profile_ax.set_ylabel("activity")
    profile_ax.set_title(
        f"Noise-Triggered Travelling Wave | mbar={config.mbar:.1f}, "
        f"cue={config.A:.2f}, noise={config.noise_sigma:.3f}"
    )
    input_line, = profile_ax.plot([], [], linewidth=1.8, label="noise + cue")
    bump_line, = profile_ax.plot([], [], linewidth=2.0, linestyle="--", label="bump")
    profile_ax.legend(loc="upper right")

    center_ax.set_xlim(float(ts[0]), float(ts[-1]))
    ymin = float(center.min())
    ymax = float(center.max())
    margin = 0.05 * max(1e-6, ymax - ymin)
    center_ax.set_ylim(ymin - margin, ymax + margin)
    center_ax.set_xlabel("time (ms)")
    center_ax.set_ylabel("center")
    center_line, = center_ax.plot([], [], linewidth=1.8)
    cursor = center_ax.axvline(float(ts[0]), color="black", linestyle=":", linewidth=1.0)
    status = center_ax.text(0.02, 0.96, "", transform=center_ax.transAxes, va="top")

    def init():
        input_line.set_data([], [])
        bump_line.set_data([], [])
        center_line.set_data([], [])
        cursor.set_xdata([float(ts[0]), float(ts[0])])
        status.set_text("")
        return input_line, bump_line, center_line, cursor, status

    def update(frame_id):
        idx = frames[frame_id]
        input_line.set_data(x, inputs[idx])
        bump_line.set_data(x, u[idx])
        center_line.set_data(ts[: idx + 1], center[: idx + 1])
        cursor.set_xdata([float(ts[idx]), float(ts[idx])])
        status.set_text(
            f"t={ts[idx]:.1f} ms\n"
            f"center={center[idx]:.3f}\n"
            f"mean input={inputs[idx].mean():.4f}"
        )
        return input_line, bump_line, center_line, cursor, status

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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Noise-triggered travelling wave in a 1D CANN+SFA: cue first, then pure Gaussian noise."
    )
    parser.add_argument("--save-dir", type=Path, default=Path("code/cann_sfa/outputs/noise_triggered_travelling_wave"))
    parser.add_argument("--m", type=float, default=0.9)
    parser.add_argument("--cue-amplitude", type=float, default=0.7)
    parser.add_argument("--noise-sigma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cue-duration", type=float, default=40.0)
    parser.add_argument("--free-duration", type=float, default=1960.0)
    parser.add_argument("--display-start", type=float, default=200.0)
    parser.add_argument("--frame-stride", type=int, default=80)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = NoiseWaveConfig(
        m=args.m,
        A=args.cue_amplitude,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        cue_duration=args.cue_duration,
        free_duration=args.free_duration,
        display_start=args.display_start,
    )
    result = simulate_noise_wave(config)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_noise_wave(
        result,
        save_path=args.save_dir / "noise_triggered_travelling_wave.png",
        show=args.show,
    )
    animate_noise_wave(
        result,
        save_path=args.save_dir / "noise_triggered_travelling_wave.gif",
        show=False,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    print(
        f"saved noise-triggered travelling wave to {args.save_dir} | "
        f"m={config.m:.2f}, mbar={config.mbar:.1f}, cue={config.A:.2f}, "
        f"noise={config.noise_sigma:.3f}, drift={result['drift_speed']:.5f}"
    )


if __name__ == "__main__":
    main()
