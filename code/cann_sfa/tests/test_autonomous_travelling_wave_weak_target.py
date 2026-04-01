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
class AutonomousWaveConfig:
    num: int = 384
    tau: float = 1.0
    tau_v: float = 50.0
    k: float = 1.0
    a: float = 0.3
    j0: float = 1.0
    m: float = 0.9
    dt: float = 0.1
    cue_amplitude: float = 0.70
    target_amplitude: float = 0.03
    cue_duration: float = 40.0
    total_duration: float = 1400.0
    cue_pos: float = -2.6
    target_pos0: float = -1.1
    target_speed: float = 0.001
    noise_sigma: float = 0.0
    seed: int = 11
    display_start: float = 120.0
    z_min: float = -np.pi
    z_max: float = np.pi

    @property
    def mbar(self) -> float:
        return self.m * self.tau_v / self.tau

    @property
    def target_vbar(self) -> float:
        return self.target_speed * self.tau_v / self.a


class CANN1D_SFA(bp.dyn.NeuDyn):
    def __init__(
        self,
        num,
        m=0.1,
        tau=1.0,
        tau_v=10.0,
        k=0.1,
        a=0.5,
        j0=4.0,
        z_min=-bm.pi,
        z_max=bm.pi,
        **kwargs,
    ):
        super().__init__(size=num, **kwargs)
        self.tau = tau
        self.tau_v = tau_v
        self.k = k
        self.a = a
        self.j0 = j0
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
            self.j0
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


def wrap_angle(x):
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def build_input_schedule(config: AutonomousWaveConfig, model: CANN1D_SFA):
    total_steps = int(round(config.total_duration / config.dt))
    cue_steps = int(round(config.cue_duration / config.dt))

    rng = np.random.default_rng(config.seed)
    noise_scale = config.noise_sigma * np.sqrt(config.tau / config.dt)
    inputs = rng.normal(0.0, noise_scale, size=(total_steps, config.num)).astype(np.float32)

    cue_profile = np.asarray(
        model.gaussian_input(config.cue_pos, config.cue_amplitude),
        dtype=np.float32,
    )
    inputs[:cue_steps] += cue_profile

    positions = config.target_pos0 + config.target_speed * np.arange(total_steps) * config.dt
    positions = wrap_angle(positions)
    target = np.asarray(
        model.gaussian_input(positions.reshape(-1, 1), config.target_amplitude),
        dtype=np.float32,
    )
    inputs += target
    return inputs, positions


def half_max_widths(activity, dx):
    widths = np.zeros(len(activity), dtype=np.float64)
    for i, row in enumerate(activity):
        peak = float(np.max(row))
        if peak <= 0.0:
            continue
        widths[i] = float(np.sum(row >= 0.5 * peak) * dx)
    return widths


def simulate_autonomous_wave(config: AutonomousWaveConfig):
    brainstate.environ.set(dt=config.dt)
    bm.set_dt(config.dt)

    model = CANN1D_SFA(
        num=config.num,
        tau=config.tau,
        tau_v=config.tau_v,
        k=config.k,
        a=config.a,
        j0=config.j0,
        m=config.m,
        z_min=config.z_min,
        z_max=config.z_max,
    )
    inputs, target_pos = build_input_schedule(config, model)

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
    common_len = min(len(ts), len(u), len(v), len(center), len(inputs), len(target_pos))
    ts = ts[:common_len]
    u = u[:common_len]
    v = v[:common_len]
    center = center[:common_len]
    inputs = inputs[:common_len]
    target_pos = target_pos[:common_len]

    x = np.asarray(model.x)
    dx = float((config.z_max - config.z_min) / config.num)
    widths = half_max_widths(u, dx)

    display_start_idx = int(round(config.display_start / config.dt))
    center_window = center[display_start_idx:]
    drift_speed = 0.0
    if len(center_window) > 1:
        drift_speed = float((center_window[-1] - center_window[0]) / (ts[-1] - ts[display_start_idx]))

    relative = wrap_angle(center - target_pos)
    near_target = np.abs(relative) < config.a
    far_target = np.abs(relative) > 2.0 * config.a
    near_width = float(widths[near_target].mean()) if np.any(near_target) else float("nan")
    far_width = float(widths[far_target].mean()) if np.any(far_target) else float("nan")

    return {
        "config": config,
        "ts": ts,
        "x": x,
        "u": u,
        "v": v,
        "input": inputs,
        "center": center,
        "target_pos": target_pos,
        "relative": relative,
        "widths": widths,
        "drift_speed": drift_speed,
        "near_width": near_width,
        "far_width": far_width,
    }


def plot_autonomous_wave(result, save_path: Path | None = None, show: bool = False):
    config = result["config"]
    start_idx = int(round(config.display_start / config.dt))
    ts = result["ts"][start_idx:]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(ts, result["center"][start_idx:], linewidth=1.8, label="bump")
    axes[0].plot(ts, np.unwrap(result["target_pos"])[start_idx:], linewidth=1.6, label="target")
    axes[0].axvline(config.cue_duration, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("position")
    axes[0].legend(loc="upper left")
    axes[0].set_title(
        f"Autonomous Travelling Wave with Weak Target | "
        f"m={config.m:.2f} (mbar={config.mbar:.1f}), "
        f"target v={config.target_speed:.4f} (vbar={config.target_vbar:.2f}), "
        f"drift={result['drift_speed']:.5f}"
    )

    axes[1].plot(ts, result["relative"][start_idx:], color="tab:red", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("bump-target")

    axes[2].plot(ts, result["widths"][start_idx:], color="tab:green", linewidth=1.2)
    axes[2].axhline(result["near_width"], color="tab:orange", linestyle=":", linewidth=1.0)
    axes[2].axhline(result["far_width"], color="tab:blue", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("half-max width")
    axes[2].set_title(
        f"near-target width={result['near_width']:.4f}, "
        f"far-target width={result['far_width']:.4f}"
    )

    extent = [ts[0], ts[-1], result["x"][0], result["x"][-1]]
    axes[3].imshow(
        result["u"][start_idx:].T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[3].plot(ts, wrap_angle(result["target_pos"][start_idx:]), color="cyan", linewidth=1.0)
    axes[3].set_xlabel("time (ms)")
    axes[3].set_ylabel("network position")
    axes[3].set_title("Activity heatmap after transient")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_autonomous_wave(
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
    target_pos = np.unwrap(result["target_pos"])[start_idx:]
    widths = result["widths"][start_idx:]

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
        f"Autonomous Travelling Wave | weak target={config.target_amplitude:.2f}, "
        f"target speed={config.target_speed:.4f}"
    )
    input_line, = profile_ax.plot([], [], linewidth=1.8, label="cue/weak target input")
    bump_line, = profile_ax.plot([], [], linewidth=2.0, linestyle="--", label="bump")
    profile_ax.legend(loc="upper right")

    center_ax.set_xlim(float(ts[0]), float(ts[-1]))
    ymin = float(min(center.min(), target_pos.min()))
    ymax = float(max(center.max(), target_pos.max()))
    margin = 0.05 * max(1e-6, ymax - ymin)
    center_ax.set_ylim(ymin - margin, ymax + margin)
    center_ax.set_xlabel("time (ms)")
    center_ax.set_ylabel("position")
    target_line, = center_ax.plot([], [], linewidth=1.8, label="target")
    center_line, = center_ax.plot([], [], linewidth=1.8, label="bump")
    cursor = center_ax.axvline(float(ts[0]), color="black", linestyle=":", linewidth=1.0)
    status = center_ax.text(0.02, 0.96, "", transform=center_ax.transAxes, va="top")
    center_ax.legend(loc="upper left")

    def init():
        input_line.set_data([], [])
        bump_line.set_data([], [])
        target_line.set_data([], [])
        center_line.set_data([], [])
        cursor.set_xdata([float(ts[0]), float(ts[0])])
        status.set_text("")
        return input_line, bump_line, target_line, center_line, cursor, status

    def update(frame_id):
        idx = frames[frame_id]
        input_line.set_data(x, inputs[idx])
        bump_line.set_data(x, u[idx])
        target_line.set_data(ts[: idx + 1], target_pos[: idx + 1])
        center_line.set_data(ts[: idx + 1], center[: idx + 1])
        cursor.set_xdata([float(ts[idx]), float(ts[idx])])
        status.set_text(
            f"t={ts[idx]:.1f} ms\n"
            f"bump={center[idx]:.3f}\n"
            f"target={target_pos[idx]:.3f}\n"
            f"width={widths[idx]:.3f}"
        )
        return input_line, bump_line, target_line, center_line, cursor, status

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
        description="Autonomous travelling wave modulated by a weak slow target."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("code/cann_sfa/outputs/autonomous_travelling_wave_weak_target"),
    )
    parser.add_argument("--m", type=float, default=0.9)
    parser.add_argument("--cue-amplitude", type=float, default=0.70)
    parser.add_argument("--target-amplitude", type=float, default=0.03)
    parser.add_argument("--target-speed", type=float, default=0.001)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--frame-stride", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = AutonomousWaveConfig(
        m=args.m,
        cue_amplitude=args.cue_amplitude,
        target_amplitude=args.target_amplitude,
        target_speed=args.target_speed,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
    result = simulate_autonomous_wave(config)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    plot_autonomous_wave(
        result,
        save_path=args.save_dir / "autonomous_travelling_wave_weak_target.png",
        show=args.show,
    )
    animate_autonomous_wave(
        result,
        save_path=args.save_dir / "autonomous_travelling_wave_weak_target.gif",
        show=False,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    print(
        f"saved autonomous travelling wave with weak target to {args.save_dir} | "
        f"m={config.m:.2f}, mbar={config.mbar:.1f}, cue={config.cue_amplitude:.2f}, "
        f"target={config.target_amplitude:.2f}, target_v={config.target_speed:.4f}, "
        f"noise={config.noise_sigma:.3f}, drift={result['drift_speed']:.5f}, "
        f"near_width={result['near_width']:.4f}, far_width={result['far_width']:.4f}"
    )


if __name__ == "__main__":
    main()
