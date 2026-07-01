import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


DT = 0.1


class CANN1D:
    # 1D CANN 模型的最小 NumPy 实现
    def __init__(self, num=256, tau=1.0, k=8.1, a=0.5, A=10.0, J0=4.0):
        self.num = num
        self.tau = tau
        self.k = k
        self.a = a
        self.A = A
        self.J0 = J0
        self.z_min = -np.pi
        self.z_max = np.pi
        self.z_range = self.z_max - self.z_min
        self.x = np.linspace(self.z_min, self.z_max, num)
        self.u = np.zeros(num)
        self.r = np.zeros(num)
        self.inp = np.zeros(num)
        self.conn_mat = self.make_conn()

    def dist(self, d):
        d = np.remainder(d, self.z_range)
        return np.where(d > 0.5 * self.z_range, d - self.z_range, d)

    def make_conn(self):
        d = self.dist(self.x[:, None] - self.x[None, :])
        return self.J0 * np.exp(-0.5 * (d / self.a) ** 2) / (
            np.sqrt(2.0 * np.pi) * self.a
        )

    def get_stimulus_by_pos(self, pos):
        return self.A * np.exp(-0.25 * (self.dist(self.x - pos) / self.a) ** 2)

    def update(self, inp, dt=DT):
        # 根据外部输入更新网络状态
        self.inp = inp
        u2 = self.u**2
        self.r = u2 / (1.0 + self.k * np.sum(u2))
        irec = self.conn_mat @ self.r
        self.u += (-self.u + irec + self.inp) / self.tau * dt
        return self.u.copy(), self.r.copy(), self.inp.copy()


def population_coding_input(model, before_duration, duration, after_duration, pos):
    # Population coding task
    total_steps = int(np.ceil((before_duration + duration + after_duration) / DT))
    inputs = np.zeros((total_steps, model.num))
    start = int(before_duration / DT)
    end = int((before_duration + duration) / DT)
    inputs[start:end] = model.get_stimulus_by_pos(pos)
    return inputs


def template_matching_input(model, duration, pos, rng):
    # Template matching task
    total_steps = int(np.ceil(duration / DT))
    stimulus = model.get_stimulus_by_pos(pos)
    noise = 0.1 * model.A * rng.standard_normal((total_steps, model.num))
    return stimulus + noise


def smooth_tracking_input(model, positions, durations):
    # Smooth tracking task
    total_steps = int(np.ceil(np.sum(durations) / DT))
    inputs = np.zeros((total_steps, model.num))
    start = 0
    for i, duration in enumerate(durations):
        steps = int(duration / DT)
        end = start + steps
        pos_seq = np.linspace(positions[i], positions[i + 1], steps)
        inputs[start:end] = np.asarray([model.get_stimulus_by_pos(pos) for pos in pos_seq])
        start = end
    if start < total_steps:
        inputs[start:] = model.get_stimulus_by_pos(positions[-1])
    return inputs


def run_simulation(model, inputs):
    # Define simulation step
    us, rs, inps = [], [], []
    for inp in inputs:
        u, r, current_input = model.update(inp)
        us.append(u)
        rs.append(r)
        inps.append(current_input)
    return np.asarray(us), np.asarray(rs), np.asarray(inps)


def animate_energy_landscape(x, u, stimulus, title):
    # Visualize
    fig, ax = plt.subplots(figsize=(7, 4))
    (u_line,) = ax.plot(x, u[0], label="u")
    (stim_line,) = ax.plot(x, stimulus[0], label="stimulus", linestyle="--")
    ax.set_xlabel("Neuron Position")
    ax.set_ylabel("Activity")
    ax.set_title(title)
    time_text = ax.text(
        0.05,
        0.95,
        "t=0.0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"},
    )
    ax.legend()
    ax.set_ylim(
        min(float(np.min(u)), float(np.min(stimulus))) * 1.05,
        max(float(np.max(u)), float(np.max(stimulus))) * 1.05 + 1e-12,
    )

    frame_step = max(1, len(u) // 200)
    frames = range(0, len(u), frame_step)

    def update(frame):
        u_line.set_ydata(u[frame])
        stim_line.set_ydata(stimulus[frame])
        time_text.set_text(f"t={frame * DT:.1f}")
        return u_line, stim_line, time_text

    animation = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    plt.show()
    return animation


def run_population_coding():
    # %%
    # 群体编码
    model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)

    # Get data and run simulation
    inputs = population_coding_input(
        model, before_duration=10.0, duration=20.0, after_duration=10.0, pos=0.0
    )
    u, r, inp = run_simulation(model, inputs)

    # Generate animation
    animate_energy_landscape(model.x, u, inp, "Energy Landscape - Population Coding")


def run_template_matching():
    # %%
    # 模板匹配
    rng = np.random.default_rng(123)
    model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)

    # Get data and run simulation
    inputs = template_matching_input(model, duration=50.0, pos=1.0, rng=rng)
    u, r, inp = run_simulation(model, inputs)

    # Generate animation
    animate_energy_landscape(model.x, u, inp, "Energy Landscape - Template Matching")


def run_smooth_tracking():
    # %%
    # 平滑追踪
    model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)

    # Get data and run simulation
    inputs = smooth_tracking_input(model, positions=[-2.0, 2.0], durations=[50.0])
    u, r, inp = run_simulation(model, inputs)

    # Generate animation
    animate_energy_landscape(model.x, u, inp, "Energy Landscape - Smooth Tracking")


if __name__ == "__main__":
    run_population_coding()
    run_template_matching()
    run_smooth_tracking()
