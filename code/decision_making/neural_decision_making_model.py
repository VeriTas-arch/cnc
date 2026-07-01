import matplotlib.pyplot as plt
import numpy as np

JE = 7.0
JM = 3.0


def section_input(values, durations, dt=0.1, return_length=False):
    inputs = []
    for value, duration in zip(values, durations):
        steps = int(duration / dt)
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            value = np.asarray([value, value], dtype=float)
        inputs.append(np.repeat(value.reshape(1, -1), steps, axis=0))
    inputs = np.concatenate(inputs, axis=0)
    return (inputs, len(inputs) * dt) if return_length else inputs


class DecisionMakingModel:
    def __init__(
        self, size, In=0.6, gamma=0.1, theta=5.0, alpha=1.5, beta=3.0, tau_s=100.0
    ):
        self.num = int(size)

        # 初始化参数
        self.In = In
        self.gamma = gamma
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.tau_s = tau_s

        # 初始化变量
        self.s1 = np.zeros(self.num)
        self.s2 = np.zeros(self.num)
        self.r1 = np.zeros(self.num)
        self.r2 = np.zeros(self.num)
        self.I10 = np.zeros(self.num)
        self.I20 = np.zeros(self.num)
        self.I1 = np.zeros(self.num)
        self.I2 = np.zeros(self.num)

        # 噪声输入
        self.I1_noise = 0
        self.I2_noise = 0

    def _exprel(self, x):
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        small = np.abs(x) < 1e-8
        out[small] = 1.0 + x[small] / 2.0
        out[~small] = np.expm1(np.clip(x[~small], None, 50.0)) / x[~small]
        return out

    def _sigmoid(self, x):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def _activation(self, x):
        return self.beta / self.gamma * np.logaddexp(0.0, (x - self.theta) / self.alpha)

    def _input_current(self, s_self, s_other, I_ext):
        return JE * s_self - JM * s_other + I_ext

    def _ds_linear(self, s_self, s_other, I_ext):
        x = self._input_current(s_self, s_other, I_ext)
        r = self._activation(x)
        dr_ds = (
            self.beta
            / self.gamma
            * self._sigmoid((x - self.theta) / self.alpha)
            * JE
            / self.alpha
        )
        return (-1.0 + self.gamma * (-r + (1.0 - s_self) * dr_ds)) / self.tau_s

    def ds1(self, s1, s2, I10=0.6):
        I1 = I10 + self.I1_noise
        x1 = self._input_current(s1, s2, I1)
        r1 = self._activation(x1)
        return (-s1 + (1.0 - s1) * self.gamma * r1) / self.tau_s

    def ds2(self, s2, s1, I20=0.6):
        I2 = I20 + self.I2_noise
        x2 = self._input_current(s2, s1, I2)
        r2 = self._activation(x2)
        return (-s2 + (1.0 - s2) * self.gamma * r2) / self.tau_s

    def update(self, I10, I20, dt):
        self.I10[:] = I10
        self.I20[:] = I20

        # 更新噪声（每次迭代都重新计算 I1, I2 中的噪声）
        self.I1_noise = self.In * np.random.randn(self.num) * np.sqrt(dt)
        self.I2_noise = self.In * np.random.randn(self.num) * np.sqrt(dt)

        # 更新 s1、s2
        ds1 = self.ds1(self.s1, self.s2, I10=self.I10)
        ds2 = self.ds2(self.s2, self.s1, I20=self.I20)
        s1_linear = self._ds_linear(self.s1, self.s2, self.I10 + self.I1_noise)
        s2_linear = self._ds_linear(self.s2, self.s1, self.I20 + self.I2_noise)
        self.s1 = self.s1 + dt * self._exprel(dt * s1_linear) * ds1
        self.s2 = self.s2 + dt * self._exprel(dt * s2_linear) * ds2

        # 用更新后的 s1、s2 计算 r1、r2
        I1 = self.I10 + self.I1_noise
        x1 = self._input_current(self.s1, self.s2, I1)
        self.r1 = self._activation(x1)
        self.I1 = I1

        I2 = self.I20 + self.I2_noise
        x2 = self._input_current(self.s2, self.s1, I2)
        self.r2 = self._activation(x2)
        self.I2 = I2

        # 重置外部输入
        self.I10[:] = 0.0
        self.I20[:] = 0.0

    def run(self, inputs, dt):
        inputs = np.asarray(inputs, dtype=float)
        ts = np.arange(inputs.shape[0]) * dt
        records = {
            key: np.zeros((inputs.shape[0], self.num))
            for key in ["s1", "s2", "r1", "r2", "I1", "I2"]
        }
        for i in range(inputs.shape[0]):
            self.update(inputs[i, 0], inputs[i, 1], dt)
            records["s1"][i] = self.s1
            records["s2"][i] = self.s2
            records["r1"][i] = self.r1
            records["r2"][i] = self.r2
            records["I1"][i] = self.I1
            records["I2"][i] = self.I2
        return ts, records


def plot_decision_traces(ts, mon, dmnet, pre_stimulus_period, stimulus_period, title=None):
    # 以 tau_s 为时间尺度，等间隔地从 runner 保存的变量中取值
    indur = int(dmnet.tau_s)
    n_steps = ts.shape[0]
    sampled = slice(0, n_steps - 1, indur)

    x_ts = ts[sampled] / indur
    y_s1 = mon["s1"][sampled]
    y_s2 = mon["s2"][sampled]
    y_r1 = mon["r1"][sampled]
    y_r2 = mon["r2"][sampled]
    y_I1 = mon["I1"][sampled]
    y_I2 = mon["I2"][sampled]

    # 可视化
    fig, gs = plt.subplots(3, 1, figsize=(8, 8), sharex="all")
    if title:
        gs[0].set_title(title)
    gs[0].plot(x_ts, y_s1, label="s1")
    gs[0].plot(x_ts, y_s2, label="s2")
    gs[0].axvline(
        pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
    )
    gs[0].axvline(
        (pre_stimulus_period + stimulus_period) / indur,
        0.0,
        1.0,
        linestyle="dashed",
        color="#444444",
    )
    gs[0].set_ylabel(r"gating variable $s$", fontsize=16)
    gs[0].legend()

    gs[1].plot(x_ts, y_r1, label="r1")
    gs[1].plot(x_ts, y_r2, label="r2")
    gs[1].axvline(
        pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
    )
    gs[1].axvline(
        (pre_stimulus_period + stimulus_period) / indur,
        0.0,
        1.0,
        linestyle="dashed",
        color="#444444",
    )
    gs[1].set_ylabel(r"firing rate $r$", fontsize=16)
    gs[1].legend()

    gs[2].plot(x_ts, y_I1, label="I1")
    gs[2].plot(x_ts, y_I2, label="I2")
    gs[2].axvline(
        pre_stimulus_period / indur, 0.0, 1.0, linestyle="dashed", color="#444444"
    )
    gs[2].axvline(
        (pre_stimulus_period + stimulus_period) / indur,
        0.0,
        1.0,
        linestyle="dashed",
        color="#444444",
    )
    gs[2].set_xlabel("time(" r"$\tau_s$)", fontsize=16)
    gs[2].set_ylabel(r"$I_{ext}$", fontsize=16)
    gs[2].legend()

    plt.subplots_adjust(hspace=0.1)
    plt.show()


def plot_phase_plane(mon, I0=0.6, title=None):
    ## DM model 的动力学分析
    # I1 = I2 的平均输入强度，修改其值以观察相平面图的变化
    model = DecisionMakingModel(1, In=0.0)
    s1_grid, s2_grid = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    model.I1_noise = 0.0
    model.I2_noise = 0.0
    ds1 = model.ds1(s1_grid, s2_grid, I10=I0)
    ds2 = model.ds2(s2_grid, s1_grid, I20=I0)

    ## 画出相图
    plt.figure(figsize=(4.5, 4.5))
    if title:
        plt.title(title)
    # 画出向量场
    skip = 8
    plt.quiver(
        s1_grid[::skip, ::skip],
        s2_grid[::skip, ::skip],
        ds1[::skip, ::skip],
        ds2[::skip, ::skip],
        color="lightgrey",
    )
    # 画出零增长等值线
    s1_nullcline = plt.contour(
        s1_grid, s2_grid, ds1, levels=[0.0], colors="tab:blue", linestyles="--"
    )
    s2_nullcline = plt.contour(
        s1_grid, s2_grid, ds2, levels=[0.0], colors="tab:orange", linestyles="--"
    )
    plt.clabel(s1_nullcline, fmt=r"$ds_1/dt=0$", inline=True)
    plt.clabel(s2_nullcline, fmt=r"$ds_2/dt=0$", inline=True)
    # 画出奇点
    speed = np.sqrt(ds1**2 + ds2**2)
    fixed_idx = np.unravel_index(np.argmin(speed), speed.shape)
    plt.plot(s1_grid[fixed_idx], s2_grid[fixed_idx], "ko", label="approx. fixed point")

    # 画出 s1, s2 的运动轨迹
    plt.plot(mon["s1"], mon["s2"], color="red", label="s1,s2 trajectory (from zero)")
    plt.xlabel(r"$s_1$")
    plt.ylabel(r"$s_2$")
    plt.legend()
    plt.show()


def run_decision_experiment(
    name,
    current_values,
    pre_stimulus_period,
    stimulus_period,
    simulation_dt=0.1,
    model_in=0.6,
    phase_i0=0.6,
):
    # 生成模型
    dmnet = DecisionMakingModel(1, In=model_in)

    # 定义电流随时间的变化
    inputs, total_period = section_input(
        values=[0, current_values],  # I10, I20 的取值
        durations=[pre_stimulus_period, stimulus_period],
        return_length=True,
        dt=simulation_dt,
    )

    # 运行数值模拟
    ts, mon = dmnet.run(inputs, simulation_dt)

    # 可视化
    plot_decision_traces(
        ts,
        mon,
        dmnet,
        pre_stimulus_period,
        stimulus_period,
        title=name,
    )
    plot_phase_plane(mon, I0=phase_i0, title=f"{name} phase plane")
    return ts, mon, dmnet, total_period


def main():
    experiments = [
        {
            "name": "Biased input",
            "pre_stimulus_period": 0.0,
            "stimulus_period": 2000.0,
            "current_values": [0.66, 0.6],
        },
        {
            "name": "Equal input",
            "pre_stimulus_period": 0.0,
            "stimulus_period": 10000.0,
            "current_values": [0.6, 0.6],
        },
    ]

    simulation_dt = 0.1
    for experiment in experiments:
        run_decision_experiment(
            name=experiment["name"],
            current_values=experiment["current_values"],
            pre_stimulus_period=experiment["pre_stimulus_period"],
            stimulus_period=experiment["stimulus_period"],
            simulation_dt=simulation_dt,
        )


if __name__ == "__main__":
    main()
