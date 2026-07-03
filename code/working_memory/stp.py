import matplotlib.pyplot as plt
import numpy as np


class STP:
    def __init__(
        self,
        g_max=0.1,
        U=0.15,
        tau_f=1500.0,
        tau_d=200.0,
        tau=8.0,
        E=1.0,
        delay_step=2,
        **kwargs
    ):
        # 初始化参数
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.tau = tau
        self.U = U
        self.g_max = g_max
        self.E = E
        self.delay_step = delay_step
        # 获取每个连接的突触前神经元 pre_ids 和突触后神经元 post_ids
        self.pre_ids = np.array([0])
        self.post_ids = np.array([0])

        # 初始化变量
        num = len(self.pre_ids)
        self.x = np.ones(num)
        self.u = np.zeros(num)
        self.g = np.zeros(num)
        self.delay = [
            np.zeros(num) for _ in range(delay_step + 1)
        ]  # 定义一个处理 g 的延迟器

    def derivative(self):
        du = -self.u / self.tau_f
        dx = (1 - self.x) / self.tau_d
        dg = -self.g / self.tau
        return du, dx, dg

    def update(self, pre_spike, post_V_rest, dt):
        # 将 g 的计算延迟 delay_step 的时间步长
        delayed_g = self.delay.pop(0)

        # 计算突触后电流
        post_g = np.zeros(1)
        np.add.at(post_g, self.post_ids, delayed_g)
        post_input = post_g * (self.E - post_V_rest)

        # 更新各个变量
        syn_sps = np.array([pre_spike])
        u = self.u * np.exp(-dt / self.tau_f)
        x = 1.0 + (self.x - 1.0) * np.exp(-dt / self.tau_d)
        g = self.g * np.exp(-dt / self.tau)
        u = np.where(syn_sps, u + self.U * (1 - self.u), u)
        x = np.where(syn_sps, x - u * self.x, x)
        g = np.where(syn_sps, g + self.g_max * u * self.x, g)
        self.u = u
        self.x = np.clip(x, 0.0, 1.0)
        self.g = np.maximum(g, 0.0)

        # 更新延迟器
        self.delay.append(self.g.copy())
        return self.u.copy(), self.x.copy(), self.g.copy(), post_input.copy()


def section_input(values, durations, dt):
    pieces = []
    for value, duration in zip(values, durations):
        pieces.append(np.ones(int(round(duration / dt))) * value)
    inputs = np.concatenate(pieces)
    return inputs, len(inputs) * dt


def simulate_lif_spikes(inputs, dt):
    V_rest = -65.0
    V_reset = -65.0
    V_th = -50.0
    tau = 10.0
    V = V_rest
    spikes = np.zeros_like(inputs)
    for i, current in enumerate(inputs):
        V += dt * (-(V - V_rest) + current) / tau
        if V >= V_th:
            spikes[i] = 1.0
            V = V_reset
    return spikes


def run_STP(title=None, **kwargs):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    syn = STP(**kwargs)

    # 分段电流
    dt = 0.1
    inputs, dur = section_input(
        values=[22.0, 0.0, 22.0, 0.0], durations=[200.0, 200.0, 25.0, 75.0], dt=dt
    )
    ts = np.arange(len(inputs)) * dt
    spikes = simulate_lif_spikes(inputs, dt)

    # 运行模拟
    u = np.zeros((len(ts), 1))
    x = np.zeros((len(ts), 1))
    g = np.zeros((len(ts), 1))
    for i, spike in enumerate(spikes):
        u[i], x[i], g[i], _ = syn.update(spike > 0, post_V_rest=-65.0, dt=dt)

    # 可视化
    fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))

    plt.sca(gs[0])
    plt.plot(ts, x[:, 0], label="x")
    plt.plot(ts, u[:, 0], label="u")
    plt.legend(loc="center right")
    if title:
        plt.title(title)

    plt.sca(gs[1])
    plt.plot(ts, g[:, 0], label="g", color="#d62728")
    plt.legend(loc="center right")

    plt.xlabel("t (ms)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_STP(title="STF", U=0.1, tau_d=15, tau_f=200)
    run_STP(title="STD", U=0.4, tau_d=200, tau_f=15)
