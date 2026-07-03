import matplotlib.pyplot as plt
import numpy as np

from synapse_utils import apply_brainpy_delay, create_post_hh


class NMDA:
    def __init__(
        self,
        g_max=0.02,
        E=0.0,
        c_Mg=1.2,
        alpha1=2.0,
        beta1=0.01,
        alpha2=0.2,
        beta2=0.5,
        delay_step=2,
        T=1.0,
        T_duration=1.0,
    ):
        self.g_max = g_max
        self.E = E
        self.c_Mg = c_Mg
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.T = T
        self.T_duration = T_duration
        self.delay_step = delay_step

        # 使用连接矩阵将 pre 侧脉冲映射到 post 侧，避免 numba 依赖
        self.x = 0.0
        self.s = 0.0
        self.g = 0.0
        self.b = 0.0

        # 初始化为一个非常大的负数，使得初始时刻 [T] 的值为 0
        # 可以思考，如果我们将其初始化为 0，会对模拟结果产生什么样的影响？
        self.spike_arrival_time = -1e7

    def _exprel(self, x):
        if abs(x) < 1e-8:
            return 1.0 + x / 2.0
        return np.expm1(x) / x

    def ds(self, x):
        return self.alpha1 * x * (1 - self.s) - self.beta1 * self.s

    def dx(self, T):
        return self.alpha2 * T * (1 - self.x) - self.beta2 * self.x

    def update(self, pre_spike, post_V, t, dt):
        ## step 1：更新突触前神经元产生 spike
        post_sp = bool(pre_spike)
        if post_sp:
            self.spike_arrival_time = t  # 计算神经递质到达突触后膜的时间

        ## step 2：计算 [T] 的影响
        T = self.T if (t - self.spike_arrival_time) < self.T_duration else 0.0

        ## step 3：更新 s，g，b 的数值
        ds = self.ds(self.x)
        dx = self.dx(T)
        s_linear = -self.alpha1 * self.x - self.beta1
        x_linear = -self.alpha2 * T - self.beta2
        self.s += dt * self._exprel(dt * s_linear) * ds
        self.x += dt * self._exprel(dt * x_linear) * dx
        self.s = np.clip(self.s, 0.0, 1.0)
        self.x = np.clip(self.x, 0.0, 1.0)
        self.g = self.g_max * self.s
        self.b = 1 / (1 + np.exp(-0.062 * post_V) * self.c_Mg / 3.57)

        ## step 4：计算突触后细胞的输入电流（conductance-based）
        current = self.g * self.b * (self.E - post_V)
        return self.g, self.b, current


def make_spike_train(sp_times, run_duration, dt):
    ts = np.arange(0.0, run_duration, dt)
    spikes = np.zeros_like(ts)
    for t in sp_times:
        idx = int(round(t / dt))
        if 0 <= idx < len(spikes):
            spikes[idx] = 1.0
    return ts, spikes


def section_input(values, durations, dt):
    pieces = []
    for value, duration in zip(values, durations):
        pieces.append(np.ones(int(round(duration / dt))) * value)
    return np.concatenate(pieces)


def run_syn(syn_model, title, run_duration=200.0, sp_times=(25, 50, 75, 100, 160), dt=0.1):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    ts, pre_spike = make_spike_train(sp_times, run_duration, dt)
    syn = syn_model()
    delayed_spike = apply_brainpy_delay(pre_spike, syn.delay_step)
    currents = section_input(values=[0.0, 8, 0.0], durations=[130, 1, 69], dt=dt)
    currents = currents[: len(ts)]

    # 运行模拟
    post = create_post_hh(-70.68)
    post_V = np.zeros_like(ts)
    g = np.zeros_like(ts)
    b = np.zeros_like(ts)
    post_input = np.zeros_like(ts)
    for i, t in enumerate(ts):
        g[i], b[i], syn_input = syn.update(delayed_spike[i], post.V[0], t, dt)
        total_input = syn_input + currents[i]
        post_input[i] = total_input
        post.update(total_input, t, dt)
        post_V[i] = post.V[0]

    # 可视化
    fig, axes = plt.subplots(5, 1, figsize=(6.0, 4.5), sharex=True)
    ax = axes[0]
    ax.plot(ts, pre_spike, label="pre.spike")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(ts, post_V, label="post.V")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(ts, g, label="g", color="#d62728")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[3]
    ax.plot(ts, b, label="b", color="blue")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[4]
    ax.plot(ts, post_input, label="PSC", color="y")
    ax.legend(loc="upper right")
    ax.set_xlabel("Time [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_syn(NMDA, title="NMDA Synapse Model")
