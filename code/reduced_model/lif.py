import matplotlib.pyplot as plt
import numpy as np


def section_input(values, durations, dt=0.1, return_length=False):
    currents = []
    for value, duration in zip(values, durations):
        steps = int(duration / dt)
        value = np.asarray(value)
        if value.ndim == 0:
            currents.append(np.full(steps, float(value)))
        else:
            currents.append(np.repeat(value.reshape(1, -1), steps, axis=0))
    currents = np.concatenate(currents, axis=0)
    return (currents, len(currents)) if return_length else currents


class LIF:
    def __init__(
        self,
        size,
        V_rest=0.0,
        V_reset=-5.0,
        V_th=20.0,
        R=1.0,
        tau=10.0,
        t_ref=5.0,
        name=None,
    ):
        self.num = int(size)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_ref = t_ref

        self.V = np.ones(self.num) * V_rest
        self.input = np.zeros(self.num)
        self.t_last_spike = np.ones(self.num) * -1e7
        self.refractory = np.zeros(self.num, dtype=bool)
        self.spike = np.zeros(self.num, dtype=bool)

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, R, Iext):
        dVdt = (-V + self.V_rest + R * Iext) / self.tau
        return dVdt

    def update(self, input_current, t, dt):
        self.input = np.asarray(input_current) + np.zeros(self.num)
        refractory = (t - self.t_last_spike) <= self.t_ref

        # 使用指数欧拉方法进行积分
        target = self.V_rest + self.R * self.input
        V = target + (self.V - target) * np.exp(-dt / self.tau)
        V = np.where(refractory, self.V, V)

        spike = V > self.V_th
        self.spike = spike
        self.t_last_spike = np.where(spike, t, self.t_last_spike)
        self.V = np.where(spike, self.V_reset, V)
        self.refractory = np.logical_or(refractory, spike)
        self.input[:] = 0.0

    def run(self, inputs, duration=None, dt=0.1):
        if np.isscalar(inputs):
            steps = int(duration / dt)
            currents = np.full((steps, self.num), float(inputs))
        else:
            currents = np.asarray(inputs)
            if currents.ndim == 1 and self.num == 1:
                currents = currents[:, None]
            elif currents.ndim == 1:
                steps = int(duration / dt)
                currents = np.repeat(currents.reshape(1, -1), steps, axis=0)

        ts = np.arange(currents.shape[0]) * dt
        Vs = np.zeros((currents.shape[0], self.num))
        spikes = np.zeros((currents.shape[0], self.num), dtype=bool)
        for i, t in enumerate(ts):
            self.update(currents[i], t, dt)
            Vs[i] = self.V
            spikes[i] = self.spike
        return ts, Vs, spikes


# 在恒定电流输入下，LIF model 以固定频率发放
currents, length = section_input(
    values=[0.0, 21], durations=[50, 150], return_length=True
)
group = LIF(1)
ts, V, _ = group.run(currents, dt=0.1)
fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
axe[0].plot(ts, V, color="blue")
axe[1].plot(ts, currents, linewidth=2, color="blue")
axe[0].set_ylabel("V")
axe[1].set_ylabel("I")
plt.xlabel("t (ms)")
plt.show()


# LIF 神经元发放率与电流的关系
duration = 1000
input_currents = np.arange(0, 600, 1)
group = LIF(len(input_currents))
_, _, spikes = group.run(input_currents, duration=duration, dt=0.1)
F = spikes.sum(axis=0) / (duration / 1000)
plt.plot(input_currents, F, linewidth=2)
plt.xlabel("Input Current")
plt.ylabel("Spiking Frequency")
plt.show()


# LIF model 的 filter 作用
in_va = np.arange(0, 100, 0.1)
duration = np.ones(len(in_va)) * 0.1
value = [40 * np.sin(i) + 20 for i in in_va]  # 40 * sin(t) + 20
currents, length = section_input(
    values=value, durations=duration, return_length=True
)
group = LIF(1)
ts, V, _ = group.run(currents, dt=0.1)
fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
axe[0].plot(ts, V, color="blue")
axe[1].plot(in_va, currents, linewidth=2, color="blue")
axe[0].set_ylabel("V")
axe[1].set_ylabel("I")
plt.xlabel("t (ms)")
plt.show()
