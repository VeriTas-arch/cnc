import matplotlib.pyplot as plt
import numpy as np


def section_input(values, durations, dt=0.1, return_length=False):
    currents = []
    for value, duration in zip(values, durations):
        steps = int(duration / dt)
        currents.append(np.full(steps, float(value)))
    currents = np.concatenate(currents, axis=0)
    return (currents, len(currents)) if return_length else currents


class ExpIF:
    def __init__(
        self,
        size,
        V_rest=-65.0,
        V_reset=-68.0,
        V_th=20.0,
        V_T=-59.9,
        delta_T=2.0,
        R=1.0,
        tau=10.0,
        tau_ref=2.0,
    ):
        self.num = int(size)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_T = V_T
        self.delta_T = delta_T
        self.R = R
        self.tau = tau
        self.tau_ref = tau_ref

        self.V = np.zeros(self.num) + self.V_rest
        self.input = np.zeros(self.num)
        self.t_last_spike = np.ones(self.num) * -1e7
        self.refractory = np.zeros(self.num, dtype=bool)
        self.spike = np.zeros(self.num, dtype=bool)

    def _exp_term(self, V):
        exp_arg = (V - self.V_T) / self.delta_T
        return np.exp(np.clip(exp_arg, None, 50.0))

    def _exprel(self, x):
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        small = np.abs(x) < 1e-8
        out[small] = 1.0 + x[small] / 2.0
        out[~small] = np.expm1(np.clip(x[~small], None, 50.0)) / x[~small]
        return out

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, Iext):
        exp_v = self.delta_T * self._exp_term(V)
        dvdt = (-(V - self.V_rest) + exp_v + self.R * Iext) / self.tau
        return dvdt

    def derivative_linear(self, V):
        return (-1.0 + self._exp_term(V)) / self.tau

    def update(self, input_current, t, dt):
        self.input = np.asarray(input_current) + np.zeros(self.num)
        refractory = (t - self.t_last_spike) <= self.tau_ref

        V = self.V + dt * self._exprel(dt * self.derivative_linear(self.V)) * self.derivative(
            self.V, self.input
        )
        V = np.where(refractory, self.V_reset, V)

        spike = V > self.V_th
        self.spike = spike
        self.t_last_spike = np.where(spike, t, self.t_last_spike)
        self.V = np.where(spike, self.V_th, V)
        self.refractory = np.logical_or(refractory, spike)
        self.input[:] = 0.0

    def run(self, inputs, dt=0.01):
        currents = np.asarray(inputs)
        if currents.ndim == 1:
            currents = currents[:, None]
        ts = np.arange(currents.shape[0]) * dt
        Vs = np.zeros((currents.shape[0], self.num))
        for i, t in enumerate(ts):
            self.update(currents[i], t, dt)
            Vs[i] = self.V
        return ts, Vs


def main():
    currents, length = section_input(
        values=[0.0, 15, 0], durations=[10, 10, 10], return_length=True, dt=0.01
    )
    delta_ts = [0.02, 1, 5]
    colors = ["blue", "red", "black"]
    fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})

    for delta_t, color in zip(delta_ts, colors):
        group = ExpIF(1, delta_T=delta_t)
        ts, V = group.run(currents, dt=0.01)
        axe[0].plot(ts, V, color=color, label=f"delta_T={delta_t}")
        axe[1].plot(ts, currents, linewidth=2, color=color)

    axe[0].legend()
    axe[0].set_ylabel("V")
    axe[1].set_ylabel("I")
    plt.xlabel("t (ms)")
    plt.show()


if __name__ == "__main__":
    main()
