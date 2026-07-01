import matplotlib.pyplot as plt
import numpy as np


class AdEx:
    def __init__(
        self,
        size,
        V_rest=-70.0,
        V_reset=-68.0,
        V_th=0.0,
        V_T=-50.0,
        delta_T=2.0,
        a=1.0,
        b=2.5,
        R=0.5,
        tau=10.0,
        tau_w=30.0,
        tau_ref=0.0,
    ):
        self.num = int(size)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_T = V_T
        self.delta_T = delta_T
        self.a = a
        self.b = b
        self.tau = tau
        self.tau_w = tau_w
        self.R = R
        self.tau_ref = tau_ref

        self.V = np.ones(self.num) * V_rest
        self.w = np.zeros(self.num)
        self.input = np.zeros(self.num)
        self.spike = np.zeros(self.num, dtype=bool)
        self.t_last_spike = np.ones(self.num) * -1e7
        self.refractory = np.zeros(self.num, dtype=bool)

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

    def dV(self, V, w, Iext):
        _tmp = self.delta_T * self._exp_term(V)
        dVdt = (-V + self.V_rest + _tmp - self.R * w + self.R * Iext) / self.tau
        return dVdt

    def dV_linear(self, V):
        return (-1.0 + self._exp_term(V)) / self.tau

    def dw(self, w, V):
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dwdt

    def dw_linear(self):
        return -1.0 / self.tau_w

    def update(self, input_current, t, dt):
        self.input = np.asarray(input_current) + np.zeros(self.num)
        refractory = (t - self.t_last_spike) <= self.tau_ref

        # 定义积分器
        V = self.V + dt * self._exprel(dt * self.dV_linear(self.V)) * self.dV(
            self.V, self.w, self.input
        )
        w = self.w + dt * self._exprel(dt * self.dw_linear()) * self.dw(
            self.w, self.V
        )
        V = np.where(refractory, self.V, V)

        spike = V > self.V_th
        self.spike = spike
        self.t_last_spike = np.where(spike, t, self.t_last_spike)
        self.V = np.where(spike, self.V_reset, V)
        self.w = np.where(spike, w + self.b, w)
        self.refractory = np.logical_or(refractory, spike)
        self.input[:] = 0.0

    def run(self, input_current, duration, dt=0.1):
        steps = int(duration / dt)
        ts = np.arange(steps) * dt
        Vs = np.zeros((steps, self.num))
        ws = np.zeros((steps, self.num))
        spikes = np.zeros((steps, self.num), dtype=bool)
        for i, t in enumerate(ts):
            self.update(input_current, t, dt)
            Vs[i] = self.V
            ws[i] = self.w
            spikes[i] = self.spike
        return ts, Vs, ws, spikes


# fmt: off
mode_params = {
    "TONIC":      dict(tau=20,  tau_w=30,   a=0,     b=60,  V_reset=-55,  input_current=65),
    "ADAPTION":   dict(tau=20,  tau_w=100,  a=0,     b=5,   V_reset=-55,  input_current=65),
    "INITIAL":    dict(tau=5,   tau_w=100,  a=0.5,   b=7,   V_reset=-51,  input_current=65),
    "BURSTING":   dict(tau=5,   tau_w=100,  a=-0.5,  b=7,   V_reset=-47,  input_current=65),
    "TRANSIENT":  dict(tau=10,  tau_w=100,  a=1,     b=10,  V_reset=-60,  input_current=55),
    "DELAYED":    dict(tau=5,   tau_w=100,  a=-1,    b=5,   V_reset=-60,  input_current=25),
}

mode_order = ["TONIC", "ADAPTION", "INITIAL", "BURSTING", "TRANSIENT", "DELAYED"]

subplot_titles = {
    "TONIC":        "Tonic Spiking",
    "ADAPTION":     "Adaptation",
    "INITIAL":      "Initial Bursting",
    "BURSTING":     "Bursting",
    "TRANSIENT":    "Transient Spiking",
    "DELAYED":      "Delayed Spiking",
}
# fmt: on

def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    for ax, mode_name in zip(axes.flat, mode_order):
        cfg = mode_params[mode_name]
        neu = AdEx(
            1,
            tau=cfg["tau"],
            tau_w=cfg["tau_w"],
            a=cfg["a"],
            b=cfg["b"],
            V_reset=cfg["V_reset"],
        )
        ts, V, w, spike = neu.run(cfg["input_current"], duration=500, dt=0.1)

        # 为了更直观看到放电，将 spike 时刻的 V 抬升到 20 mV
        # v_for_plot = np.where(spike, 20.0, V)
        # 如果不想抬升，可以直接使用 V 进行绘图
        v_for_plot = V

        ax.plot(ts, v_for_plot, label="V")
        ax.plot(ts, w, label="w")
        ax.set_title(subplot_titles[mode_name])
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("Value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
