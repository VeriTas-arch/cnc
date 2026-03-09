import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class AdEx(bp.dyn.NeuDyn):
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
        name=None,
    ):
        super(AdEx, self).__init__(size=size, name=name)
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

        self.V = bm.Variable(bm.ones(self.num) * V_rest)
        self.w = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

        # 定义积分器
        self.integral = bp.odeint(f=self.derivative, method="exp_auto")

    def dV(self, V, t, w, Iext):
        _tmp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dVdt = (-V + self.V_rest + _tmp - self.R * w + self.R * Iext) / self.tau
        return dVdt

    def dw(self, w, t, V):
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dwdt

    @property
    def derivative(self):
        return bp.JointEq(self.dV, self.dw)

    def update(self, *args, **kwargs):
        _t, _dt = bp.share["t"], bp.share["dt"]
        refractory = (_t - self.t_last_spike) <= self.tau_ref

        V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)
        V = bm.where(refractory, self.V, V)

        spike = V > self.V_th
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.w.value = bm.where(spike, w + self.b, w)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.0


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
    runner = bp.DSRunner(
        neu, monitors=["V", "w", "spike"], inputs=("input", cfg["input_current"])
    )
    runner.predict(500)

    # 为了更直观看到放电，将 spike 时刻的 V 抬升到 20 mV
    v_for_plot = bm.where(runner.mon.spike, 20.0, runner.mon.V)
    # 如果不想抬升，可以直接使用 runner.mon.V 进行绘图
    # v_for_plot = runner.mon.V

    ax.plot(runner.mon.ts, v_for_plot, label="V")
    ax.plot(runner.mon.ts, runner.mon.w, label="w")
    ax.set_title(subplot_titles[mode_name])
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("Value")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
