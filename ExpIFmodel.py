import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class ExpIF(bp.dyn.NeuDyn):
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
        name=None,
    ):
        super(ExpIF, self).__init__(size=size, name=name)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_T = V_T
        self.delta_T = delta_T
        self.R = R
        self.tau = tau
        self.tau_ref = tau_ref

        self.V = bm.Variable(bm.zeros(self.num) + self.V_rest)
        self.input = bm.Variable(bm.zeros(self.num))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

        self.integral = bp.odeint(f=self.derivative, method="exp_auto")

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, t, Iext):
        exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dvdt = (-(V - self.V_rest) + exp_v + self.R * Iext) / self.tau
        return dvdt

    def update(self, *args, **kwargs):
        t, dt = bp.share["t"], bp.share["dt"]
        refractory = (t - self.t_last_spike) <= self.tau_ref

        V = self.integral(self.V, t, self.input, dt=dt)
        V = bm.where(refractory, self.V_reset, V)

        spike = V > self.V_th
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_th, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.0


currents, length = bp.inputs.section_input(
    values=[0.0, 15, 0], durations=[10, 10, 10], return_length=True, dt=0.01
)
delta_ts = [0.02, 1, 5]
colors = ["blue", "red", "black"]
fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})

for delta_t, color in zip(delta_ts, colors):
    group = ExpIF(1, delta_T=delta_t)
    runner = bp.DSRunner(
        group, monitors=["V"], inputs=["input", currents, "iter"], dt=0.01
    )
    runner.predict(length)
    axe[0].plot(runner.mon.ts, runner.mon.V, color=color, label=f"delta_T={delta_t}")
    axe[1].plot(runner.mon.ts, currents, linewidth=2, color=color)

axe[0].legend()
axe[0].set_ylabel("V")
axe[1].set_ylabel("I")
plt.xlabel("t (ms)")
plt.show()
