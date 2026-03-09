import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class LIF(bp.dyn.NeuDyn):
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
        super(LIF, self).__init__(size=size, name=name)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_ref = t_ref

        self.V = bm.Variable(bm.ones(self.num) * V_rest)
        self.input = bm.Variable(bm.zeros(self.num))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

        ...
        # 使用指数欧拉方法进行积分
        self.integral = bp.odeint(f=self.derivative, method="exp_auto")

    # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, t, R, Iext):
        dVdt = (-V + self.V_rest + R * Iext) / self.tau
        return dVdt

    def update(self, *args, **kwargs):
        t, dt = bp.share["t"], bp.share["dt"]
        refractory = (t - self.t_last_spike) <= self.t_ref

        V = self.integral(self.V, t, self.R, self.input, dt=dt)
        V = bm.where(refractory, self.V, V)

        spike = V > self.V_th
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.0


# 在恒定电流输入下，LIF model 以固定频率发放
currents, length = bp.inputs.section_input(
    values=[0.0, 21], durations=[50, 150], return_length=True
)
group = LIF(1)
runner = bp.DSRunner(group, monitors=["V"], inputs=["input", currents, "iter"])
runner.predict(length)
fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
axe[0].plot(runner.mon.ts, runner.mon.V, color="blue")
axe[1].plot(runner.mon.ts, currents, linewidth=2, color="blue")
axe[0].set_ylabel("V")
axe[1].set_ylabel("I")
plt.xlabel("t (ms)")
plt.show()


# LIF 神经元发放率与电流的关系
duration = 1000
input_currents = bm.arange(0, 600, 1)
group = LIF(len(input_currents))
runner = bp.DSRunner(group, monitors=["spike"], inputs=["input", input_currents])
runner.predict(duration)
F = runner.mon.spike.sum(axis=0) / (duration / 1000)
plt.plot(input_currents, F, linewidth=2)
plt.xlabel("Input Current")
plt.ylabel("Spiking Frequency")
plt.show()


# LIF model 的 filter 作用
in_va = bm.arange(0, 100, 0.1)
duration = bm.ones(len(in_va)) * 0.1
value = [40 * bm.sin(i) + 20 for i in in_va]  # 40 * sin(t) + 20
currents, length = bp.inputs.section_input(
    values=value, durations=duration, return_length=True
)
group = LIF(1)
runner = bp.DSRunner(group, monitors=["V"], inputs=["input", currents, "iter"])
runner.predict(length)
fig, axe = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
axe[0].plot(runner.mon.ts, runner.mon.V, color="blue")
axe[1].plot(in_va, currents, linewidth=2, color="blue")
axe[0].set_ylabel("V")
axe[1].set_ylabel("I")
plt.xlabel("t (ms)")
plt.show()
