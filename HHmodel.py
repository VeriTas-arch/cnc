import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np


# 定义神经元模型
class HH(bp.dyn.NeuDyn):

    # 设定神经元参数
    def __init__(
        self,
        size,
        ENa=50.0,
        gNa=120.0,
        EK=-77.0,
        gK=36.0,
        EL=-54.387,
        gL=0.03,
        V_th=20.0,
        C=1.0,
    ):
        super(HH, self).__init__(size=size)
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

        # 定义模型中使用的变量
        self.V = bm.Variable(-70.68 * bm.ones(self.num))
        self.m = bm.Variable(0.0266 * bm.ones(self.num))
        self.h = bm.Variable(0.772 * bm.ones(self.num))
        self.n = bm.Variable(0.235 * bm.ones(self.num))
        self.gNa_ = bm.Variable(0 * bm.ones(self.num))
        self.gK_ = bm.Variable(0 * bm.ones(self.num))

        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # 定义积分函数
        self.intergral = bp.odeint(f=self.derivative, method="exp_auto")

    def dm(self, m, t, V):
        alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
        beta = 4.0 * bm.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 65) / 20.0)
        beta = 1 / (1 + bm.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, t, V):
        alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
        beta = 0.125 * bm.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def dV(self, V, t, h, n, m):
        I_Na = (self.gNa * m**3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n**4.0) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (-I_Na - I_K - I_leak + self.input) / self.C
        return dVdt

    @property
    def derivative(self):
        return bp.JointEq(self.dV, self.dm, self.dh, self.dn)

    # 更新函数：每个时间步都会运行此函数完成变量更新
    def update(self, *args, **kwargs):
        t = bp.share["t"]
        dt = bp.share["dt"]

        # 更新下一时刻变量的值
        V, m, h, n = self.intergral(self.V, self.m, self.h, self.n, t, dt=dt)
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike.value = bm.where(self.spike, t, self.t_last_spike)
        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n

        self.gNa_.value = self.gNa * m**3.0 * h  # 记录钠电导变化
        self.gK_.value = self.gK * n**4.0  # 记录钾电导变化
        self.input[:] = 0.0  # 重置神经元接收到的输入


# 一个简单的示例，观察神经元在10ms的刺激电流作用下的活动
currents, length = bp.inputs.section_input(
    values=[0.0, 10.0, 0.0], durations=[10, 5, 30], return_length=True
)

hh = HH(1)
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel="V (mV)")
plt.plot(runner.mon.ts, np.asarray(currents) - 90)
plt.title("An example of HH model")
plt.legend(["membrane potential", "input current"])
plt.tight_layout()
plt.show()


# 神经元在不同刺激电流强度和时长下的活动
## 1.不同刺激电流强度
currents, length = bp.inputs.section_input(
    values=[0.0, bm.asarray([1.0, 2.0, 4.0, 8.0, 10.0, 15.0]), 0.0],
    durations=[10, 2, 25],
    return_length=True,
)
hh = HH(currents.shape[1])
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

# 可视化
bp.visualize.line_plot(
    runner.mon.ts, runner.mon.V, ylabel="V (mV)", plot_ids=np.arange(currents.shape[1])
)
# 将电流变化画在膜电位变化的下方
plt.plot(runner.mon.ts, np.asarray(currents) - 90)
plt.title("Different current amplitudes")
plt.legend(["I=1.0mA", "I=2.0mA", "I=4.0mA", "I=8.0mA", "I=10.0mA", "I=15.0mA"])
plt.tight_layout()
plt.show()


## 2.不同时长
currents, length = bp.inputs.section_input(
    values=[0.0, 10, 0.0], durations=[10, 50, 25], return_length=True
)
hh = HH(1)
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

# 可视化
bp.visualize.line_plot(runner.mon.ts, runner.mon.V, ylabel="V (mV)")
# 将电流变化画在膜电位变化的下方
plt.plot(runner.mon.ts, np.asarray(currents) - 90)
plt.title("Different current durations")
plt.legend(["V", "I"])
plt.tight_layout()
plt.show()


# 了解动作电位大小和形状不随刺激电流的变化而变化
currents, length = bp.inputs.section_input(
    values=[0.0, bm.asarray([10.0, 15.0, 20.0, 25.0]), 0.0],
    durations=[10, 2, 25],
    return_length=True,
)
hh = HH(currents.shape[1])
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

# 可视化
bp.visualize.line_plot(
    runner.mon.ts, runner.mon.V, ylabel="V (mV)", plot_ids=np.arange(currents.shape[1])
)
# 将电流变化画在膜电位变化的下方
plt.plot(runner.mon.ts, np.asarray(currents) - 90)
plt.title("Action potential amplitude and shape")
plt.legend(["I=10.0mA", "I=15.0mA", "I=20.0mA", "I=25.0mA"])
plt.tight_layout()
plt.show()


# 了解动作电位的不应期
currents, length = bp.inputs.section_input(
    values=[0.0, 10, 0, bm.asarray([10.0, 15.0, 40.0]), 0.0],
    durations=[10, 2, 2, 2, 25],
    return_length=True,
)
hh = HH(currents.shape[1])
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

# 可视化
bp.visualize.line_plot(
    runner.mon.ts, runner.mon.V, ylabel="V (mV)", plot_ids=np.arange(currents.shape[1])
)
# 将电流变化画在膜电位变化的下方
plt.plot(runner.mon.ts, np.asarray(currents) - 90)
plt.title("Refractory period")
plt.tight_layout()
plt.show()


# 了解动作电位发生时电导和门控变量随时间变化规律
currents, length = bp.inputs.section_input(
    values=[0.0, 10, 0.0], durations=[10, 2, 25], return_length=True
)
hh = HH(1)
runner = bp.DSRunner(
    hh, monitors=["V", "m", "h", "n", "gNa_", "gK_"], inputs=["input", currents, "iter"]
)
runner.predict(length)

# 可视化
fig, axe = plt.subplots(3, 1)
axe[0].plot(runner.mon.ts, runner.mon.V, linewidth=2)
axe[0].set_ylabel("V (mV)")
axe[1].plot(runner.mon.ts, runner.mon.gNa_, linewidth=2, color="blue")
axe[1].plot(runner.mon.ts, runner.mon.gK_, linewidth=2, color="red")
axe[1].set_ylabel("Conductance")
axe[1].legend(["gNa", "gK"])
axe[2].plot(runner.mon.ts, runner.mon.m, linewidth=2, color="blue")
axe[2].plot(runner.mon.ts, runner.mon.n, linewidth=2, color="red")
axe[2].plot(runner.mon.ts, runner.mon.h, linewidth=2, color="green")
axe[2].set_ylabel("Channel")
axe[2].legend(["m", "n", "h"])
plt.xlabel("Time (ms)")
plt.tight_layout()
plt.show()
