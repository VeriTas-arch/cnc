"""
基础版代码，对应课件中的示例。
"""

import matplotlib.pyplot as plt
import numpy as np


def section_input(values, durations, dt=0.01, return_length=False):
    width = 1
    for value in values:
        value = np.asarray(value, dtype=float)
        if value.ndim > 0:
            width = max(width, value.size)

    currents = []
    for value, duration in zip(values, durations):
        steps = int(duration / dt)
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            currents.append(np.full((steps, width), float(value)))
        else:
            currents.append(np.repeat(value.reshape(1, -1), steps, axis=0))
    currents = np.concatenate(currents, axis=0)
    if width == 1:
        currents = currents.reshape(-1)
    return (currents, len(currents)) if return_length else currents


def _safe_rate(numerator, denominator, fallback):
    value = numerator / denominator
    return np.where(np.abs(denominator) < 1e-7, fallback, value)


def line_plot(ts, values, ylabel=None, plot_ids=None):
    values = np.asarray(values)
    if values.ndim == 1 or values.shape[1] == 1:
        plt.plot(ts, values.reshape(-1))
    else:
        ids = np.arange(values.shape[1]) if plot_ids is None else plot_ids
        for i in ids:
            plt.plot(ts, values[:, i])
    if ylabel is not None:
        plt.ylabel(ylabel)


# 定义神经元模型
class HH:

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
        self.num = int(size)
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

        # 定义模型中使用的变量
        self.V = -70.68 * np.ones(self.num)
        self.m = 0.0266 * np.ones(self.num)
        self.h = 0.772 * np.ones(self.num)
        self.n = 0.235 * np.ones(self.num)
        self.gNa_ = 0 * np.ones(self.num)
        self.gK_ = 0 * np.ones(self.num)

        self.input = np.zeros(self.num)
        self.spike = np.zeros(self.num, dtype=bool)
        self.t_last_spike = np.ones(self.num) * -1e7

    def _exprel(self, x):
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        small = np.abs(x) < 1e-8
        out[small] = 1.0 + x[small] / 2.0
        out[~small] = np.expm1(np.clip(x[~small], None, 50.0)) / x[~small]
        return out

    def _linearize(self, func, value):
        eps = 1e-6 * np.maximum(1.0, np.abs(value))
        return (func(value + eps) - func(value - eps)) / (2.0 * eps)

    def dm(self, m, V):
        denominator = 1 - np.exp(-(V + 40) / 10)
        alpha = _safe_rate(0.1 * (V + 40), denominator, 1.0)
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dh(self, h, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.0)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, V):
        denominator = 1 - np.exp(-(V + 55) / 10)
        alpha = _safe_rate(0.01 * (V + 55), denominator, 0.1)
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def dV(self, V, h, n, m, input_current):
        I_Na = (self.gNa * m**3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n**4.0) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (-I_Na - I_K - I_leak + input_current) / self.C
        return dVdt

    def derivative(self, V, m, h, n, input_current):
        return (
            self.dV(V, h, n, m, input_current),
            self.dm(m, V),
            self.dh(h, V),
            self.dn(n, V),
        )

    # 更新函数：每个时间步都会运行此函数完成变量更新
    def update(self, input_current, t, dt):
        self.input = np.asarray(input_current) + np.zeros(self.num)

        # 更新下一时刻变量的值
        dV, dm, dh, dn = self.derivative(self.V, self.m, self.h, self.n, self.input)
        V = (
            self.V
            + dt
            * self._exprel(
                dt
                * self._linearize(
                    lambda V: self.dV(V, self.h, self.n, self.m, self.input), self.V
                )
            )
            * dV
        )
        m = (
            self.m
            + dt
            * self._exprel(dt * self._linearize(lambda m: self.dm(m, self.V), self.m))
            * dm
        )
        h = (
            self.h
            + dt
            * self._exprel(dt * self._linearize(lambda h: self.dh(h, self.V), self.h))
            * dh
        )
        n = (
            self.n
            + dt
            * self._exprel(dt * self._linearize(lambda n: self.dn(n, self.V), self.n))
            * dn
        )

        self.spike = np.logical_and(self.V < self.V_th, V >= self.V_th)
        self.t_last_spike = np.where(self.spike, t, self.t_last_spike)
        self.V = V
        self.m = m
        self.h = h
        self.n = n

        self.gNa_ = self.gNa * m**3.0 * h  # 记录钠电导变化
        self.gK_ = self.gK * n**4.0  # 记录钾电导变化
        self.input[:] = 0.0  # 重置神经元接收到的输入

    def run(self, currents, dt=0.01):
        currents = np.asarray(currents, dtype=float)
        if currents.ndim == 1:
            currents = currents[:, None]
        ts = np.arange(currents.shape[0]) * dt
        records = {
            "V": np.zeros((currents.shape[0], self.num)),
            "m": np.zeros((currents.shape[0], self.num)),
            "h": np.zeros((currents.shape[0], self.num)),
            "n": np.zeros((currents.shape[0], self.num)),
            "gNa_": np.zeros((currents.shape[0], self.num)),
            "gK_": np.zeros((currents.shape[0], self.num)),
        }
        for i, t in enumerate(ts):
            self.update(currents[i], t, dt)
            records["V"][i] = self.V
            records["m"][i] = self.m
            records["h"][i] = self.h
            records["n"][i] = self.n
            records["gNa_"][i] = self.gNa_
            records["gK_"][i] = self.gK_
        return ts, records


if __name__ == "__main__":
    # 一个简单的示例，观察神经元在10ms的刺激电流作用下的活动
    currents, length = section_input(
        values=[0.0, 10.0, 0.0], durations=[10, 5, 30], return_length=True
    )

    hh = HH(1)
    ts, mon = hh.run(currents)

    line_plot(ts, mon["V"], ylabel="V (mV)")
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("An example of HH model")
    plt.legend(["membrane potential", "input current"])
    plt.tight_layout()
    plt.show()

    # 神经元在不同刺激电流强度和时长下的活动
    ## 1.不同刺激电流强度
    currents, length = section_input(
        values=[0.0, np.asarray([1.0, 2.0, 4.0, 8.0, 10.0, 15.0]), 0.0],
        durations=[10, 2, 25],
        return_length=True,
    )
    hh = HH(currents.shape[1])
    ts, mon = hh.run(currents)

    # 可视化
    line_plot(ts, mon["V"], ylabel="V (mV)", plot_ids=np.arange(currents.shape[1]))
    # 将电流变化画在膜电位变化的下方
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("Different current amplitudes")
    plt.legend(["I=1.0mA", "I=2.0mA", "I=4.0mA", "I=8.0mA", "I=10.0mA", "I=15.0mA"])
    plt.tight_layout()
    plt.show()

    ## 2.不同时长
    currents, length = section_input(
        values=[0.0, 10, 0.0], durations=[10, 50, 25], return_length=True
    )
    hh = HH(1)
    ts, mon = hh.run(currents)

    # 可视化
    line_plot(ts, mon["V"], ylabel="V (mV)")
    # 将电流变化画在膜电位变化的下方
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("Different current durations")
    plt.legend(["V", "I"])
    plt.tight_layout()
    plt.show()

    # 了解动作电位大小和形状不随刺激电流的变化而变化
    currents, length = section_input(
        values=[0.0, np.asarray([10.0, 15.0, 20.0, 25.0]), 0.0],
        durations=[10, 2, 25],
        return_length=True,
    )
    hh = HH(currents.shape[1])
    ts, mon = hh.run(currents)

    # 可视化
    line_plot(ts, mon["V"], ylabel="V (mV)", plot_ids=np.arange(currents.shape[1]))
    # 将电流变化画在膜电位变化的下方
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("Action potential amplitude and shape")
    plt.legend(["I=10.0mA", "I=15.0mA", "I=20.0mA", "I=25.0mA"])
    plt.tight_layout()
    plt.show()

    # 了解动作电位的不应期
    currents, length = section_input(
        values=[0.0, 10, 0, np.asarray([10.0, 15.0, 40.0]), 0.0],
        durations=[10, 2, 2, 2, 25],
        return_length=True,
    )
    hh = HH(currents.shape[1])
    ts, mon = hh.run(currents)

    # 可视化
    line_plot(ts, mon["V"], ylabel="V (mV)", plot_ids=np.arange(currents.shape[1]))
    # 将电流变化画在膜电位变化的下方
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("Refractory period")
    plt.tight_layout()
    plt.show()

    # 了解动作电位发生时电导和门控变量随时间变化规律
    currents, length = section_input(
        values=[0.0, 10, 0.0], durations=[10, 2, 25], return_length=True
    )
    hh = HH(1)
    ts, mon = hh.run(currents)

    # 可视化
    fig, axe = plt.subplots(3, 1)
    axe[0].plot(ts, mon["V"], linewidth=2)
    axe[0].set_ylabel("V (mV)")
    axe[1].plot(ts, mon["gNa_"], linewidth=2, color="blue")
    axe[1].plot(ts, mon["gK_"], linewidth=2, color="red")
    axe[1].set_ylabel("Conductance")
    axe[1].legend(["gNa", "gK"])
    axe[2].plot(ts, mon["m"], linewidth=2, color="blue")
    axe[2].plot(ts, mon["n"], linewidth=2, color="red")
    axe[2].plot(ts, mon["h"], linewidth=2, color="green")
    axe[2].set_ylabel("Channel")
    axe[2].legend(["m", "n", "h"])
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()
