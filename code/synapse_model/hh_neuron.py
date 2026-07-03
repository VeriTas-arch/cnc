import numpy as np


def _safe_rate(numerator, denominator, fallback):
    value = numerator / denominator
    return np.where(np.abs(denominator) < 1e-7, fallback, value)


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
