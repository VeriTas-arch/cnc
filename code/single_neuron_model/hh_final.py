"""
在基础版代码的基础上，增加了课后思考题中提到的内容，即 m shift 后神经元发放的变化。
"""

from hh import HH, _safe_rate, line_plot, np, plt, section_input

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

    # 更改模型中参数后，观察 m shift 后神经元发放的变化
    class ShiftedHH(HH):
        def __init__(self, size, mshift=0.0, **kwargs):
            super().__init__(size=size, **kwargs)
            self.mshift = mshift

        def dm(self, m, V):
            input = V + self.mshift  # 将输入电压进行平移
            denominator = 1 - np.exp(-(input + 40) / 10)
            alpha = _safe_rate(0.1 * (input + 40), denominator, 1.0)
            beta = 4.0 * np.exp(-(input + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m
            return dmdt

    currents, length = section_input(
        values=[0.0, 10, 0.0], durations=[10, 2, 25], return_length=True
    )
    hh = ShiftedHH(1, mshift=-10.0)
    ts, mon = hh.run(currents)
    # 可视化
    line_plot(ts, mon["V"], ylabel="V (mV)")
    # 将电流变化画在膜电位变化的下方
    plt.plot(ts, np.asarray(currents) - 90)
    plt.title("HH model with m shift")
    plt.tight_layout()
    plt.show()
