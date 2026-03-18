import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class Delta(bp.dyn.SynConn):
    def __init__(self, pre, post, conn, g_max=1.0, delay_step=0, E=0.0):
        super().__init__(pre=pre, post=post, conn=conn)
        self.g_max = g_max
        self.delay_step = delay_step  # 控制突触前神经元产生的 spike delay 时间
        self.E = E

        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)

        # 使用连接矩阵聚合事件，避免 CPU 下 pre2post 事件算子对 numba 的依赖
        self.conn_mat = bm.asarray(self.conn.require("conn_mat"), dtype=bm.float_)

    def update(self):
        # 取出延迟了 delay_step 时间步长的突触前脉冲信号
        delayed_pre_spike = self.delay(self.delay_step)
        # 根据最新的突触前脉冲信号更新延迟变量
        self.delay.update(self.pre.spike)
        # 根据连接矩阵计算各个突触后神经元收到的信号强度
        pre_sp = bm.asarray(delayed_pre_spike, dtype=bm.float_)
        post_sp = bm.matmul(pre_sp, self.conn_mat) * self.g_max
        self.g.value = post_sp
        # 将 Delta 突触产生的电流累加到突触后神经元输入
        self.post.input += self.g


"""
假如要让 5 个突触前神经元在 20ms 时刻同时产生一个 spike，可以定义如下：

neu1 = bp.neurons.SpikeTimeGroup(
    5,  # 神经元数量
    times=[20, 20, 20, 20, 20],  # 每个 spike 的时间点
    indices=[0, 1, 2, 3, 4],  # 每个 spike 对应的神经元索引，0-4 分别对应 5 个神经元
)

另一个例子：

neu1 = bp.neurons.SpikeTimeGroup(
    2, times=[20, 30, 60, 70, 100, 100, 140, 180], indices=[0, 1, 0, 1, 0, 0, 1, 0]
)
"""
neu1 = bp.neurons.SpikeTimeGroup(
    1,  # 突触前神经元，第一个参数为神经元数量
    times=[20, 60, 100, 140, 180],
    indices=[0, 0, 0, 0, 0],
)
neu2 = bp.neurons.HH(1, V_initializer=bp.initialize.Constant(-70.68))  # 突触后神经元
syn1 = Delta(
    neu1, neu2, conn=bp.connect.All2All(), g_max=2.0
)  # All2All 意味着每个突触前神经元都连接到每个突触后神经元
net = bp.DynSysGroup(pre=neu1, syn=syn1, post=neu2)

runner = bp.DSRunner(net, monitors=["pre.spike", "post.V", "syn.g"])
runner.predict(200)

# 可视化
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6.0)
ax = fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon["pre.spike"], label="pre.spike")
plt.legend(loc="upper right")
plt.title("Delta Synapse Model (Current-Based)")
plt.xticks([])
ax = fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon["syn.g"], label="g", color="#d62728")
plt.legend(loc="upper right")
plt.xticks([])
ax = fig.add_subplot(gs[2, 0])
plt.plot(runner.mon.ts, runner.mon["post.V"], label="post.V")
plt.legend(loc="upper right")
plt.xlabel("t (ms)")
plt.show()
