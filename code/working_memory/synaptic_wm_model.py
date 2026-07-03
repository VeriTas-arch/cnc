import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# fmt: off
alpha = 1.5
J_EE = 8.0          # the connection strength in each excitatory neural clusters
J_IE = 1.75         # Synaptic efficacy E -> I
J_EI = 1.1          # Synaptic efficacy I -> E
tau_f = 1.5         # time constant of STF [s]
tau_d = 0.3         # time constant of STD [s]
U = 0.3             # minimum STF value
tau = 0.008         # time constant of firing rate of the excitatory neurons [s]
tau_I = tau         # time constant of firing rate of the inhibitory neurons

Ib = 3.3    # background input and external input
Iinh = 0.0  # the background input of inhibtory neuron

cluster_num = 16    # the number of the clusters

# the parameters of external input
stimulus_num = 5
Iext_train = 225    # the strength of the external input
Ts_interval = 0.07  # the time interval between the consequent external input [s]
Ts_duration = 0.03  # the time duration of the external input [s]
duration = 2.500    # [s]
# fmt: on


# Working-memory network with excitatory clusters and one inhibitory pool
class WM:
    def __init__(self):
        # variables
        self.u = np.ones(cluster_num) * U
        self.x = np.ones(cluster_num)
        self.h = np.zeros(cluster_num)
        self.r = self.log(self.h)
        self.input = np.zeros(cluster_num)
        self.inh_h = np.zeros(1)
        self.inh_r = self.log(self.inh_h)

    def log(self, h):
        return alpha * np.log(1.0 + np.exp(h / alpha))

    def update(self, current_input, dt):
        self.input[:] = current_input
        du = (U - self.u) / tau_f + U * (1 - self.u) * self.r
        self.u += du * dt
        uxr = self.u * self.x * self.r
        dx = (1 - self.x) / tau_d - uxr
        dh = (-self.h + J_EE * uxr - J_EI * self.inh_r + self.input + Ib) / tau
        self.h += dh * dt
        dhi = (-self.inh_h + J_IE * np.sum(self.r) + Iinh) / tau_I
        self.x += dx * dt
        self.inh_h += dhi * dt
        self.r[:] = self.log(self.h)
        self.inh_r[:] = self.log(self.inh_h)

        # 更新外界输入
        self.input[:] = 0.0

    def run(self, inputs, dt):
        ts = np.arange(inputs.shape[0]) * dt
        records = {
            key: np.zeros((inputs.shape[0], cluster_num))
            for key in ["u", "x", "r", "h"]
        }
        for i, current_input in enumerate(inputs):
            self.update(current_input, dt)
            records["u"][i] = self.u
            records["x"][i] = self.x
            records["r"][i] = self.r
            records["h"][i] = self.h
        return ts, records


dt = 0.0001  # [s]
# the external input
I_inputs = np.zeros((int(duration / dt), cluster_num))
for i in range(stimulus_num):
    t_start = (Ts_interval + Ts_duration) * i + Ts_interval
    t_end = t_start + Ts_duration
    idx_start, idx_end = int(t_start / dt), int(t_end / dt)
    I_inputs[idx_start:idx_end, i] = Iext_train

# running
ts, mon = WM().run(I_inputs, dt)

# visualization
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

fig, axes = plt.subplots(5, 1, figsize=(2, 12))
for i in range(stimulus_num):
    axes[0].plot(ts, mon["r"][:, i], colors[i], label="Cluster-{}".format(i))
axes[0].set_ylabel("$r (Hz)$")
axes[0].legend(loc="right")

hist_Jux = J_EE * mon["u"] * mon["x"]
for i in range(stimulus_num):
    axes[1].plot(ts, hist_Jux[:, i], colors[i])
axes[1].set_ylabel("$J_{EE}ux$")

for i in range(stimulus_num):
    axes[2].plot(ts, mon["u"][:, i], colors[i])
axes[2].set_ylabel("u")

for i in range(stimulus_num):
    axes[3].plot(ts, mon["x"][:, i], colors[i])
axes[3].set_ylabel("x")

for i in range(stimulus_num):
    axes[4].plot(ts, mon["r"][:, i], colors[i])
axes[4].set_ylabel("h")
axes[4].set_xlabel("time [s]")

plt.show()
