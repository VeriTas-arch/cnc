import numpy as np

from hh_neuron import HH


def apply_brainpy_delay(spikes, delay_step):
    # BrainPy 的 LengthDelay 在当前 DynSysGroup 更新/监测顺序下会让 g 比 pre.spike 晚两个 dt。
    delay_len = int(delay_step) + 2
    delayed = np.roll(spikes, delay_len)
    if delay_len:
        delayed[:delay_len] = 0.0
    return delayed


def create_post_hh(initial_v=-70.68, size=1):
    post = HH(size)
    post.V[:] = initial_v
    return post
