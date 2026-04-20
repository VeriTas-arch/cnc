# %%
import brainpy.math as bm
from canns.analyzer.visualization import (PlotConfigs,
                                          energy_landscape_1d_animation)
from canns.models.basic import CANN1D
from canns.task.tracking import (PopulationCoding1D, SmoothTracking1D,
                                 TemplateMatching1D)

# %%
# 群体编码
model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)

# Population coding task
task_pc = PopulationCoding1D(
    cann_instance=model,
    before_duration=10.0,
    after_duration=10.0,
    Iext=0.0,
    duration=20.0,
    time_step=bm.get_dt(),
)

# Get data and run simulation
task_pc.get_data()


# Define simulation step
def run_step(t, inp):
    model.update(inp)
    return model.u.value, model.r.value, model.inp.value


u_pc, r_pc, inp_pc = bm.for_loop(
    run_step, operands=(task_pc.run_steps, task_pc.data), progress_bar=10
)

# Visualize
config_anim = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=100,  # 100 time steps = 1 second of real time
    fps=20,  # 20 frames per second
    title="Energy Landscape Animation - Population Coding",
    xlabel="Neuron Position",
    ylabel="Firing Rate",
    repeat=True,
    show=True,
    save_path=None,  # Set to 'animation.gif' to save
)

# Generate animation
energy_landscape_1d_animation(
    data_sets={"u": (model.x, u_pc), "stimulus": (model.x, inp_pc)}, config=config_anim
)

# %%
# 模板匹配
model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)


# Template matching task
task_tm = TemplateMatching1D(
    cann_instance=model, Iext=1.0, duration=50.0, time_step=bm.get_dt()
)

# Get data and run simulation
task_tm.get_data()


# Define simulation step
def run_step(t, inp):
    model.update(inp)
    return model.u.value, model.r.value, model.inp.value


u_tm, r_tm, inp_tm = bm.for_loop(
    run_step, operands=(task_tm.run_steps, task_tm.data), progress_bar=10
)

# Visualize
config_anim = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=100,  # 100 time steps = 1 second of real time
    fps=20,  # 20 frames per second
    title="Energy Landscape Animation - Template Matching",
    xlabel="Neuron Position",
    ylabel="Firing Rate",
    repeat=True,
    show=True,
    save_path=None,  # Set to 'animation.gif' to save
)

# Generate animation
energy_landscape_1d_animation(
    data_sets={"u": (model.x, u_tm), "stimulus": (model.x, inp_tm)}, config=config_anim
)

# %%
# 平滑追踪
model = CANN1D(num=256, tau=1.0, k=8.1, a=0.5, A=10, J0=4.0)


# Smooth tracking task
task_st = SmoothTracking1D(
    cann_instance=model, Iext=[-2.0, 2.0], duration=[50.0], time_step=bm.get_dt()
)

# Get data and run simulation
task_st.get_data()


# Define simulation step
def run_step(t, inp):
    model.update(inp)
    return model.u.value, model.r.value, model.inp.value


u_st, r_st, inp_st = bm.for_loop(
    run_step, operands=(task_st.run_steps, task_st.data), progress_bar=10
)

# Visualize
config_anim = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=100,  # 100 time steps = 1 second of real time
    fps=20,  # 20 frames per second
    title="Energy Landscape Animation - Smooth Tracking",
    xlabel="Neuron Position",
    ylabel="Firing Rate",
    repeat=True,
    show=True,
    save_path=None,  # Set to 'animation.gif' to save
)

# Generate animation
energy_landscape_1d_animation(
    data_sets={"u": (model.x, u_st), "stimulus": (model.x, inp_st)}, config=config_anim
)
