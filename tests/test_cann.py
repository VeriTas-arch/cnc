"""
reference: <https://routhleck.com/canns/zh/3_full_detail_tutorials/01_cann_modeling/04_parameter_effects.html>
"""

import brainpy.math as bm
from canns.models.basic import CANN1D_SFA
from canns.task.tracking import SmoothTracking1D
from canns.analyzer.visualization import PlotConfigs, energy_landscape_1d_animation

# Setup environment
bm.set_dt(0.1)


def run_experiment(model, title="", save_path=None):
    """Run standard experiment and visualize results"""

    # Create smooth tracking task
    task = SmoothTracking1D(
        cann_instance=model,
        Iext=[0.0, 3.0, 2.0, 3.0],
        duration=[20.0, 20.0, 10.0],
        time_step=0.1,
    )

    # Get task data
    task.get_data()

    def run_step(t, inp):
        model.update(inp)
        return model.u.value, model.r.value, model.inp.value

    u_history, r_history, input_history = bm.for_loop(
        run_step, operands=(task.run_steps, task.data), progress_bar=10
    )

    # Configure and create visualization
    config = PlotConfigs.energy_landscape_1d_animation(
        time_steps_per_second=100,
        fps=20,
        title=title,
        xlabel="Position",
        ylabel="Firing Rate",
        repeat=True,
        show=True,
        save_path=save_path,
    )

    energy_landscape_1d_animation(
        data_sets={"u": (model.x, u_history), "stimulus": (model.x, input_history)},
        config=config,
    )

    return r_history


# Test different SFA time constants
for tau_v_val in [20.0, 50.0, 100.0]:
    model = CANN1D_SFA(
        num=256, tau=1.0, tau_v=tau_v_val, k=8.1, a=0.3, A=0.2, J0=1.0, m=0.3
    )
    run_experiment(model, title=f"SFA Time Constant: tau_v={tau_v_val}", save_path=None)
