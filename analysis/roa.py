import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models import rimless_wheel as model

def simulate_until_converged(initial_state,params,timestep,fixed_point=None,
                             max_sim_time=30.0,convergence_tolerance=1e-2,n_consecutive=5):
    current_time=0.0
    current_state=np.asarray(initial_state,dtype=float).copy()
    impact_angular_velocities=[]
    state_history=[]

    while current_time<max_sim_time:
        dt=min(timestep,max_sim_time-current_time)
        current_state,impacted,impact_state=model.step_with_impact(current_state,params,dt)
        current_time+=dt

        if impacted:
            impact_angular_velocities.append(impact_state[1])

            if fixed_point is not None and len(impact_angular_velocities)>=n_consecutive:
                recent=impact_angular_velocities[-n_consecutive:]

                if np.max(np.abs(np.array(recent)-fixed_point))<convergence_tolerance:
                    return "limit cycle",current_time

        else:
            state_history.append(current_state.copy())

            if fixed_point is None and len(state_history)>=n_consecutive:
                recent=np.array(state_history[-n_consecutive:])

                if np.max(np.ptp(recent,axis=0))<convergence_tolerance:
                    return "equilibrium",current_time

    return "outside",current_time

def build_roa_grid(params,theta_bounds,timestep,fixed_point=None,
                   number_of_bounds=25,angular_velocity_bounds=(-10,10)):
    theta_range=np.linspace(theta_bounds[0],theta_bounds[1],number_of_bounds)
    angular_velocity_range=np.linspace(angular_velocity_bounds[0],angular_velocity_bounds[1],number_of_bounds)
    all_status=np.empty((number_of_bounds,number_of_bounds),dtype=object)

    for i,angular_velocity_0 in enumerate(angular_velocity_range):
        for j,theta_0 in enumerate(theta_range):
            initial_state=np.array([theta_0,angular_velocity_0])

            all_status[i,j]=simulate_until_converged(
                initial_state,
                params,
                timestep,
                fixed_point=fixed_point
            )[0]

    return theta_range,angular_velocity_range,all_status

def _plot_vector_field(theta_range,theta_dot_range,model_dynamics,params,quiver_density=15):
    quiver_theta=np.linspace(theta_range[0],theta_range[-1],quiver_density)
    quiver_theta_dot=np.linspace(theta_dot_range[0],theta_dot_range[-1],quiver_density)
    grid_theta,grid_theta_dot=np.meshgrid(quiver_theta,quiver_theta_dot)
    theta_derivative=np.zeros_like(grid_theta)
    theta_dot_derivative=np.zeros_like(grid_theta_dot)

    for i in range(grid_theta.shape[0]):
        for j in range(grid_theta.shape[1]):
            state=np.array([grid_theta[i,j],grid_theta_dot[i,j]])
            derivative=model_dynamics(0,state,params)
            theta_derivative[i,j]=derivative[0]
            theta_dot_derivative[i,j]=derivative[1]

    plt.quiver(grid_theta,grid_theta_dot,theta_derivative,theta_dot_derivative,color="gray",alpha=0.5)

def _break_trajectory_at_resets(trajectory,params):
    angle_between_spokes=2*np.pi/params["number_spokes"]
    jumps=np.abs(np.diff(trajectory[:,0]))>0.5*angle_between_spokes
    theta_plot=np.insert(trajectory[:,0],np.where(jumps)[0]+1,np.nan)
    theta_dot_plot=np.insert(trajectory[:,1],np.where(jumps)[0]+1,np.nan)

    return theta_plot,theta_dot_plot

def plot_roa_map(theta_range,theta_dot_range,results,trajectory,model_dynamics,params,
                 fixed_point=None,theta_post_reset=None,timestep=1e-2,
                 title="Region of Attraction map"):
    plt.figure(figsize=(10,6))

    color_matrix=np.zeros(results.shape)
    color_matrix[results=="limit cycle"]=1.0

    cmap=plt.colormaps["RdYlGn"].resampled(2)

    plt.imshow(
        color_matrix,
        extent=[
            theta_range[0],
            theta_range[-1],
            theta_dot_range[0],
            theta_dot_range[-1]
        ],
        origin="lower",
        cmap=cmap,
        alpha=0.35,
        aspect="auto"
    )

    _plot_vector_field(
        theta_range,
        theta_dot_range,
        model_dynamics,
        params
    )

    theta_plot,theta_dot_plot=_break_trajectory_at_resets(
        trajectory,
        params
    )

    trajectory_line,=plt.plot(
        theta_plot,
        theta_dot_plot,
        color="black",
        linewidth=1,
        label="trajectory"
    )

    handles=[trajectory_line]

    if fixed_point is not None and theta_post_reset is not None:
        from analysis import poincare

        limit_cycle=poincare.simulate_one_step_full(
            fixed_point,
            theta_post_reset,
            params,
            timestep=timestep
        )

        theta_cycle,theta_dot_cycle=_break_trajectory_at_resets(
            limit_cycle,
            params
        )

        cycle_line,=plt.plot(
            theta_cycle,
            theta_dot_cycle,
            color="red",
            linewidth=4,
            label="limit cycle"
        )

        angle_between_spokes=2*np.pi/params["number_spokes"]
        theta_trigger=theta_post_reset+angle_between_spokes

        reset_line,=plt.plot(
            [theta_trigger,theta_post_reset],
            [limit_cycle[-1,1],fixed_point],
            color="red",
            linewidth=4
        )

        fixed_point_marker,=plt.plot(
            theta_post_reset,
            fixed_point,
            "r*",
            markersize=18,
            label="Poincare fixed point"
        )

        handles.extend([
            cycle_line,
            fixed_point_marker
        ])

    limit_cycle_patch=Patch(
        facecolor="green",
        alpha=0.35,
        label="limit cycle RoA"
    )

    outside_patch=Patch(
        facecolor="red",
        alpha=0.35,
        label="outside RoA"
    )

    handles.extend([
        limit_cycle_patch,
        outside_patch
    ])

    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title(title)
    plt.legend(handles=handles,loc="upper right")
    angle_between_spokes=2*np.pi/params["number_spokes"]
    theta_center=params["ground_inclination"]
    theta_half_range=2.5*angle_between_spokes

    plt.xlim(
        min(theta_range[0],theta_center-theta_half_range),
        max(theta_range[-1],theta_center+theta_half_range)
    )
    plt.ylim(
        theta_dot_range[0],
        theta_dot_range[-1]
    )
    plt.tight_layout()
