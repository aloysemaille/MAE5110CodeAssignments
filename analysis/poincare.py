import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from analysis import roa

def simulate_one_step(theta_dot_n,theta_post_reset,params,timestep=1e-3,max_time=5.0):
    current_time=0.0
    current_state=np.array([theta_post_reset,theta_dot_n],dtype=float)

    while current_time<max_time:
        dt=min(timestep,max_time-current_time)
        current_state,impacted,impact_state=model.step_with_impact(current_state,params,dt)
        current_time+=dt

        if impacted:
            return impact_state[1]

    return None

def build_return_map(theta_dot_range,theta_post_reset,params,timestep=1e-3,max_time=5.0):
    theta_dot_next=np.full_like(theta_dot_range,np.nan,dtype=float)

    for i,theta_dot_n in enumerate(theta_dot_range):
        result=simulate_one_step(
            theta_dot_n,
            theta_post_reset,
            params,
            timestep=timestep,
            max_time=max_time
        )
        if result is not None:
            theta_dot_next[i]=result

    return theta_dot_next

def find_fixed_point(theta_dot_range,theta_dot_next):
    difference=theta_dot_next-theta_dot_range

    for i in range(len(theta_dot_range)-1):
        y0=difference[i]
        y1=difference[i+1]

        if not(np.isfinite(y0) and np.isfinite(y1)):
            continue

        if y0==0:
            return theta_dot_range[i]

        if y0*y1<0:
            x0=theta_dot_range[i]
            x1=theta_dot_range[i+1]
            return x0-y0*(x1-x0)/(y1-y0)

    return None

def estimate_floquet_multiplier(fixed_point,theta_post_reset,params,perturbation=1e-4,timestep=1e-3,max_time=5.0):
    theta_dot_plus=simulate_one_step(
        fixed_point+perturbation,
        theta_post_reset,
        params,
        timestep=timestep,
        max_time=max_time
    )
    theta_dot_minus=simulate_one_step(
        fixed_point-perturbation,
        theta_post_reset,
        params,
        timestep=timestep,
        max_time=max_time
    )

    if theta_dot_plus is None or theta_dot_minus is None:
        return None

    return (theta_dot_plus-theta_dot_minus)/(2*perturbation)

def simulate_one_step_full(theta_dot_n,theta_post_reset,params,timestep=1e-3,max_time=5.0):
    current_time=0.0
    current_state=np.array([theta_post_reset,theta_dot_n],dtype=float)
    trajectory=[current_state.copy()]

    while current_time<max_time:
        dt=min(timestep,max_time-current_time)
        current_state,impacted,impact_state=model.step_with_impact(current_state,params,dt)
        current_time+=dt

        if impacted:
            trajectory.append(impact_state.copy())
            return np.array(trajectory)

        trajectory.append(current_state.copy())

    return np.array(trajectory)

def run_stability_sweep(param_name,param_values,base_params,timestep,
                        return_map_points=30,theta_dot_search_range=(0.5,6.0),
                        roa_points=15,angular_velocity_bounds=(-10,10)):
    roa_sizes=[]
    floquet_multipliers=[]

    for value in param_values:
        sweep_params=base_params.copy()
        sweep_params[param_name]=value

        theta_bounds,theta_post_reset=model.compute_theta_bounds(sweep_params)

        theta_dot_samples=np.linspace(
            theta_dot_search_range[0],
            theta_dot_search_range[1],
            return_map_points
        )

        theta_dot_next=build_return_map(
            theta_dot_samples,
            theta_post_reset,
            sweep_params,
            timestep=timestep
        )
        fixed_point=find_fixed_point(theta_dot_samples,theta_dot_next)

        if fixed_point is None:
            floquet_multipliers.append(np.nan)
            roa_sizes.append(0.0)
            continue

        multiplier=estimate_floquet_multiplier(
            fixed_point,
            theta_post_reset,
            sweep_params,
            timestep=timestep
        )

        floquet_multipliers.append(
            np.nan if multiplier is None else multiplier
        )

        _,_,roa_results=roa.build_roa_grid(
            sweep_params,
            theta_bounds,
            timestep,
            fixed_point=fixed_point,
            number_of_bounds=roa_points,
            angular_velocity_bounds=angular_velocity_bounds
        )

        roa_sizes.append(
            np.mean(roa_results=="limit cycle")
        )

    return np.array(roa_sizes),np.array(floquet_multipliers)

def plot_return_map(theta_dot_range,theta_dot_next,fixed_point=None,title="Return map"):
    plt.figure(figsize=(6,6))
    plt.plot(theta_dot_range,theta_dot_next,"o-",label="return map",markersize=3)
    plt.plot(theta_dot_range,theta_dot_range,"--",label="identity line")

    if fixed_point is not None:
        plt.plot(fixed_point,fixed_point,"*",markersize=15,label="fixed point")

    plt.xlabel("theta_dot at impact n (rad/s)")
    plt.ylabel("theta_dot at impact n+1 (rad/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_sweep_results(param_values,roa_sizes,floquet_multipliers,param_label):
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))

    axes[0].plot(param_values,roa_sizes,"o-")
    axes[0].set_xlabel(param_label)
    axes[0].set_ylabel("RoA size")
    axes[0].set_title(f"RoA size versus {param_label}")

    axes[1].plot(param_values,floquet_multipliers,"o-")
    axes[1].axhline(1,linestyle="--")
    axes[1].axhline(-1,linestyle="--")
    axes[1].set_xlabel(param_label)
    axes[1].set_ylabel("Floquet multiplier")
    axes[1].set_title(f"Floquet multiplier versus {param_label}")

    fig.tight_layout()
