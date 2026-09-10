import numpy as np
import matplotlib.pyplot as plt


def simulate_one_step(theta_dot_n,theta_post_reset,params,dynamics_fn,reset_fn,timestep=1e-3,max_time=5.0):
    from integrators import rk4 as integrator

    current_time=0.0
    current_state=np.array([theta_post_reset,theta_dot_n],dtype=float)

    while current_time<max_time:
        dt=min(timestep,max_time-current_time)
        _,step_states=integrator.rk4(dt,params,dynamics_fn,current_state,dt)
        state_after_step=step_states[:,-1]
        new_state=reset_fn(current_time,state_after_step,params)

        impacted=not np.isclose(new_state[0],state_after_step[0],rtol=0.0,atol=1e-12)

        current_state=new_state
        current_time+=dt

        if impacted:
            return current_state[1]

    return None


def build_return_map(theta_dot_range,theta_post_reset,params,dynamics_fn,reset_fn,**sim_kwargs):
    theta_dot_next=np.full_like(theta_dot_range,np.nan,dtype=float)

    for i,theta_dot_n in enumerate(theta_dot_range):
        result=simulate_one_step(theta_dot_n,theta_post_reset,params,dynamics_fn,reset_fn,**sim_kwargs)
        if result is not None:
            theta_dot_next[i]=result

    return theta_dot_next


def find_fixed_point(theta_dot_range,theta_dot_next):
    theta_dot_range=np.asarray(theta_dot_range)
    theta_dot_next=np.asarray(theta_dot_next)
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


def estimate_floquet_multiplier(fixed_point,theta_post_reset,params,dynamics_fn,reset_fn, perturbation=1e-3,**sim_kwargs):
    theta_dot_plus=simulate_one_step(fixed_point+perturbation,theta_post_reset,params,dynamics_fn, reset_fn,**sim_kwargs)
    theta_dot_minus=simulate_one_step(fixed_point-perturbation,theta_post_reset,params,dynamics_fn, reset_fn,**sim_kwargs)

    if theta_dot_plus is None or theta_dot_minus is None:
        return None

    return (theta_dot_plus-theta_dot_minus)/(2*perturbation)


def plot_return_map(theta_dot_range,theta_dot_next,fixed_point=None,title="Return map"):
    plt.figure(figsize=(6,6))
    plt.plot(theta_dot_range,theta_dot_next,"o-",label="return map",markersize=3)
    plt.plot(theta_dot_range,theta_dot_range,"--",color="gray",label="identity line")

    if fixed_point is not None:
        plt.plot(fixed_point,fixed_point,"*",color="red",markersize=15,label="fixed point")

    plt.xlabel("theta_dot at impact n (rad/s)")
    plt.ylabel("theta_dot at impact n+1 (rad/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def run_stability_sweep(param_name,param_values,base_params,dynamics_fn,reset_fn,compute_theta_bounds_fn,
                        timestep,roa_grid_size=15,return_map_points=30,theta_dot_search_range=(0.5,6.0)):
    from analysis import roa

    roa_fractions=[]
    floquet_multipliers=[]

    for value in param_values:
        sweep_params=dict(base_params)
        sweep_params[param_name]=value

        theta_bounds,theta_post_reset=compute_theta_bounds_fn(sweep_params)

        _,_,roa_results=roa.build_roa_grid(sweep_params,dynamics_fn,reset_fn,theta_bounds,timestep,
                                           number_of_bounds=roa_grid_size)
        roa_fractions.append(roa.compute_walking_fraction(roa_results))

        theta_dot_samples=np.linspace(*theta_dot_search_range,return_map_points)
        theta_dot_next=build_return_map(theta_dot_samples,theta_post_reset,sweep_params,dynamics_fn,
                                        reset_fn,timestep=timestep)

        fixed_point=find_fixed_point(theta_dot_samples,theta_dot_next)
        multiplier=np.nan

        if fixed_point is not None:
            multiplier=estimate_floquet_multiplier(fixed_point,theta_post_reset,sweep_params,dynamics_fn,
                                                   reset_fn,timestep=timestep)

        floquet_multipliers.append(multiplier)

    return np.array(roa_fractions),np.array(floquet_multipliers)


def plot_sweep_results(param_values,roa_fractions,floquet_multipliers,param_label):
    fig,axes=plt.subplots(1,2,figsize=(11,4))

    axes[0].plot(param_values,roa_fractions,"o-")
    axes[0].set_xlabel(param_label)
    axes[0].set_ylabel("fraction of grid -> walking")
    axes[0].set_title(f"RoA size vs {param_label}")

    axes[1].plot(param_values,floquet_multipliers,"o-")
    axes[1].axhline(1,color="gray",linestyle="--")
    axes[1].axhline(-1,color="gray",linestyle="--")
    axes[1].set_xlabel(param_label)
    axes[1].set_ylabel("Floquet multiplier")
    axes[1].set_title(f"Floquet multiplier vs {param_label}")

    plt.tight_layout()