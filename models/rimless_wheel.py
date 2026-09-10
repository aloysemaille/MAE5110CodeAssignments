import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4 as integrator

def dynamics(t,state,params):
    gravity=params["gravity"]
    length=params["length"]
    mass=params["mass"]
    damping_coeff=params["damping_coeff"]
    theta=state[0]
    theta_dot=state[1]
    theta_ddot=(mass*gravity*length*np.sin(theta)-damping_coeff*theta_dot)/(mass*length**2)
    return np.array([theta_dot,theta_ddot])

def get_trigger_angle(params):
    angle_between_spokes=2*np.pi/params["number_spokes"]
    return params["ground_inclination"]+angle_between_spokes/2

def is_impact(state,params):
    return state[0]>=get_trigger_angle(params) and state[1]>0

def reset(state,params):
    angle_between_spokes=2*np.pi/params["number_spokes"]
    return np.array([
        state[0]-angle_between_spokes,
        state[1]*np.cos(angle_between_spokes)
    ])

def next_spoke_trigger(t,state,params):
    if is_impact(state,params):
        return reset(state,params)
    return state

def detect_impact_crossing(state_before,state_after,params):
    trigger_angle=get_trigger_angle(params)
    return state_before[0]<trigger_angle<=state_after[0] and state_after[1]>0

def step_with_impact(state,params,timestep):
    state_before=np.asarray(state,dtype=float).copy()
    state_after=integrator.rk4_step(timestep,params,dynamics,state_before)

    if not detect_impact_crossing(state_before,state_after,params):
        return state_after,False,None

    trigger_angle=get_trigger_angle(params)
    theta_change=state_after[0]-state_before[0]

    if theta_change<=0:
        return state_after,False,None

    fraction=np.clip((trigger_angle-state_before[0])/theta_change,0.0,1.0)
    impact_dt=timestep*fraction
    impact_pre=integrator.rk4_step(impact_dt,params,dynamics,state_before)
    impact_pre[0]=trigger_angle
    impact_post=reset(impact_pre,params)

    remaining_dt=timestep-impact_dt

    if remaining_dt>1e-12:
        state_final=integrator.rk4_step(remaining_dt,params,dynamics,impact_post)
    else:
        state_final=impact_post

    return state_final,True,impact_post

def compute_theta_bounds(params):
    angle_between_spokes=2*np.pi/params["number_spokes"]
    gamma=params["ground_inclination"]
    lower_bound=gamma-angle_between_spokes/2
    upper_bound=gamma+angle_between_spokes/2
    return (lower_bound,upper_bound),lower_bound

def simulate_trajectory(initial_state,params,timestep,simulation_time):
    current_time=0.0
    current_state=np.asarray(initial_state,dtype=float).copy()
    time_history=[current_time]
    state_history=[current_state.copy()]

    while current_time<simulation_time:
        dt=min(timestep,simulation_time-current_time)
        current_state,_,_=step_with_impact(current_state,params,dt)
        current_time+=dt
        time_history.append(current_time)
        state_history.append(current_state.copy())

    return np.array(time_history),np.array(state_history)

def calculate_energy(state_history,params):
    gravity=params["gravity"]
    length=params["length"]
    mass=params["mass"]
    theta=state_history[:,0]
    theta_dot=state_history[:,1]
    kinetic_energy=0.5*mass*(length*theta_dot)**2
    potential_energy=mass*gravity*length*np.cos(theta)
    return potential_energy,kinetic_energy

def plot_energy_vs_time(time_history,potential_energy,kinetic_energy,total_energy):
    plt.figure()
    plt.plot(time_history,potential_energy,label="potential energy")
    plt.plot(time_history,kinetic_energy,label="kinetic energy")
    plt.plot(time_history,total_energy,label="total energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Energies versus Time")
    plt.legend()
