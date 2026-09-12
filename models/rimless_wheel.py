import numpy as np

def pendulum(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (mass * gravity * length * np.sin(angle) - damping_coeff * angular_velocity) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative

def next_spoke_trigger(state, params):
    angle_between_spokes = 2 * np.pi / params["number_spokes"]
    if state[0] >= (params["ground_inclination"] + angle_between_spokes / 2) and state[1] > 0:
        return np.array([state[0] - angle_between_spokes, state[1] * np.cos(angle_between_spokes)])
    return state

def compute_theta_bounds(params):
    angle_between_spokes=2*np.pi/params["number_spokes"]
    gamma=params["ground_inclination"]

    lower_bound=gamma-angle_between_spokes/2
    upper_bound=gamma+angle_between_spokes/2

    return (lower_bound,upper_bound),lower_bound

def simulate_trajectory(initial_state, params, timestep, simulation_time, integrator, model):
    current_time = 0.0
    current_state = initial_state
    time_history = [current_time]
    state_history = [current_state]

    while current_time <= simulation_time:
        _, state_traj = integrator.rk4(timestep, params, model.pendulum, current_state, timestep)
        new_state = state_traj[:, -1]
        current_state = next_spoke_trigger(new_state, params)
        current_time += timestep
        time_history.append(current_time)
        state_history.append(current_state)

    return np.array(time_history), np.array(state_history)

def calculate_energy(state_history,params):
    gravity=params["gravity"]
    length=params["length"]
    mass=params["mass"]
    theta=state_history[:,0]
    theta_dot=state_history[:,1]
    kinetic_energy=0.5*mass*(length*theta_dot)**2
    potential_energy=mass*gravity*length*np.cos(theta)
    total_energy = kinetic_energy + potential_energy
    return potential_energy,kinetic_energy, total_energy
