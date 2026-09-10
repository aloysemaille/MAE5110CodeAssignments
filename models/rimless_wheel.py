import numpy as np
import matplotlib.pyplot as plt

from integrators import rk4 as integrator


def dynamics(t, state, params):

    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    theta_angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (mass * gravity * length * np.sin(theta_angle) - 
                            damping_coeff * angular_velocity) / (mass * length ** 2)

    return np.array([angular_velocity, angular_acceleration])


def next_spoke_trigger(t, state, params):
    theta_angle = state[0]
    angular_velocity = state[1]

    angle_between_adjacent_spokes = (2 * np.pi / params["number_spokes"])

    trigger_angle = (params["ground_inclination"] + angle_between_adjacent_spokes / 2)

    if theta_angle < trigger_angle:
        return state

    theta_angle -= (angle_between_adjacent_spokes)
    angular_velocity *= np.cos(angle_between_adjacent_spokes)

    return np.array([theta_angle,angular_velocity])


def compute_theta_bounds(params):

    angle_between_adjacent_spokes = (2 * np.pi / params["number_spokes"])
    ground_inclination = (params["ground_inclination"])

    lower_bound = (ground_inclination - angle_between_adjacent_spokes / 2)
    upper_bound = (ground_inclination + angle_between_adjacent_spokes / 2)

    theta_reset = lower_bound

    return (lower_bound, upper_bound), theta_reset


def simulate_trajectory(initial_state, params, timestep, simulation_time):
    current_time = 0.0

    current_state = np.asarray(initial_state, dtype=float).copy()

    time_history = [current_time]

    state_history = [current_state.copy()]

    while current_time < simulation_time:

        dt = min(timestep, simulation_time - current_time)

        _, step_states = integrator.rk4(dt, params, dynamics, current_state,dt)

        state_after_step = (step_states[:, -1])

        current_state = (next_spoke_trigger(current_time, state_after_step, params))

        current_time += dt

        time_history.append(current_time)

        state_history.append(current_state.copy())

    return (np.array(time_history), np.array(state_history))


def calculate_energy(state_history, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state_history[:, 0]
    angular_velocity = state_history[:, 1]

    kinetic_energy = (0.5 * mass * (length * angular_velocity) ** 2)
    potential_energy = (mass * gravity * length * np.cos(angle))

    return (potential_energy, kinetic_energy)


def plot_energy_vs_time(time_history, potential_energy, kinetic_energy, total_energy):
    plt.figure()
    plt.plot(time_history, potential_energy, label="potential energy")
    plt.plot(time_history, kinetic_energy, label="kinetic energy")
    plt.plot(time_history, total_energy, label="total energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Energies versus Time")
    plt.legend()
