import numpy as np
import matplotlib.pyplot as plt

from models import inverted_pendulum_walker as model
from analysis import roa
from tools.controller import choose_ankle_torque


def policy_angle_of_attack(angular_velocity, angular_velocity_grid, best_angle_of_attack):
    index = np.argmin(np.abs(angular_velocity_grid - angular_velocity))
    return best_angle_of_attack[index]


def simulate_full_trajectory(
    initial_state, params, timestep, angular_velocity_grid, best_angle_of_attack,
    theta_points, angular_velocity_points, grid_result, max_time=20.0,
):
    state = np.array(initial_state, dtype=float)
    time = 0.0
    state_history = [state.copy()]
    completed_steps = 0

    while time < max_time:
        if roa.state_in_roa(state, theta_points, angular_velocity_points, grid_result):
            params["ankle_torque"] = choose_ankle_torque(state, params)
        else:
            params["ankle_torque"] = 0.0
            chosen_angle = policy_angle_of_attack(state[1], angular_velocity_grid, best_angle_of_attack)
            if np.isfinite(chosen_angle):
                params["angle_of_attack"] = chosen_angle

        next_state = state + timestep * model.dynamics(time, state, params)

        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(next_state, params)
            completed_steps += 1

        state = next_state
        time += timestep
        state_history.append(state.copy())

        if roa.state_in_roa(state, theta_points, angular_velocity_points, grid_result):
            break

    return np.array(state_history).T, completed_steps


def plot_state_space_trajectories(
    fastest_state_history, fastest_steps, slowest_state_history, slowest_steps,
    initial_state, save_path=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fastest_state_history[0, :], fastest_state_history[1, :], label=f"fastest ({fastest_steps} steps)")
    ax.plot(slowest_state_history[0, :], slowest_state_history[1, :], "--", label=f"slowest ({slowest_steps} steps)")
    ax.plot(initial_state[0], initial_state[1], "o", color="black", markersize=10, label="Initial state")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title("State-Space Trajectory")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()
    return fig


def plot_steps_to_standstill(angular_velocity_grid, steps_to_standstill, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    finite_mask = np.isfinite(steps_to_standstill)
    ax.plot(angular_velocity_grid[finite_mask], steps_to_standstill[finite_mask], "o-")
    ax.set_xlabel(r"Initial $\dot\theta_0$ at $\theta=0$ (rad/s)")
    ax.set_ylabel("Steps to standstill")
    ax.set_title("Steps to reach standstill vs initial angular velocity")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()
    return fig