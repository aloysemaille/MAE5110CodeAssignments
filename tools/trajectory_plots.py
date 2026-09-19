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
    time_history = [time]
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
        time_history.append(time)
        state_history.append(state.copy())

        if roa.state_in_roa(state, theta_points, angular_velocity_points, grid_result):
            break

    return np.array(time_history), np.array(state_history).T, completed_steps


def plot_trajectory(time_history, state_history, title):
    fig, (ax_theta, ax_velocity) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_theta.plot(time_history, state_history[0, :])
    ax_theta.set_ylabel(r"$\theta$ (rad)")
    ax_velocity.plot(time_history, state_history[1, :])
    ax_velocity.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax_velocity.set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig


def plot_steps_to_standstill(angular_velocity_grid, steps_to_standstill):
    fig, ax = plt.subplots(figsize=(8, 5))
    finite_mask = np.isfinite(steps_to_standstill)
    ax.plot(angular_velocity_grid[finite_mask], steps_to_standstill[finite_mask], "o-")
    ax.set_xlabel(r"Initial $\dot\theta_0$ at $\theta=0$ (rad/s)")
    ax.set_ylabel("Steps to standstill")
    ax.set_title("Steps to reach standstill vs initial angular velocity")
    plt.tight_layout()
    plt.show()
    return fig