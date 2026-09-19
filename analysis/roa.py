import numpy as np
import matplotlib.pyplot as plt

from tools.controller import choose_ankle_torque


def make_roa_grid(params, theta_range, angular_velocity_range, n_theta, n_angular_velocity, timestep, simulation_time, integrator, model, theta_tolerance=1e-3, angular_velocity_tolerance=1e-3):
    """Builds the grid for the Region of Attraction map and finds for each grid point whether the ankle controller brings it to the upright standstill."""

    theta_points = np.linspace(theta_range[0], theta_range[1], n_theta)
    angular_velocity_points = np.linspace(angular_velocity_range[0], angular_velocity_range[1], n_angular_velocity)
    grid_result = np.empty((n_angular_velocity, n_theta), dtype=object)

    for i, theta0 in enumerate(theta_points):
        for j, theta_dot0 in enumerate(angular_velocity_points):

            grid_point_params = dict(params)
            _, state_history = model.simulate_trajectory(np.array([theta0, theta_dot0]), grid_point_params, timestep, simulation_time)
            final_state = state_history[:, -1]

            if abs(final_state[0]) > 1.5:
                grid_result[j, i] = "diverged"
                continue

            if abs(final_state[0]) < theta_tolerance and abs(final_state[1]) < angular_velocity_tolerance:
                grid_result[j, i] = "fixed_point"
                continue

            grid_result[j, i] = "undecided"

    return theta_points, angular_velocity_points, grid_result


def plot_roa(theta_points, angular_velocity_points, grid_result):
    """Plots the region of attraction map as a discrete color grid."""

    categories = ["fixed_point", "diverged", "undecided"]
    label_map = {cat: i for i, cat in enumerate(categories)}
    grid = np.vectorize(label_map.get)(grid_result)

    cmap = plt.get_cmap("viridis", len(categories))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, origin="lower", aspect="auto", extent=[theta_points[0], theta_points[-1], angular_velocity_points[0], angular_velocity_points[-1]], cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5, interpolation="nearest")

    legend_labels = ["fixed point", "diverged", "undecided"]
    for i, label in enumerate(legend_labels):
        ax.plot([], [], marker="s", linestyle="", color=cmap(i), label=label)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\dot\theta_0$")
    ax.set_title("Region of Attraction")

    plt.tight_layout()
    plt.show()

    return fig, ax


def compute_roa_size(grid_result):
    return np.mean(grid_result == "fixed_point")


def state_in_roa(state, theta_points, angular_velocity_points, grid_result):
    theta_index = np.argmin(np.abs(theta_points - state[0]))
    angular_velocity_index = np.argmin(np.abs(angular_velocity_points - state[1]))
    return grid_result[angular_velocity_index, theta_index] == "fixed_point"