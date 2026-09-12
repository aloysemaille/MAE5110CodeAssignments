import numpy as np
import matplotlib.pyplot as plt

def make_roa_grid(params, theta_range, angular_velocity_range, n_theta, n_angular_velocity,timestep, simulation_time, integrator, model,speed_tolerance=1e-1, cycle_tolerance=1e-1, n_consecutive=3):

    theta_points = np.linspace(theta_range[0], theta_range[1], n_theta)
    angular_velocity_points = np.linspace(angular_velocity_range[0], angular_velocity_range[1], n_angular_velocity)
    result = np.empty((n_angular_velocity, n_theta), dtype=object)

    angle_between_spokes = 2 * np.pi / params["number_spokes"]

    for i, theta0 in enumerate(theta_points):
        for j, theta_dot0 in enumerate(angular_velocity_points):

            _, state_history = model.simulate_trajectory(np.array([theta0, theta_dot0]), params, timestep,simulation_time, integrator, model)
            final_state = state_history[-1]

            if abs(final_state[1]) < speed_tolerance:
                result[j, i] = "fixed_point"
                continue

            theta = state_history[:, 0]
            reset_indices = np.where(np.diff(theta) < -0.5 * angle_between_spokes)[0]
            impact_velocities = state_history[reset_indices + 1, 1]

            if len(impact_velocities) >= n_consecutive:
                recent = impact_velocities[-n_consecutive:]
                if np.max(np.abs(np.diff(recent))) < cycle_tolerance:
                    result[j, i] = "limit_cycle"
                    continue

            result[j, i] = "undecided"

    return theta_points, angular_velocity_points, result


def plot_roa(theta_points, angular_velocity_points, result):
    label_map = {"fixed_point": 0, "limit_cycle": 1, "undecided": 2}
    grid = np.vectorize(label_map.get)(result)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto",extent=[theta_points[0], theta_points[-1],angular_velocity_points[0], angular_velocity_points[-1]],cmap="viridis", vmin=0, vmax=2,)

    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["fixed point", "limit cycle", "undecided"])

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\dot\theta_0$")
    ax.set_title("Region of Attraction")

    plt.tight_layout()
    plt.show()

    return fig, ax

def compute_roa_size(result):
    return np.mean(result == "limit_cycle")