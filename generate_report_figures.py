import numpy as np

from models import inverted_pendulum_walker as model
from analysis import roa
from tools.lookup_table import (
    build_lookup_table,
    find_single_step_stabilizable,
    compute_fastest_steps_to_standstill,
    compute_slowest_steps_to_standstill,
)
from tools.trajectory_plots import simulate_full_trajectory, plot_state_space_trajectories, plot_steps_to_standstill

params = model.generate_params()

theta_points, angular_velocity_points, grid_result = roa.make_roa_grid(
    params,
    theta_range=(-0.3, 0.3),
    angular_velocity_range=(-1.5, 1.5),
    n_theta=25,
    n_angular_velocity=25,
    timestep=1e-3,
    simulation_time=3.0,
    integrator=None,
    model=model,
)
roa.plot_roa(theta_points, angular_velocity_points, grid_result)

angle_of_attack_min = np.pi / 8
angle_of_attack_max = np.pi / 7
lookup_timestep = 1e-3


def test_grid_resolution(resolutions, params, timestep, test_velocities):
    results_by_resolution = {}

    for n_velocity, n_angle in resolutions:
        froude_two_velocity = np.sqrt(2 * 2 * params["gravity"] / params["length"])
        velocity_grid = np.linspace(0.0, froude_two_velocity, n_velocity)
        angle_grid = np.linspace(angle_of_attack_min, angle_of_attack_max, n_angle)

        table = build_lookup_table(velocity_grid, angle_grid, params, timestep)
        single_step = find_single_step_stabilizable(velocity_grid, theta_points, angular_velocity_points, grid_result)
        steps, _ = compute_fastest_steps_to_standstill(table, velocity_grid, angle_grid, single_step)

        outcomes = [steps[np.argmin(np.abs(velocity_grid - v))] for v in test_velocities]
        results_by_resolution[(n_velocity, n_angle)] = outcomes

    return results_by_resolution


test_velocities = [1.0, 2.0, 3.0]
resolutions = [(10, 5), (20, 10), (40, 20), (80, 40)]
results = test_grid_resolution(resolutions, params, lookup_timestep, test_velocities)
for resolution, outcomes in results.items():
    print(resolution, outcomes)

froude_two_velocity = np.sqrt(2 * 2 * params["gravity"] / params["length"])
angular_velocity_grid = np.linspace(0.0, froude_two_velocity, 40)
angle_of_attack_grid = np.linspace(angle_of_attack_min, angle_of_attack_max, 20)

table = build_lookup_table(angular_velocity_grid, angle_of_attack_grid, params, lookup_timestep)
single_step_stabilizable = find_single_step_stabilizable(
    angular_velocity_grid, theta_points, angular_velocity_points, grid_result
)

fastest_steps_to_standstill, fastest_angle_of_attack = compute_fastest_steps_to_standstill(
    table, angular_velocity_grid, angle_of_attack_grid, single_step_stabilizable
)
slowest_steps_to_standstill, slowest_angle_of_attack = compute_slowest_steps_to_standstill(
    table, angular_velocity_grid, angle_of_attack_grid, single_step_stabilizable
)

example_initial_state = np.array([0.0, 3.0])

fastest_state_history, fastest_completed_steps = simulate_full_trajectory(
    example_initial_state, dict(params), lookup_timestep,
    angular_velocity_grid, fastest_angle_of_attack, theta_points, angular_velocity_points, grid_result,
)
slowest_state_history, slowest_completed_steps = simulate_full_trajectory(
    example_initial_state, dict(params), lookup_timestep,
    angular_velocity_grid, slowest_angle_of_attack, theta_points, angular_velocity_points, grid_result,
)

plot_state_space_trajectories(
    fastest_state_history, fastest_completed_steps,
    slowest_state_history, slowest_completed_steps,
    example_initial_state, save_path="images/state_space_trajectory.png",
)

plot_steps_to_standstill(angular_velocity_grid, fastest_steps_to_standstill, save_path="images/steps_to_standstill.png")

print(f"Fastest policy: {fastest_completed_steps} steps")
print(f"Slowest policy: {slowest_completed_steps} steps")