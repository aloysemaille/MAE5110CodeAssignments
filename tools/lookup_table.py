import numpy as np

from analysis import roa
from analysis.poincare import simulate_step


def build_lookup_table(angular_velocity_grid, angle_of_attack_grid, params, timestep, max_time=5.0):
    table = np.full((len(angular_velocity_grid), len(angle_of_attack_grid)), np.nan)

    for i, angular_velocity in enumerate(angular_velocity_grid):
        for j, angle_of_attack in enumerate(angle_of_attack_grid):
            result = simulate_step(angular_velocity, angle_of_attack, params, timestep, max_time)
            if result is not None:
                table[i, j] = result

    return table


def find_single_step_stabilizable(
    table, angular_velocity_grid, angle_of_attack_grid,
    theta_points, angular_velocity_points, grid_result,
):
    stabilizable = np.zeros(len(angular_velocity_grid), dtype=bool)
    single_step_angle_of_attack = np.full(len(angular_velocity_grid), np.nan)

    for i in range(len(angular_velocity_grid)):
        for j, next_angular_velocity in enumerate(table[i, :]):
            if not np.isfinite(next_angular_velocity):
                continue

            if roa.state_in_roa(
                np.array([0.0, next_angular_velocity]),
                theta_points, angular_velocity_points, grid_result,
            ):
                stabilizable[i] = True
                single_step_angle_of_attack[i] = angle_of_attack_grid[j]
                break

    return stabilizable, single_step_angle_of_attack


def compute_fastest_steps_to_standstill(
    table, angular_velocity_grid, angle_of_attack_grid,
    single_step_stabilizable, single_step_angle_of_attack,
):
    steps_to_standstill = np.full(len(angular_velocity_grid), np.inf)
    steps_to_standstill[single_step_stabilizable] = 1

    best_angle_of_attack = np.full(len(angular_velocity_grid), np.nan)
    best_angle_of_attack[single_step_stabilizable] = single_step_angle_of_attack[single_step_stabilizable]

    still_updating = True

    while still_updating:
        still_updating = False

        for i in range(len(angular_velocity_grid)):
            if np.isfinite(steps_to_standstill[i]):
                continue

            candidate_costs = np.full(len(angle_of_attack_grid), np.nan)

            for j, next_angular_velocity in enumerate(table[i, :]):
                if not np.isfinite(next_angular_velocity):
                    continue

                next_index = np.argmin(np.abs(angular_velocity_grid - next_angular_velocity))

                if np.isfinite(steps_to_standstill[next_index]):
                    candidate_costs[j] = steps_to_standstill[next_index] + 1

            if np.all(np.isnan(candidate_costs)):
                continue

            chosen_action_index = np.nanargmin(candidate_costs)

            steps_to_standstill[i] = candidate_costs[chosen_action_index]
            best_angle_of_attack[i] = angle_of_attack_grid[chosen_action_index]

            still_updating = True

    return steps_to_standstill, best_angle_of_attack


def compute_slowest_steps_to_standstill(
    table, angular_velocity_grid, angle_of_attack_grid,
    single_step_stabilizable, single_step_angle_of_attack, max_steps=20,
):
    steps_to_standstill = np.full(len(angular_velocity_grid), -np.inf)
    steps_to_standstill[single_step_stabilizable] = 1

    best_angle_of_attack = np.full(len(angular_velocity_grid), np.nan)
    best_angle_of_attack[single_step_stabilizable] = single_step_angle_of_attack[single_step_stabilizable]

    for _ in range(max_steps):
        previous_steps = steps_to_standstill.copy()
        still_updating = False

        for i in range(len(angular_velocity_grid)):
            if single_step_stabilizable[i]:
                continue

            candidate_costs = np.full(len(angle_of_attack_grid), np.nan)

            for j, next_angular_velocity in enumerate(table[i, :]):
                if not np.isfinite(next_angular_velocity):
                    continue

                next_index = np.argmin(np.abs(angular_velocity_grid - next_angular_velocity))

                if np.isfinite(previous_steps[next_index]):
                    candidate_costs[j] = previous_steps[next_index] + 1

            if np.all(np.isnan(candidate_costs)):
                continue

            chosen_action_index = np.nanargmax(candidate_costs)
            candidate_steps = candidate_costs[chosen_action_index]

            if candidate_steps > steps_to_standstill[i]:
                steps_to_standstill[i] = candidate_steps
                best_angle_of_attack[i] = angle_of_attack_grid[chosen_action_index]
                still_updating = True

        if not still_updating:
            break

    steps_to_standstill[steps_to_standstill == -np.inf] = np.inf

    return steps_to_standstill, best_angle_of_attack