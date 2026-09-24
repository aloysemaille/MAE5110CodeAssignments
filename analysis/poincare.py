import numpy as np

from models import inverted_pendulum_walker as model


def simulate_one_step(theta_dot_n, theta_post_reset, params, model, integrator, timestep, max_time=5.0):
    """Simulates the wheel from a post-reset state until the first spoke impact and returns the resulting angular velocity."""

    angle_between_spokes = 2 * np.pi / params["number_spokes"]
    initial_state = np.array([theta_post_reset, theta_dot_n])

    _, state_history = model.simulate_trajectory(initial_state, params, timestep, max_time, integrator, model)

    theta = state_history[:, 0]
    reset_indices = np.where(np.diff(theta) < -0.5 * angle_between_spokes)[0]

    if len(reset_indices) == 0:
        return None

    return state_history[reset_indices[0] + 1, 1]


def build_return_map(theta_dot_range, theta_post_reset, params, model, integrator, timestep, max_time=5.0):
    """Builds the Poincaré return map by simulating one impact-to-impact step for each initial angular velocity in theta_dot_range."""

    theta_dot_next = np.full_like(theta_dot_range, np.nan, dtype=float)

    for i, theta_dot_n in enumerate(theta_dot_range):
        result = simulate_one_step(theta_dot_n, theta_post_reset, params, model, integrator, timestep=timestep, max_time=max_time)
        if result is not None:
            theta_dot_next[i] = result

    return theta_dot_next


def find_fixed_point(theta_dot_range, theta_dot_next):
    """Finds a fixed point of the return map by locating a sign change in (theta_dot_next - theta_dot_range) and interpolating linearly."""

    difference = theta_dot_next - theta_dot_range

    for i in range(len(theta_dot_range) - 1):
        y0 = difference[i]
        y1 = difference[i + 1]

        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue

        if y0 == 0:
            return theta_dot_range[i]

        if y0 * y1 < 0:
            x0 = theta_dot_range[i]
            x1 = theta_dot_range[i + 1]
            return x0 - y0 * (x1 - x0) / (y1 - y0)

    return None


def estimate_floquet_multiplier(fixed_point, theta_post_reset, params, model, integrator, perturbation=1e-4, timestep=1e-2, max_time=5.0):
    """Estimates the Floquet multiplier at a fixed point via central finite difference of the return map."""

    theta_dot_plus = simulate_one_step(fixed_point + perturbation, theta_post_reset, params, model, integrator, timestep=timestep, max_time=max_time)
    theta_dot_minus = simulate_one_step(fixed_point - perturbation, theta_post_reset, params, model, integrator, timestep=timestep, max_time=max_time)

    if theta_dot_plus is None or theta_dot_minus is None:
        return None

    return (theta_dot_plus - theta_dot_minus) / (2 * perturbation)


def sweep_stability(param_name, param_values, base_params, timestep, simulation_time, integrator, model, roa,
                     theta_dot_search_range=(0.5, 6.0), return_map_points=40, max_time=5.0,
                     n_theta=15, n_angular_velocity=15, angular_velocity_range=(-10.0, 10.0)):
    """Sweeps a single parameter over param_values, computing the fixed point, Floquet multiplier, and region-of-attraction size at each value."""

    roa_sizes = []
    floquet_multipliers = []

    for value in param_values:
        sweep_params = base_params.copy()
        sweep_params[param_name] = value

        sweep_theta_bounds, sweep_theta_post_reset = model.compute_theta_bounds(sweep_params)

        theta_dot_range = np.linspace(theta_dot_search_range[0], theta_dot_search_range[1], return_map_points)
        theta_dot_next = build_return_map(theta_dot_range, sweep_theta_post_reset, sweep_params, model, integrator, timestep=timestep, max_time=max_time)
        fixed_point = find_fixed_point(theta_dot_range, theta_dot_next)

        if fixed_point is None:
            floquet_multipliers.append(np.nan)
            roa_sizes.append(0.0)
            continue

        multiplier = estimate_floquet_multiplier(fixed_point, sweep_theta_post_reset, sweep_params, model, integrator, timestep=timestep, max_time=max_time)
        floquet_multipliers.append(np.nan if multiplier is None else multiplier)

        _, _, sweep_result = roa.make_roa_grid(
            sweep_params, sweep_theta_bounds, angular_velocity_range=angular_velocity_range,
            n_theta=n_theta, n_angular_velocity=n_angular_velocity, timestep=timestep,
            simulation_time=simulation_time, integrator=integrator, model=model,
        )
        roa_sizes.append(roa.compute_roa_size(sweep_result))

    return np.array(roa_sizes), np.array(floquet_multipliers)

def simulate_step(theta_dot_k, angle_of_attack, params, timestep, max_time=5.0):
    step_params = dict(params)
    step_params["angle_of_attack"] = angle_of_attack
    step_params["ankle_torque"] = 0.0

    state = np.array([0.0, theta_dot_k])
    reset_done = False
    t = 0.0

    while t < max_time:
        next_state = state + timestep * model.dynamics(t, state, step_params)

        if not reset_done and model.event_guard(state, next_state, step_params):
            next_state = model.event_dynamics(next_state, step_params)
            reset_done = True

        if reset_done and state[0] < 0.0 <= next_state[0]:
            return next_state[1]

        state = next_state
        t += timestep

    return None