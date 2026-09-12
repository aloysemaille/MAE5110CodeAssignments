import numpy as np

def rk4(timestep, params, model, initial_state, sim_time):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        k1 = model(state, params)
        k2 = model(state + timestep / 2 * k1, params)
        k3 = model(state + timestep / 2 * k2, params)
        k4 = model(state + timestep * k3, params)

        state_traj[:, step + 1] = state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return time_traj, state_traj