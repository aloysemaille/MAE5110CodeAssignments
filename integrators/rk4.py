import numpy as np

def rk4(timestep, params, model, initial_state, sim_time):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        k1 = model(t, state_traj[:, step], params)
        k2 = model(t + timestep / 2, state_traj[:, step] + timestep / 2 * k1, params)
        k3 = model(t + timestep / 2, state_traj[:, step] + timestep / 2 * k2, params)
        k4 = model(t + timestep, state_traj[:, step] + timestep * k3, params)

        state_traj[:, step + 1] = state_traj[:, step] + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return time_traj, state_traj

    
def rk4_step(timestep, params, model, state):
    k1 = model(0, state, params)
    k2 = model(0, state + timestep / 2 * k1, params)
    k3 = model(0, state + timestep / 2 * k2, params)
    k4 = model(0, state + timestep * k3, params)

    return state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    

    
    

