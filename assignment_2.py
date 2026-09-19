from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
from analysis import roa
from tools.controller import choose_ankle_torque

params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}

angle_of_attack_min = np.pi / 8
angle_of_attack_max = np.pi / 7


def choose_angle_of_attack(state, params):
    angle_of_attack = params["angle_of_attack"]
    return np.clip(angle_of_attack, angle_of_attack_min, angle_of_attack_max)


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

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]

    if roa.state_in_roa(state, theta_points, angular_velocity_points, grid_result):
        params["ankle_torque"] = choose_ankle_torque(state, params)
    else:
        params["ankle_torque"] = 0.0

    next_state = state + timestep * model.dynamics(t, state, params)

    if model.event_guard(state, next_state, params):
        params["angle_of_attack"] = choose_angle_of_attack(state, params)
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()

roa.plot_roa(theta_points, angular_velocity_points, grid_result)

from tools.lookup_table import build_lookup_table, find_single_step_stabilizable, compute_steps_to_standstill

froude_two_velocity = np.sqrt(2 * 2 * params["gravity"] / params["length"])
angular_velocity_grid = np.linspace(0.0, froude_two_velocity, 30)
angle_of_attack_grid = np.linspace(angle_of_attack_min, angle_of_attack_max, 15)

lookup_timestep = 1e-3
table = build_lookup_table(angular_velocity_grid, angle_of_attack_grid, params, lookup_timestep)

single_step_stabilizable = find_single_step_stabilizable(
    angular_velocity_grid, theta_points, angular_velocity_points, grid_result
)

steps_to_standstill, best_angle_of_attack = compute_steps_to_standstill(
    table, angular_velocity_grid, angle_of_attack_grid, single_step_stabilizable
)