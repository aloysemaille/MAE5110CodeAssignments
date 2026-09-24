import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from analysis import roa, poincare
from tools import plot as plot
from integrators import rk4 as integrator

params={
    "gravity":9.81,
    "length":1,
    "mass":0.25,
    "damping_coeff":0.2,
    "number_spokes":8,
    "ground_inclination":0.2,
}

initial_state=np.array([params["ground_inclination"]+0.05,0.0])

timestep=1e-2
simulation_time=60

theta_bounds,_ = model.compute_theta_bounds(params)

time_history, state_history = model.simulate_trajectory(initial_state,params,timestep,simulation_time,integrator,model)
potential_energy,kinetic_energy, total_energy = model.calculate_energy(state_history, params)

fig, ax = plt.subplots()
plot.plot(time_history, potential_energy, labels="Potential Energy",xlabel="time (s)", ylabel="energy (J)", title="Energy",diagonal=False, marker_point=None, hlines=None, ax=ax)
plot.plot(time_history, kinetic_energy, labels="Kinetic Energy",xlabel="time (s)", ylabel="energy (J)", title="Energy",diagonal=False, marker_point=None, hlines=None, ax=ax)
plot.plot(time_history, total_energy, labels="Total Energy",xlabel="time (s)", ylabel="energy (J)", title="Energy",diagonal=False, marker_point=None, hlines=None, ax=ax)
ax.legend()

theta_points, angular_velocity_points, result = roa.make_roa_grid(params,theta_bounds,angular_velocity_range=(-10.0, 10.0),n_theta=25,n_angular_velocity=25,timestep=timestep,simulation_time=simulation_time,integrator=integrator,model=model,)

theta_dot_range = np.linspace(0.5, 6.0, 40)
theta_dot_next = poincare.build_return_map(theta_dot_range, theta_bounds[0], params, model, integrator, timestep=timestep, max_time=5.0)
fixed_point = poincare.find_fixed_point(theta_dot_range, theta_dot_next)

floquet_multiplier = None
if fixed_point is not None:
    floquet_multiplier = poincare.estimate_floquet_multiplier(fixed_point, theta_bounds[0], params, model, integrator, timestep=timestep, max_time=5.0)

fig, ax = plt.subplots()
plot.plot(theta_dot_range, theta_dot_next, labels="return map",
          xlabel="theta_dot_n (rad/s)", ylabel="theta_dot_n+1 (rad/s)",
          title="Poincaré return map", diagonal=True,
          marker_point=(fixed_point, fixed_point) if fixed_point is not None else None,
          hlines=None, ax=ax)


fig, ax = plt.subplots(figsize=(8, 6))

label_map = {"fixed_point": 0, "limit_cycle": 1, "undecided": 2}
grid = np.vectorize(label_map.get)(result)

im = ax.imshow(grid, origin="lower", aspect="auto",
               extent=[theta_points[0], theta_points[-1], angular_velocity_points[0], angular_velocity_points[-1]],
               cmap="viridis", vmin=0, vmax=2, alpha=0.5)

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["fixed point", "limit cycle", "undecided"])

plot.plot(state_history[:, 0], state_history[:, 1], labels="trajectory", xlabel="theta (rad)", ylabel="angular velocity (rad/s)", title="State Space + RoA", diagonal=False, marker_point=None,hlines=None, ax=ax)

plt.tight_layout()

gamma_values = np.linspace(0.1, np.pi/3, 8)
roa_size_by_gamma, floquet_by_gamma = poincare.sweep_stability("ground_inclination", gamma_values, params, timestep, simulation_time, integrator, model, roa)

n_spokes_values = np.arange(6, 13)
roa_size_by_n_spokes, floquet_by_n_spokes = poincare.sweep_stability("number_spokes", n_spokes_values, params, timestep, simulation_time, integrator, model, roa)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot.plot(gamma_values, roa_size_by_gamma, xlabel="gamma (rad)", ylabel="RoA size", title="RoA size versus gamma", ax=axes[0])
plot.plot(gamma_values, floquet_by_gamma, xlabel="gamma (rad)", ylabel="Floquet multiplier", title="Floquet multiplier versus gamma", hlines=[1, -1], ax=axes[1])
fig.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot.plot(n_spokes_values, roa_size_by_n_spokes, xlabel="number of spokes N", ylabel="RoA size", title="RoA size versus number of spokes N", ax=axes[0])
plot.plot(n_spokes_values, floquet_by_n_spokes, xlabel="number of spokes N", ylabel="Floquet multiplier", title="Floquet multiplier versus number of spokes N", hlines=[1, -1], ax=axes[1])
fig.tight_layout()

print("graph appear")

plt.show()