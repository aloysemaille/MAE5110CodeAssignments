import sys
sys.dont_write_bytecode=True

import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from analysis import roa
from analysis import poincare

params={
    "gravity":9.81,
    "length":1,
    "mass":0.25,
    "damping_coeff":0.0,
    "number_spokes":8,
    "ground_inclination":0.2,
}

initial_state=np.array([params["ground_inclination"]+0.05,0.0])

timestep=1e-2
simulation_time=30.0

theta_bounds,theta_post_reset=model.compute_theta_bounds(params)

time_history,state_history=model.simulate_trajectory(initial_state,params,timestep,simulation_time)

potential_energy,kinetic_energy=model.calculate_energy(state_history,params)
total_energy=potential_energy+kinetic_energy

# Poincare return map
theta_dot_range=np.linspace(0.5,6.0,40)

theta_dot_next=poincare.build_return_map(theta_dot_range,theta_post_reset,params,timestep=timestep)

fixed_point=poincare.find_fixed_point(theta_dot_range,theta_dot_next)

# Floquet multiplier
floquet_multiplier=None

if fixed_point is not None:
    floquet_multiplier=poincare.estimate_floquet_multiplier(fixed_point,theta_post_reset,params,timestep=timestep)

# Region of Attraction
roa_theta_range,roa_theta_dot_range,roa_results=roa.build_roa_grid(
    params,
    theta_bounds,
    timestep,
    fixed_point=fixed_point,
    number_of_bounds=25,
    angular_velocity_bounds=(-10,10)
)

# Ground inclination sweep
gamma_values=np.linspace(0.1,np.pi/3,8)

roa_size_by_gamma,floquet_by_gamma=poincare.run_stability_sweep("ground_inclination",gamma_values,params,timestep)

# Number of spokes sweep
n_spokes_values=np.arange(6,13)

roa_size_by_n_spokes,floquet_by_n_spokes=poincare.run_stability_sweep("number_spokes",n_spokes_values,params,timestep)

# Plots
model.plot_energy_vs_time(time_history,potential_energy,kinetic_energy,total_energy)

roa.plot_roa_map(
    roa_theta_range,
    roa_theta_dot_range,
    roa_results,
    state_history,
    model.dynamics,
    params,
    fixed_point=fixed_point,
    theta_post_reset=theta_post_reset,
    timestep=timestep,
    title=f"RoA map, gamma = {params['ground_inclination']:.3f} rad"
)

poincare.plot_return_map(theta_dot_range,theta_dot_next,fixed_point,
                         title=f"Return map, gamma = {params['ground_inclination']:.3f} rad")

if fixed_point is not None:
    print(f"Fixed point theta_dot*: {fixed_point:.4f} rad/s")

if floquet_multiplier is not None:
    print(f"Floquet multiplier: {floquet_multiplier:.4f}")
    print("Stable" if abs(floquet_multiplier)<1 else "Unstable")

poincare.plot_sweep_results(gamma_values,roa_size_by_gamma,floquet_by_gamma,"gamma (rad)")
poincare.plot_sweep_results(n_spokes_values,roa_size_by_n_spokes,floquet_by_n_spokes,"number of spokes N")

plt.show()
