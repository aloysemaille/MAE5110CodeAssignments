import numpy as np

from models import pendulum as model
from integrators import rk4 as integrator

# Generate params
params_undamped = model.generate_params()
initial_state = model.generate_initial_condition()

# Generate initial conditions
sim_time = 1.0
time_step = 0.01
n_steps = int(sim_time / time_step) + 1
time_traj = np.arange(n_steps) * time_step
state_traj = np.zeros((2, n_steps))
state_traj[:, 0] = initial_state

undamped_derivative = np.zeros((2, n_steps))
damped_derivative = np.zeros((2, n_steps))

# Calculate intial energy
initial_ke, initial_pe = model.calculate_energy(state_traj[:, 0], params_undamped)
initial_energy = initial_ke + initial_pe
energy_conserved = True

# Run simulation loop for 100 steps
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator(model.dynamics, t, state_traj[:, step], time_step, params_undamped)
    next_ke, potential_ke = model.calculate_energy(state_traj[:, step + 1], params_undamped)
    next_energy = next_ke + potential_ke
    
    if not np.isclose(
        initial_energy,
        next_energy,
        rtol=1e-5,
        atol=1e-6
    ):
        energy_conserved = False
        print("Energy conservation tolerance exceeded")
        print(f"Time:           {time_traj[step + 1]:.3f} s")
        print(f"Initial energy: {initial_energy:.10f}")
        print(f"Current energy: {next_energy:.10f}")
        print(f"Difference:     {next_energy - initial_energy:.10e}")
        break

# Check if damping is implemented correctly
# Here we compare if the energy at the next time step is continuously decreasing
# In addition, if the difference in accelration is equivalent to the ratio of -b * v / (ml^2)
params_damped = params_undamped.copy()
params_damped["damping_coeff"] = 0.1

state_traj_damped = np.zeros((2, n_steps))
state_traj_damped[:, 0] = initial_state

damping_works = True

for step, t in enumerate(time_traj[:-1]):
    initial_ke, initial_pe = model.calculate_energy(state_traj_damped[:, step], params_damped)
    initial_energy = initial_ke + initial_pe
    
    damped_derivative[:, step] = model.dynamics(t, state_traj_damped[:, step], params_damped)
    undamped_derivative = model.dynamics(t, state_traj_damped[:, step], params_undamped)
    
    state_traj_damped[:, step + 1] = integrator(model.dynamics, t, state_traj_damped[:, step], time_step, params_damped)
    
    next_ke, next_pe = model.calculate_energy(state_traj_damped[:, step + 1], params_damped)
    next_energy = next_ke + next_pe
    
    if (
        next_energy > initial_energy
        or
        not np.isclose(
            damped_derivative[1, step] - undamped_derivative[1],
            (-params_damped["damping_coeff"] * damped_derivative[0, step])
            / (params_damped["mass"] * params_damped["length"] ** 2),
            rtol=1e-5,
            atol=1e-6
        ) 
    ):
        damping_works = False
        print("Damping check failed")
        break


# Check if torque is implemented correctly
# We check if the torque is implemented correctly by setting the acceleration of outputted equal to max acceleration
# caused by graviy, because if both values are equal, the pendulum should remain fixed.
params_torque = params_undamped.copy()
params_torque["torque"] = -(params_torque["mass"] * params_torque["gravity"] * params_torque["length"] * np.sin(initial_state[0]))
state_traj_torque = np.zeros((2, n_steps))
state_traj_torque[:, 0] = initial_state

torque_works = True

for step, t in enumerate(state_traj[:, -1]):
    state_traj_torque[:, step + 1] = integrator(model.dynamics, t, state_traj_torque[:, step], time_step, params_torque)
    
    if  state_traj_torque[1, step] != 0:
        torque_works = False
        print("Torque is not implemented correctly")
        print(state_traj_torque[1, step])
        print(state_traj_torque[1, step + 1])
        break

if not (energy_conserved or damping_works or torque_works):
    print("One implementation did not work correctly, check logs.")
else:
    print("All implementations work correctly.")