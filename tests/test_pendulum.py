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


def test_energy_conservation():
    state_traj = np.zeros((2, n_steps))
    state_traj[:, 0] = initial_state
    
    # Run simulation loop for 100 steps
    for step, t in enumerate(time_traj[:-1]):
        
        initial_ke, initial_pe = model.calculate_energy(state_traj[:, 0], params_undamped)
        initial_energy = initial_ke + initial_pe
        
        state_traj[:, step + 1] = integrator(model.dynamics, t, state_traj[:, step], time_step, params_undamped)
        next_ke, potential_ke = model.calculate_energy(state_traj[:, step + 1], params_undamped)
        next_energy = next_ke + potential_ke
        
        assert np.isclose(
            initial_energy,
            next_energy,
            rtol=1e-5,
            atol=1e-6
        ), (
            f"Energy conservation tolerance exceeded\n"
            f"Time: {time_traj[step + 1]:.3f} s\n"
            f"Initial energy: {initial_energy:.10f}\n"
            f"Current energy: {next_energy:.10f}\n"
            f"Difference: {next_energy - initial_energy:.10e}"
        )

def test_damping():
    # Check if damping is implemented correctly
    # Here we compare if the energy at the next time step is continuously decreasing
    # In addition, if the difference in accelration is equivalent to the ratio of -b * v / (ml^2)
    params_damped = params_undamped.copy()
    params_damped["damping_coeff"] = 0.1

    state_traj_damped = np.zeros((2, n_steps))
    state_traj_damped[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        undamped_derivative = np.zeros((2, n_steps))
        damped_derivative = np.zeros((2, n_steps))


        initial_ke, initial_pe = model.calculate_energy(state_traj_damped[:, step], params_damped)
        initial_energy = initial_ke + initial_pe
        
        damped_derivative[:, step] = model.dynamics(t, state_traj_damped[:, step], params_damped)
        undamped_derivative = model.dynamics(t, state_traj_damped[:, step], params_undamped)
        
        state_traj_damped[:, step + 1] = integrator(model.dynamics, t, state_traj_damped[:, step], time_step, params_damped)
        
        next_ke, next_pe = model.calculate_energy(state_traj_damped[:, step + 1], params_damped)
        next_energy = next_ke + next_pe
        
        assert next_energy <= initial_energy + 1e-10, (
            f"Damping energy check failed at "
            f"{time_traj[step + 1]:.3f} s"
        )
        
        assert np.isclose(
            damped_derivative[1, step] - undamped_derivative[1],
            (-params_damped["damping_coeff"] * damped_derivative[0, step])
            / (
                params_damped["mass"]
                * params_damped["length"] ** 2
            ),
            rtol=1e-5,
            atol=1e-6
        ), (
            f"Damping acceleration check failed at "
            f"{time_traj[step]:.3f} s"
        )

def test_torque():
    # Check if torque is implemented correctly
    # We check if the torque is implemented correctly by setting the acceleration of outputted equal to max acceleration
    # caused by graviy, because if both values are equal, the pendulum should remain fixed.
    params_torque = params_undamped.copy()
    params_torque["torque"] = -(params_torque["mass"] * params_torque["gravity"] * params_torque["length"] * np.sin(initial_state[0]))
    state_traj_torque = np.zeros((2, n_steps))
    state_traj_torque[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj_torque[:, step + 1] = integrator(model.dynamics, t, state_traj_torque[:, step], time_step, params_torque)
        
        assert np.isclose(
            state_traj_torque[1, step + 1],
            0.0,
            rtol=1e-5,
            atol=1e-6
        ), (
            f"Torque velocity check failed at "
            f"{time_traj[step + 1]:.3f} s"
        )

        assert np.isclose(
            state_traj_torque[0, step + 1],
            initial_state[0],
            rtol=1e-5,
            atol=1e-6
        ), (
            f"Torque angle check failed at "
            f"{time_traj[step + 1]:.3f} s"
        )