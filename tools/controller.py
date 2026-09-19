import numpy as np


def ankle_torque_bounds(params):
    mass = params["mass"]
    gravity = params["gravity"]
    length = params["length"]
    lower_bound = -0.1 * mass * gravity * length
    upper_bound = 0.05 * mass * gravity * length
    return lower_bound, upper_bound


def feedback_linearizing_torque(state, params, proportional_gain=15.0, derivative_gain=6.0):
    theta, angular_velocity = state
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    torque = -mass * length**2 * (
        (gravity / length) * np.sin(theta)
        + proportional_gain * theta
        + derivative_gain * angular_velocity
    )
    return torque


def choose_ankle_torque(state, params, proportional_gain=15.0, derivative_gain=6.0):
    torque = feedback_linearizing_torque(state, params, proportional_gain, derivative_gain)
    lower_bound, upper_bound = ankle_torque_bounds(params)
    return np.clip(torque, lower_bound, upper_bound)