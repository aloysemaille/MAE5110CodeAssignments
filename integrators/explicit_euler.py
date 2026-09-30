def explicit_euler(dynamics, t, state, timestep, params):

    new_state = state + timestep * dynamics(t, state, params)
    
    return new_state