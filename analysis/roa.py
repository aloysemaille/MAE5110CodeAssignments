import numpy as np
import matplotlib.pyplot as plt
from integrators import rk4 as integrator


def simulate_until_converged(initial_state,params,model_dynamics,model_trigger,timestep,max_sim_time=30.0,
                             convergence_tolerance=1e-2,n_consecutive=3,stopped_tolerance=1e-2):
    current_time=0.0
    current_state=np.asarray(initial_state,dtype=float).copy()
    impact_angular_velocities=[]

    while current_time<max_sim_time:
        dt=min(timestep,max_sim_time-current_time)
        _,step_states=integrator.rk4(dt,params,model_dynamics,current_state,dt)
        state_after_step=step_states[:,-1]
        current_state=model_trigger(current_time,state_after_step,params)
        current_time+=dt
        impact_angular_velocities.append(current_state[1])

        if abs(current_state[1])<stopped_tolerance:
            return "stopped",current_time

        if len(impact_angular_velocities)>=n_consecutive:
            recent_impacts=impact_angular_velocities[-n_consecutive:]
            if max(recent_impacts)-min(recent_impacts)<convergence_tolerance:
                return "walking",current_time

    return "unresolved",current_time


def build_roa_grid(params,model_dynamics,model_trigger,theta_bounds,timestep,number_of_bounds=25,
                   angular_velocity_bounds=(-10,10)):
    theta_range=np.linspace(theta_bounds[0],theta_bounds[1],number_of_bounds)
    angular_velocity_range=np.linspace(angular_velocity_bounds[0],angular_velocity_bounds[1],number_of_bounds)
    all_status=np.empty((number_of_bounds,number_of_bounds),dtype=object)

    for i,angular_velocity_0 in enumerate(angular_velocity_range):
        for j,theta_0 in enumerate(theta_range):
            initial_state=np.array([theta_0,angular_velocity_0])
            all_status[i,j]=simulate_until_converged(initial_state,params,model_dynamics,model_trigger,
                                                     timestep)[0]

    return theta_range,angular_velocity_range,all_status


def compute_walking_fraction(results):
    if results.size==0:
        return 0.0
    return np.sum(results=="walking")/results.size


def plot_roa_map(theta_range,theta_dot_range,results,trajectory,model_dynamics,params,
                 title="Region of Attraction map"):
    plt.figure(figsize=(10,6))

    theta_min=-2*np.pi
    theta_max=2*np.pi
    #n_tiles=int(np.ceil((theta_max-theta_min)/(theta_range[-1]-theta_range[0])))
    #tiled_results=np.tile(results,(1,n_tiles))
    #tiled_theta=np.linspace(theta_min,theta_max,tiled_results.shape[1])

    #color_matrix=np.zeros(tiled_results.shape)
    #color_matrix[tiled_results=="walking"]=1.0
    #color_matrix[tiled_results=="unresolved"]=0.5

    #cmap=plt.colormaps["RdYlGn"].resampled(3)

    #plt.imshow(color_matrix,extent=[theta_min,theta_max,theta_dot_range[0],theta_dot_range[-1]],
    #           origin="lower",cmap=cmap,alpha=0.35,aspect="auto")

    _plot_vector_field(theta_min,theta_max,theta_dot_range,model_dynamics,params)

    plt.plot(trajectory[:,0],trajectory[:,1],color="black",linewidth=2,label="trajectory")
    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def _plot_vector_field(theta_min,theta_max,theta_dot_range,model_dynamics,params,quiver_density=50):
    theta=np.linspace(theta_min,theta_max,quiver_density)
    theta_dot=np.linspace(theta_dot_range[0],theta_dot_range[-1],quiver_density)
    grid_theta,grid_theta_dot=np.meshgrid(theta,theta_dot)

    theta_derivative=np.zeros_like(grid_theta)
    theta_dot_derivative=np.zeros_like(grid_theta_dot)

    for i in range(grid_theta.shape[0]):
        for j in range(grid_theta.shape[1]):
            state=np.array([grid_theta[i,j],grid_theta_dot[i,j]])
            derivative=model_dynamics(0,state,params)
            theta_derivative[i,j]=derivative[0]
            theta_dot_derivative[i,j]=derivative[1]

    magnitude=np.sqrt(theta_derivative**2+theta_dot_derivative**2)
    magnitude[magnitude==0]=1.0
    theta_derivative/=magnitude
    theta_dot_derivative/=magnitude

    plt.quiver(grid_theta,grid_theta_dot,theta_derivative,theta_dot_derivative,color="gray",alpha=0.5,scale=25)
    quiver_theta = np.linspace(theta_min, theta_max, quiver_density)

    quiver_theta_dot = np.linspace(theta_dot_range[0], theta_dot_range[-1], quiver_density)

    grid_theta, grid_theta_dot = np.meshgrid(quiver_theta, quiver_theta_dot)

    theta_derivative = np.zeros_like(grid_theta)
    theta_dot_derivative = np.zeros_like(grid_theta_dot)

    for i in range(grid_theta.shape[0]):
        for j in range(grid_theta.shape[1]):

            state = np.array([grid_theta[i, j],grid_theta_dot[i, j]])

            derivative = model_dynamics(0,state,params)

            theta_derivative[i, j] = derivative[0]
            theta_dot_derivative[i, j] = derivative[1]

    magnitude = np.sqrt(theta_derivative**2 + theta_dot_derivative**2)

    magnitude[magnitude == 0] = 1.0

    theta_derivative /= magnitude
    theta_dot_derivative /= magnitude

    plt.quiver(grid_theta, grid_theta_dot, theta_derivative, theta_dot_derivative, color="gray", 
               alpha=0.5, scale=25)