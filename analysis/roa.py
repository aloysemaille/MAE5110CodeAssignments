import numpy as np
import matplotlib.pyplot as plt
from integrators import rk4 as integrator
from matplotlib.patches import Patch


def simulate_until_converged(initial_state,params,model_dynamics,model_trigger,timestep,max_sim_time=30.0,
                             convergence_tolerance=1e-2,n_consecutive=3,stopped_tolerance=1e-2):
    current_time=0.0
    current_state=np.asarray(initial_state,dtype=float).copy()
    impact_angular_velocities=[]

    while current_time<max_sim_time:
        dt=min(timestep,max_sim_time-current_time)
        state_after_step=integrator.rk4_step(dt,params,model_dynamics,current_state)
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
                   angular_velocity_bounds=(-5,5)):
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


def _plot_vector_field(theta_min,theta_max,theta_dot_min,theta_dot_max,model_dynamics,params,
                       quiver_density=15):
    theta=np.linspace(theta_min,theta_max,quiver_density)
    theta_dot=np.linspace(theta_dot_min,theta_dot_max,quiver_density)
    grid_theta,grid_theta_dot=np.meshgrid(theta,theta_dot)

    theta_derivative=np.zeros_like(grid_theta)
    theta_dot_derivative=np.zeros_like(grid_theta_dot)

    for i in range(grid_theta.shape[0]):
        for j in range(grid_theta.shape[1]):
            state=np.array([grid_theta[i,j],grid_theta_dot[i,j]])
            derivative=model_dynamics(0,state,params)
            theta_derivative[i,j]=derivative[0]
            theta_dot_derivative[i,j]=derivative[1]

    theta_span=theta_max-theta_min
    theta_dot_span=theta_dot_max-theta_dot_min

    vx=theta_derivative/theta_span
    vy=theta_dot_derivative/theta_dot_span

    magnitude=np.sqrt(vx**2+vy**2)
    magnitude[magnitude==0]=1.0
    vx/=magnitude
    vy/=magnitude

    arrow_length_fraction=0.4
    dx=vx*theta_span*(arrow_length_fraction/quiver_density)
    dy=vy*theta_dot_span*(arrow_length_fraction/quiver_density)

    plt.quiver(grid_theta,grid_theta_dot,dx,dy,color="gray",alpha=0.6,pivot="mid",angles="xy",scale_units="xy",scale=0.8,width=0.002,headwidth=4,headlength=3)
    plt.streamplot(grid_theta,grid_theta_dot,theta_derivative,theta_dot_derivative,color="cyan",linewidth=0.3,density=1.0,arrowsize=0)

def plot_roa_map(theta_range,theta_dot_range,results,trajectory,model_dynamics,params,
                 title="Region of Attraction map"):
    plt.figure(figsize=(10,6))

    theta_min=theta_range[0]
    theta_max=theta_range[-1]
    theta_dot_min=-5.0
    theta_dot_max=5.0

    color_matrix=np.zeros(results.shape)
    color_matrix[results=="walking"]=1.0
    color_matrix[results=="unresolved"]=0.5

    cmap=plt.colormaps["RdYlGn"].resampled(3)

    plt.imshow(color_matrix,extent=[theta_min,theta_max,theta_dot_range[0],theta_dot_range[-1]],
              origin="lower",cmap=cmap,alpha=0.35,aspect="auto")

    _plot_vector_field(theta_min,theta_max,theta_dot_min,theta_dot_max,model_dynamics,params)

    angle_between_adjacent_spokes = 2*np.pi/params["number_spokes"]
    theta_plot, theta_dot_plot = _break_trajectory_at_resets(trajectory, angle_between_adjacent_spokes)
    plt.plot(theta_plot, theta_dot_plot, color="black", linewidth=1, label="trajectory")

    margin=0.15*(theta_max-theta_min)
    plt.xlim(theta_min-margin,theta_max+margin)
    plt.axvline(0, color="white", linewidth=0.8)
    plt.axhline(0, color="white", linewidth=0.8)

    walking_patch = Patch(color=cmap(1.0), alpha=0.35, label="walking")
    stopped_patch = Patch(color=cmap(0.0), alpha=0.35, label="stopped")
    unresolved_patch = Patch(color=cmap(0.5), alpha=0.35, label="unresolved")
    trajectory_line = plt.Line2D([], [], color="black", linewidth=1, label="trajectory")

    plt.xlabel("theta (rad)")
    plt.ylabel("theta_dot (rad/s)")
    plt.title(title)
    plt.legend(handles=[trajectory_line, walking_patch, unresolved_patch, stopped_patch])
    plt.tight_layout()

def _break_trajectory_at_resets(trajectory, angle_between_adjacent_spokes, tolerance_fraction=0.5):
    theta = trajectory[:,0]
    theta_dot = trajectory[:,1]

    threshold = tolerance_fraction * angle_between_adjacent_spokes
    jumps = np.abs(np.diff(theta)) > threshold

    theta_plot = np.insert(theta, np.where(jumps)[0]+1, np.nan)
    theta_dot_plot = np.insert(theta_dot, np.where(jumps)[0]+1, np.nan)

    return theta_plot, theta_dot_plot