# Assignment 1: Rimless Wheel Stability Analysis

## 1. Parameters used

For the main analysis I used:

| Parameter | Value |
|---|---:|
| Gravity $g$ | 9.81 m/s$^2$ |
| Mass $m$ | 0.25 kg |
| Leg length $L$ | 1 m |
| Damping $b$ | 0 |
| Number of spokes $N$ | 8 |
| Ground inclination $\gamma$ | 0.2 rad |

The timestep used for the simulations was 1e-2 s.

The initial conditions where 0.25 for theta and 0 for theta dot.

---

## 2. Sanity Checks

### 2.1 Zero gravity

The first sanity check was to set the gravity to zero keeping the current parameters. 
With theta dot = 0, we are expecting a flat total energy line and nothing else in the energy plot as nothing would be happening. 

The results confirm that behavior. We can clearly see a flat line.

<img src="images/set parameter g 0.png" width="500" alt="Energy plot for gravity = 0">

---

### 2.2 No slope

Second check was to keep all the parameters as initial and changing the slope to 0. 
The expected result is the behavior of a regular pendulum since the slope will not allow the total energy to increase and the wheel to keep rolling forward. We will just oscillate between two spokes.

Running the code did confirm that. If we compare the plot below to the one obtained in assignement 0, we can observe a similar behavior.

<img src="images/noslope.png" width="500" alt="none">

---

### 2.3 Theta boundaries and Spoke change behavior

With the set initial parameters, I was then able to look at the State Space plot and RoA map to check two things. First, that the theta boundaries were repected. Those are expected to be between the slope angle +/- half the angle between two spokes. Second, that the spoke change was indeed happening. To check that, I looked at if my trajectory was jumping from one end of the boundary to the other. 

The graph below confirms both those checks. We can see the clear theta bounds as nothing is plotted beyond them, and the trajectory is cleary bouncing from on to the other (the bounce itself beeing "invisible" as it is not actually part of the trajectory).

<img src="images/RoA_map.png" width="500" alt="none">

---

## 3. Region of Attraction

<img src="images/RoA_map.png" width="500" alt="none">

---

## 4. Poincaré Return Map

<img src="images/Poincaré_map.png" width="500" alt="none">

---

## 5. How the slope and number of spokes affect the RoA and local convergence

I am not sure my RoA plot is fully right. I tried to fix it but I am not sure about what we actually want and how to get it. 

The plots I have use the number of walking points over the total of points as what is region of attraction.

However what they seem to say is that the number of spokes does not have that much impact on the RoA as soon as it is above 6, same for floquet multiplier. While the gamma shows a sharp decline in RoA at around 0.42 radians with a floquet multiplier that shows a more stable attractor starting at this slope angle.

<img src="images/N_spokes_sweep.png" width="500" alt="none">
<img src="images/gamma_sweep.png" width="500" alt="none">

---

## 6. How to Run the Code

The main analysis is run from `assignment_1.py`.

From the assignment directory, run:

```bash
python assignment_1.py
```
---

## 7. AI disclosure

I did use AI to figure out how to write some fonctions. I have never coded a lot of those things before and needed help trying to figure out how to do it. 