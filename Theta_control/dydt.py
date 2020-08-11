import math
import numpy as np
from R_Earth import REarth

F_grav = 9.81
tau_Roll = 0.1
tau_T = 0.1
zeta = 1
wn = 16

def RK4Order(init_States, u, dt):
    current_states = init_States
    k1 = dydt(init_States, u)
    init_states = current_states + 0.5 * k1 * dt
    k2 = dydt(init_states, u)
    init_states = current_states + 0.5 * k2 * dt
    k3 = dydt(init_states, u)
    init_states = current_states + k3 * dt
    k4 = dydt(init_states, u)

    new_States = current_states + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Alpha dot Mod => -pi to pi
    if new_States[0] > math.pi:
        new_States[0] = new_States[2] - 2 * math.pi
    elif new_States[0] < -math.pi:
        new_States[0] = new_States[2] + 2 * math.pi

    # Alpha mod => -pi to pi
    if new_States[1] > math.pi:
        new_States[1] = new_States[6] - 2 * math.pi
    elif new_States[1] < -math.pi:
        new_States[1] = new_States[6] + 2 * math.pi

    # Phi Mod => -pi to pi
    if new_States[2] > math.pi:
        new_States[2] = new_States[2] - 2 * math.pi
    elif new_States[2] < -math.pi:
        new_States[2] = new_States[2] + 2 * math.pi

    # Psi mod => -pi to pi
    if new_States[4] > math.pi:
        new_States[4] = new_States[4] - 2 * math.pi
    elif new_States[4] < -math.pi:
        new_States[4] = new_States[4] + 2 * math.pi

    return new_States

def dydt(states, u):

    states_dot = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    r_earth = REarth(states[5])

    # Alpha dot
    states_dot[0] = -2 * zeta * wn * states[0] - wn * wn * states[1] + wn * wn * u[1]
    # Alpha
    states_dot[1] = states[0]

    # Roll
    states_dot[2] = (u[2] - states[2]) / tau_Roll
    # Velocity
    states_dot[3] = (u[0] - states[3]) / tau_T
    # Azimuth
    states_dot[4] = (F_grav / states[3]) * (math.sin(states[2]) / math.cos(states[1]))

    # Latitude
    states_dot[5] = states[3] * math.cos(states[1]) * math.cos(states[4]) / (r_earth[0] + states[7])
    # Longitude
    states_dot[6] = states[3] * math.cos(states[1]) * math.sin(states[4]) / (r_earth[1] + states[7]) / math.cos(
        states[6])
    # Altitude
    states_dot[7] = -1 * -1 * states[3] * math.sin(states[1])

    return states_dot
