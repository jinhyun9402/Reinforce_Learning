# Import Basic Function
import sys
import numpy as np
import math
import pylab
from mpl_toolkits.mplot3d import Axes3D

# Import Dynamics
from dydt import RK4Order

from keras.models import load_model

# Constant Parameters
D2R = math.pi/180
R2D = 180/math.pi
e = 0
timer = 0
Epoch = 10

states, command, time = [], [], []
theta_commnad = []
Lat, Long, Alt = [], [], []

done = False

# Simulation initialize
dt = 0.1
Blue_States = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
Blue_States[0] = 0                       # Alpha dot
Blue_States[1] = 0                       # Alpha
Blue_States[2] = 0                       # Phi
Blue_States[3] = 250                     # Velocity
Blue_States[4] = 0                       # Psi
Blue_States[5] = 37 * D2R                # Latitude
Blue_States[6] = 126 * D2R               # Longitude
Blue_States[7] = 10000                   # Altitude

model = load_model('./save_model/DQNTheta10v1.h5')


sB = Blue_States                    # States
uB = [sB[3], sB[1], sB[2]]          # Command(Velocity, Alpha, Phi)

# New Theta Angle
sB[1] = np.random.randint(-10, 10) * D2R
theta_cmd = sB[1] + np.random.randint(-10, 10) * D2R

while not done:
    theta_Err = theta_cmd - sB[1]
    if theta_Err > math.pi:
        theta_Err = theta_Err - 360 * D2R
    elif theta_Err < -math.pi:
        theta_Err = theta_Err + 360 * D2R
    Theta_States = [theta_Err, uB[1]]
    Theta_States = np.reshape(Theta_States, [1,2])
    q_value = model(Theta_States)
    action_index = np.argmax(q_value[0])

    # Action 0 = Stay, 1 = +1deg Command, 2 = -1deg Command
    if action_index == 2:
        dCmd = -1
    else:
        dCmd = action_index

    uB[1] = uB[1] + dCmd * D2R

    # State Update
    snB = RK4Order(sB, uB, dt)

    # New DQN States
    theta_Err = theta_cmd - snB[1]
    if theta_Err > math.pi:
        theta_Err = theta_Err - 360 * D2R
    elif theta_Err < -math.pi:
        theta_Err = theta_Err + 360 * D2R
    ThetaN_States = [theta_Err, uB[1]]

    # new State
    sB = snB

    # Data append
    states.append(sB[1] * R2D)
    command.append(uB[1] * R2D)
    time.append(timer)
    Lat.append(sB[5] * R2D)
    Long.append(sB[6] * R2D)
    Alt.append(sB[7])

    timer = timer + dt

    if timer > Epoch:
        done = True

        # Plot
        pylab.figure(1)
        pylab.plot(time, states, label='Theta State')
        pylab.plot(time, command, label='Theta Command')
        pylab.axhline(theta_cmd * R2D, color='k', linestyle='dashed', linewidth=1, label='Desire Theta')
        pylab.xlabel('time')
        pylab.ylabel('angle')
        pylab.grid()
        pylab.legend()

        pylab.figure(2)
        pylab.gca(projection='3d')
        pylab.plot(Lat, Long, Alt)
        pylab.title('')
        pylab.show()