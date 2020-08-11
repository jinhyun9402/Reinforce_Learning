# Import Basic Function
import sys
import numpy as np
import math

# Import DQN Parameters
from Agent_Theta import DQNTheta
from RewardFun import RewardThetaErr

# Import Dynamics
from dydt import RK4Order

# Constant Parameters
D2R = math.pi/180
R2D = 180/math.pi
e = 0

# Simulation initialize
dt = 0.05
Blue_States = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
Blue_States[0] = 0                       # Alpha dot
Blue_States[1] = 0                       # Alpha
Blue_States[2] = 0                       # Phi
Blue_States[3] = 250                     # Velocity
Blue_States[4] = 0                       # Psi
Blue_States[5] = 37 * D2R                # Latitude
Blue_States[6] = 126 * D2R               # Longitude
Blue_States[7] = 10000                   # Altitude

EPISODES = 500
Epoch = 10
scores, episodes = [], []
load_flag = False
Agent = DQNTheta(2, 3, load_flag)

while True:
    done = False
    e = e + 1
    score = 0
    r = 0
    timer = 0
    mean = 0
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
        action_index = Agent.get_action(Theta_States)

        # Action 0 = Stay, 1 = +1deg Command, 2 = -1deg Command
        if action_index == 2:
            dCmd = -1
        else:
            dCmd = action_index

        uB[1] = uB[1] + dCmd * 0.5 * D2R

        # State Update
        snB = RK4Order(sB, uB, dt)

        # New DQN States
        theta_Err = theta_cmd - snB[1]
        if theta_Err > math.pi:
            theta_Err = theta_Err - 360 * D2R
        elif theta_Err < -math.pi:
            theta_Err = theta_Err + 360 * D2R
        ThetaN_States = [theta_Err, uB[1]]
        r, done = RewardThetaErr(theta_Err)

        # DQN State Append = [state action reward next_state flag]
        Agent.append_sample(Theta_States, action_index, r, ThetaN_States, done)

        # Training Model
        if len(Agent.memory) >= Agent.train_start:
            Agent.train_model()

        score += r

        # new State
        sB = snB

        timer = timer + dt

        if timer > Epoch:
            done = True
            if abs(theta_cmd - sB[1]) <= 2 * D2R:
                scored = score
                score = 1000
            else:
                scored = score
                score = -1000

        if done:
            # Target Model Update
            Agent.update_target_model()

            scores.append(score)
            episodes.append(e)
            print("episode:", e, "  score:", score, "(", scored, ")", "  time:",
                  round(timer), "  cmd:", round(theta_cmd * R2D, 1), " Theta:", round(sB[1] * R2D, 1))
            if np.mean(scores[-min(10, len(scores)):]) >= 1000 and e > 10:
                Agent.model.save("./save_model/DQNTheta10v1.h5")
                sys.exit()
