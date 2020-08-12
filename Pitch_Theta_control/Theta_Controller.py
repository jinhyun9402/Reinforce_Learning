# Import Basic Function
import sys
import numpy as np
import math

# Import DQN Parameters
from Agent_controller import DQNController
from RewardFun import RewardErr

# Import Dynamics
from dydt import RK4Order

# Constant Parameters
D2R = math.pi/180
R2D = 180/math.pi
e = 0

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

EPISODES = 500
Epoch = 10
theta_scores, psi_scores, episodes = [], [], []
load_flag = False
Theta_Agent = DQNController(2, 3, load_flag)
Psi_Agent = DQNController(2, 3, load_flag)

while True:
    theta_done = False
    psi_done = False
    e = e + 1
    theta_score = 0
    psi_score = 0
    theta_r = 0
    psi_r = 0
    timer = 0
    mean = 0
    sB = Blue_States                    # States
    uB = [sB[3], sB[1], sB[2]]          # Command(Velocity, Alpha, Phi)

    # New Theta Angle
    sB[1] = np.random.randint(-10, 10) * D2R
    sB[4] = np.random.randint(-10, 10) * D2R
    theta_cmd = sB[1] + np.random.randint(-10, 10) * D2R
    psi_cmd = sB[4] + np.random.randint(-10, 10) * D2R
    while not theta_done and not psi_done:
        # Theta
        theta_Err = theta_cmd - sB[1]
        if theta_Err > math.pi:
            theta_Err = theta_Err - 360 * D2R
        elif theta_Err < -math.pi:
            theta_Err = theta_Err + 360 * D2R
        Theta_States = [theta_Err, uB[1]]
        action_Theta_index = Theta_Agent.get_action(Theta_States)
        # Psi
        psi_Err = psi_cmd - sB[4]
        if psi_Err > math.pi:
            psi_Err = psi_Err - 360 * D2R
        elif psi_Err < -math.pi:
            psi_Err = psi_Err + 360 * D2R
        Psi_States = [psi_Err, uB[2]]
        action_Psi_index = Psi_Agent.get_action(Psi_States)

        # Action 0 = Stay, 1 = +1deg Command, 2 = -1deg Command
        # Theta
        if action_Theta_index == 2:
            dtheta_Cmd = -1
        else:
            dtheta_Cmd = action_Theta_index
        # Psi
        if action_Psi_index == 2:
            dpsi_Cmd = -1
        else:
            dpsi_Cmd = action_Psi_index

        uB[1] = uB[1] + dtheta_Cmd * D2R
        uB[2] = uB[2] + dpsi_Cmd * D2R

        # State Update
        snB = RK4Order(sB, uB, dt)

        # New DQN States
        # Theta
        theta_Err = theta_cmd - snB[1]
        if theta_Err > math.pi:
            theta_Err = theta_Err - 360 * D2R
        elif theta_Err < -math.pi:
            theta_Err = theta_Err + 360 * D2R
        ThetaN_States = [theta_Err, uB[1]]
        theta_r, theta_done = RewardErr(theta_Err)
        # Psi
        psi_Err = psi_cmd - snB[4]
        if psi_Err > math.pi:
            psi_Err = psi_Err - 360 * D2R
        elif psi_Err < -math.pi:
            psi_Err = psi_Err + 360 * D2R
        PsiN_States = [psi_Err, uB[2]]
        psi_r, psi_done = RewardErr(psi_Err)

        # DQN State Append = [state action reward next_state flag]
        Theta_Agent.append_sample(Theta_States, action_Theta_index, theta_r, ThetaN_States, theta_done)
        Psi_Agent.append_sample(Psi_States, action_Psi_index, psi_r, PsiN_States, psi_done)

        # Training Model
        # Theta
        if len(Theta_Agent.memory) >= Theta_Agent.train_start:
            Theta_Agent.train_model()
        # Psi
        if len(Psi_Agent.memory) >= Psi_Agent.train_start:
            Psi_Agent.train_model()

        theta_score += theta_r
        psi_score += psi_r

        # new State
        sB = snB

        timer = timer + dt

        if timer > Epoch:
            theta_done = True
            psi_done = True
            # Theta
            if abs(theta_cmd - sB[1]) <= 2 * D2R:
                theta_scored = theta_score
                theta_score = 1000
            else:
                theta_scored = theta_score
                theta_score = -1000
            # Psi
            if abs(psi_cmd - sB[4]) <= 2 * D2R:
                psi_scored = psi_score
                psi_score = 1000
            else:
                psi_scored = psi_score
                psi_score = -1000

        if theta_done and psi_done:
            # Target Model Update
            Theta_Agent.update_target_model()
            Psi_Agent.update_target_model()

            theta_scores.append(theta_score)
            psi_scores.append(psi_score)
            episodes.append(e)

            print("episode:", e, " Theta score:", theta_score, "(", round(theta_scored), ")",
                  " Psi score:", psi_score, "(", round(psi_scored), ")", " time:", round(timer),
                  " theta cmd:", round(theta_cmd * R2D, 1), " Theta:", round(sB[1] * R2D, 1),
                  " psi cmd:", round(psi_cmd * R2D, 1), " Psi:", round(sB[4] * R2D, 1))
            if np.mean(theta_scores[-min(10, len(theta_scores)):]) >= 1000 and np.mean(psi_scores[-min(10, len(psi_scores)):]) >= 1000 and e > 10:
                Theta_Agent.model.save("./save_model/DQNController_theta.h5")
                Psi_Agent.model.save("./save_model/DQNController_psi.h5")
                sys.exit()
