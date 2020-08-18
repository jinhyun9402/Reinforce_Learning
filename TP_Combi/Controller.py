# Import Basic Function
import sys
import numpy as np
import math

# Import DQN Parameters
from Agent import DQNController
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

EPISODES = 2
Epoch = 10
scores, episodes = [], []

load_flag = True
Agent = DQNController(4, 6, load_flag)

while True:
    done = False
    e = e + 1
    score = 0
    scored = 0
    r = 0
    timer = 0
    mean = 0
    sB = Blue_States                    # States
    uB = [sB[3], sB[1], sB[2]]          # Command(Velocity, Alpha, Phi)

    # New Theta Angle
    sB[1] = np.random.randint(-10, 10) * D2R    # Alpha random value
    sB[4] = np.random.randint(-10, 10) * D2R    # Psi random value
    theta_cmd = sB[1] + np.random.randint(-10, 10) * D2R
    psi_cmd = sB[4] + np.random.randint(-10, 10) * D2R
    while not done:
        # Theta
        theta_Err = theta_cmd - sB[1]
        if theta_Err > math.pi:
            theta_Err = theta_Err - 360 * D2R
        elif theta_Err < -math.pi:
            theta_Err = theta_Err + 360 * D2R
        Theta_States = [theta_Err, uB[1]]
        # Psi
        psi_Err = psi_cmd - sB[4]
        if psi_Err > math.pi:
            psi_Err = psi_Err - 360 * D2R
        elif psi_Err < -math.pi:
            psi_Err = psi_Err + 360 * D2R
        Psi_States = [psi_Err, uB[2]]

        DQN_states = [theta_Err, uB[1], psi_Err, uB[2]]
        action_Theta_index, action_Phi_index = Agent.get_action(DQN_states)
        action_index = [action_Theta_index, action_Phi_index]

        # Action 0 = Stay, 1 = +1deg Command, 2 = -1deg Command
        # Theta
        if action_Theta_index == 2:
            dtheta_Cmd = -1
        else:
            dtheta_Cmd = action_Theta_index
        # Psi
        if action_Phi_index == 2:
            dphi_Cmd = -1
        else:
            dphi_Cmd = action_Phi_index

        uB[1] = uB[1] + dtheta_Cmd * D2R
        uB[2] = uB[2] + dphi_Cmd * D2R

        # State Update
        snB = RK4Order(sB, uB, dt)

        # New DQN States
        # Theta
        theta_Err = theta_cmd - snB[1]
        if theta_Err > math.pi:
            theta_Err = theta_Err - 360 * D2R
        elif theta_Err < -math.pi:
            theta_Err = theta_Err + 360 * D2R
        # ThetaN_States = [theta_Err, uB[1]]
        # Psi
        psi_Err = psi_cmd - snB[4]
        if psi_Err > math.pi:
            psi_Err = psi_Err - 360 * D2R
        elif psi_Err < -math.pi:
            psi_Err = psi_Err + 360 * D2R
        # PsiN_States = [psi_Err, uB[2]]

        DQN_next_states = [theta_Err, uB[1], psi_Err, uB[2]]

        r, done = RewardErr(theta_Err, psi_Err)

        # DQN State Append = [state action reward next_state flag]
        Agent.append_sample(DQN_states, action_index, r, DQN_next_states, done)

        # Training Model
        # Theta
        if len(Agent.memory) >= Agent.train_start:
            Agent.train_model()

        score += r

        # new State
        sB = snB

        timer = timer + dt

        if timer > Epoch:
            done = True
            # Theta
            if abs(theta_cmd - sB[1]) <= 5 * D2R and abs(psi_cmd - sB[4]) <= 5 * D2R:
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

            print("episode:", e, " Score:", score, "(", scored, ")", " time:", round(timer),
                  " theta cmd:", round(theta_cmd * R2D, 1), " Theta:", round(sB[1] * R2D, 1),
                  " psi cmd:", round(psi_cmd * R2D, 1), " Psi:", round(sB[4] * R2D, 1))
            if np.mean(scores[-min(15, len(scores)):]) >= 1000 and e > 10:
                Agent.model.save("./save_model/DQN_TPC_NewWeight.h5")
                sys.exit()