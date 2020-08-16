import math
D2R = math.pi/180
R2D = 180/math.pi

def RewardErr(theta_Err, psi_Err):
    theta_Err_deg = theta_Err * R2D
    psi_Err_deg = psi_Err * R2D
    Flag = False
    T = abs(theta_Err_deg)
    P = abs(psi_Err_deg)
    if T <= 4:
        if P <= 4:
            Reward = 1
        else:
            Reward = 1/P
    elif T > 90 or P > 180:
        Reward = -1000
        Flag = True
    else:
        if P <= 4:
            Reward = 1/T
        else:
            Reward = (1/T + 1/P)/2

    return Reward, Flag