import math
D2R = math.pi/180
R2D = 180/math.pi

def RewardErr(Err):
    Err_deg = Err * R2D
    PsiFlag = False
    P = abs(Err_deg)
    if P <= 2:
        PsiReward = 1
    elif P > 180:
        PsiReward = -1000
        PsiFlag = True
    else:
        PsiReward = 1/P

    return PsiReward, PsiFlag