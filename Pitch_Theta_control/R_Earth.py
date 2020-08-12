import numpy as np
import math
Re = 6378137
Ecc_Earth = 0.0818191908426

def REarth(Lat):
    r_earth = np.array([0, 0], dtype=np.float64)
    r_earth[0] = Re*(1-Ecc_Earth*Ecc_Earth)/(1-Ecc_Earth*Ecc_Earth*math.sin(Lat)*math.sin(Lat))**1.5
    r_earth[1] = Re/(1-Ecc_Earth*Ecc_Earth * math.sin(Lat)*math.sin(Lat))**0.5
    return r_earth
