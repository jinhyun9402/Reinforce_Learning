import numpy as np
from convert2csv import saveWB
from keras.models import load_model

#theta_agent = load_model('./save_model/DQNTheta10v1.h5')
psi_agent = load_model('./save_model/DQNPsi10v1.h5')
saveWB(psi_agent)
