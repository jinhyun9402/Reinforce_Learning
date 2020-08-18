import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras.models import load_model

Weight = pd.read_csv('./save_model/tp_weight.csv', header = None)
Bias = pd.read_csv('./save_model/tp_bias.csv', header = None)
Agent = load_model('./save_model/DQN_ThetaPsi_Controller_NoWB.h5')

Agent.layers[1].weights[0].assign(Weight[0:4], None)
Agent.layers[1].bias = Bias[0:1]

Agent.layers[2].weights[0].assign(Weight[4:44], None)
Agent.layers[2].bias = Bias[1:2]

Agent.layers[3].weights[0].assign(Weight[44:84], None)
Agent.layers[3].bias = Bias[2:3]

Agent.layers[4].weights[0].assign(Weight[84:124], None)
Agent.layers[4].bias = Bias[3:4]

Agent.layers[5].weights[0].assign(tf.Variable(Weight)[124:164, 0:6], None)
Agent.layers[5].bias = Bias[4:5]

Agent.save("./save_model/DQN_ThetaPsi_Controller_NewWB.h5")
print("done")