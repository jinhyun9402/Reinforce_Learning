import numpy as np
def saveWB(agent):

    text = open('./save_model/get_weight_v3.csv', 'w')
    w0 = agent.layers[1].get_weights()
    for ee in range(w0[0].shape[0]):
        for e in range(w0[0].shape[1]):
            W0 = np.array2string(w0[0][ee][e])
            text.write(W0 + ",")
        text.write("\n")

    # text.write("\n")
    w1 = agent.layers[2].get_weights()
    for ee in range(w1[0].shape[0]):
        for e in range(w1[0].shape[1]):
            W1 = np.array2string(w1[0][ee][e])
            text.write(W1 + ",")
        text.write("\n")

    # text.write("\n")
    w2 = agent.layers[3].get_weights()
    for ee in range(w2[0].shape[0]):
        for e in range(w2[0].shape[1]):
            W2 = np.array2string(w2[0][ee][e])
            text.write(W2 + ",")
        text.write("\n")

    # text.write("\n")
    w3 = agent.layers[4].get_weights()
    for ee in range(w3[0].shape[0]):
        for e in range(w3[0].shape[1]):
            W3 = np.array2string(w3[0][ee][e])
            text.write(W3 + ",")
        text.write("\n")

    # text.write("\n")
    w4 = agent.layers[5].get_weights()
    for ee in range(w4[0].shape[0]):
        for e in range(w4[0].shape[1]):
            W4 = np.array2string(w4[0][ee][e])
            text.write(W4 + ",")
        text.write("\n")

    text.close()

    text = open('./save_model/get_bias_v3.csv', 'w')
    for e in range(w0[1].shape[0]):
        W0 = np.array2string(w0[1][e])
        text.write(W0 + ",")
    text.write("\n")

    for e in range(w1[1].shape[0]):
        W1 = np.array2string(w1[1][e])
        text.write(W1 + ",")
    text.write("\n")

    for e in range(w2[1].shape[0]):
        W2 = np.array2string(w2[1][e])
        text.write(W2 + ",")
    text.write("\n")

    for e in range(w3[1].shape[0]):
        W3 = np.array2string(w3[1][e])
        text.write(W3 + ",")
    text.write("\n")

    for e in range(w4[1].shape[0]):
        W4 = np.array2string(w4[1][e])
        text.write(W4 + ",")
    text.write("\n")

    text.close()

    return agent

def saveWB_thetapsi(agent):

    text = open('./save_model/get_weight_thetapsi.csv', 'w')
    w0 = agent.layers[1].get_weights()
    for ee in range(6):
        for e in range(42):
            W0 = np.array2string(w0[0][ee][e])
            text.write(W0 + ",")
        text.write("\n")

    # text.write("\n")
    w1 = agent.layers[2].get_weights()
    for ee in range(42):
        for e in range(42):
            W1 = np.array2string(w1[0][ee][e])
            text.write(W1 + ",")
        text.write("\n")

    # text.write("\n")
    w2 = agent.layers[3].get_weights()
    for ee in range(42):
        for e in range(42):
            W2 = np.array2string(w2[0][ee][e])
            text.write(W2 + ",")
        text.write("\n")

    # text.write("\n")
    w3 = agent.layers[4].get_weights()
    for ee in range(42):
        for e in range(42):
            W3 = np.array2string(w3[0][ee][e])
            text.write(W3 + ",")
        text.write("\n")

    # text.write("\n")
    w4 = agent.layers[5].get_weights()
    for ee in range(42):
        for e in range(3):
            W4 = np.array2string(w4[0][ee][e])
            text.write(W4 + ",")
        text.write("\n")

    # text.write("\n")
    w5 = agent.layers[6].get_weights()
    for ee in range(42):
        for e in range(3):
            W5 = np.array2string(w5[0][ee][e])
            text.write(W5 + ",")
        text.write("\n")

    text.close()

    text = open('./save_model/get_bias_thetapsi.csv', 'w')
    for e in range(42):
        W0 = np.array2string(w0[1][e])
        text.write(W0 + ",")
    text.write("\n")

    for e in range(42):
        W1 = np.array2string(w1[1][e])
        text.write(W1 + ",")
    text.write("\n")

    for e in range(42):
        W2 = np.array2string(w2[1][e])
        text.write(W2 + ",")
    text.write("\n")

    for e in range(42):
        W3 = np.array2string(w3[1][e])
        text.write(W3 + ",")
    text.write("\n")

    for e in range(3):
        W4 = np.array2string(w4[1][e])
        text.write(W4 + ",")
    text.write("\n")

    for e in range(3):
        W5 = np.array2string(w5[1][e])
        text.write(W5 + ",")
    text.write("\n")

    text.close()

    return agent
