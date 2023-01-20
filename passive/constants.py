from datetime import datetime
time_str = datetime.now().strftime("%Y_%m_%d_%H")
ITER = 2000  # number of iterations of gradient descent default: 2000
PARTICIPANT_V = 2 # Number of participants vertically
PARTICIPANT_H = 4 # Number of participants horizontally
HIDDEN_LAYER = 256  # number of neurons in the hidden layer
CONF_INIT_LAMBDA = 0.01
SAMPLING_PROB = 0.1
MAX_NORM = 1
SIGMA = 0.001
BATCHES = 10
RECORDING_PATH = "./output/"