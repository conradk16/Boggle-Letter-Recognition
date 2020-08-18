import pickle
import network2
import numpy as np

with open("training_data.p", "rb") as f:
    training_data = pickle.load(f)

with open("validation_data.p", "rb") as f:
    validation_data = pickle.load(f)

with open("test_data.p", "rb") as f:
    test_data = pickle.load(f)

net = network2.Network([900, 165, 21], cost_func="soft_max")
net.SGD(training_data, 1, 0.015, validation_data = test_data, regularization_param = 0.0065, epochs_to_decrease_eta=10)

with open("final_network.p", "wb") as f:
    pickle.dump(net, f)
