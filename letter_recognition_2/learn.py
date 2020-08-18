import pickle
import network2
import numpy as np

print("loading data")

with open("training_data.p", "rb") as f:
    training_data = pickle.load(f)

with open("validation_data.p", "rb") as f:
    validation_data = pickle.load(f)

with open("test_data.p", "rb") as f:
    test_data = pickle.load(f)

print("done loading data")

net = network2.Network([900, 165, 84], cost_func="soft_max")
net.SGD(training_data, 30, 0.015, validation_data = validation_data, regularization_param = 0.0065)
