import network
import network2
import network2_encoder

import warnings
warnings.filterwarnings("ignore")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# net = network.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = network2_encoder.Network([784, 256, 64, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 50, 10, 0.5, lmbda=5.0, evaluation_data=test_data,
    monitor_evaluation_accuracy=True)