from numpy import *
import scipy.linalg
import sys

class FeedForwardNN:

    def __init__(self, input_size, hidden_layer_size, output_layer_size, epslilon):

        print("===== initializing =====")

        # Store sizes of input layer, hidden layer, and output layer
        self.I = input_size
        self.J = hidden_layer_size
        self.K = output_layer_size

        # store the learning paramenter epsilon
        self.epsilon = epslilon

        # randomly initialize all weight arrays
        self.W_ji = random.uniform(low = -0.5, high = 0.5, size=(self.J, self.I))
        self.W_kj = random.uniform(low = -0.5, high = 0.5, size=(self.K, self.J))
        print("Weight ji", self.W_ji)
        print("Weight kj", self.W_kj)

        # initialize biases, fixing bias values to 1, if needed can be changed to some other value later on
        self.B_ji = identity(self.J, dtype=float)
        self.Wb_ji = random.uniform(low=-0.5, high=0.5, size=(self.J, 1))
        self.B_kj = identity(self.K, dtype=float)
        self.Wb_kj = random.uniform(low=-0.5, high=0.5, size=(self.K, 1))

        print("Bias Weight ji", self.Wb_ji)
        print("Bias Weight kj", self.Wb_kj)
