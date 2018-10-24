from numpy import *
import matplotlib.pyplot as plt
import scipy.linalg
import sys

class Rbfnn:

    def __init__(self, input_size, hidden_layer_size, output_size, eta1, eta2, variance):

        # storing network parameters as global variables
        self.I = input_size
        self.J = hidden_layer_size
        self.K = output_size
        self.eta1 = eta1
        self.eta2 = eta2
        self.var = variance

        # initializing the weight matrix
        self.W_kj = random.uniform(low=-0.5, high=0.5, size=(self.K, self.J))

        # initialize center matrix for the hidden layer
        self.C_ji = random.uniform(low=-0.5, high=0.5, size=(self.J, self.I))
        # self.C_ji = array([[0,2],[0,1],[0,0]] )

    def train(self, training_vector, target_vector):
        print("sfsdfdsf")

    def forwardpass(self, input_vector):

        # Steps:
        # Step 1 : check length of input vector, and make it into a 1xj row matrix
        # Step 2 : calculate the Phi matrix by the following steps:
        #   Step 2.1 : expand the input row matrix by copying the row j times, making X, a jxi matrix
        #   Step 2.2 : subtract C from the X matrix, and square each element to generate matrix X' = (X - C).^2
        #   Step 2.3 : sum each row of the matrix to generate column matrix K
        #   Step 2.4 : generate matrix Phi, such that Phi_i = exp(-K_i/(2*variance))
        # Step 3: calculate output matrix Y = W_kj x Phi

        if len(input_vector) != self.I:
            raise Exception("Input vector size mismatch")

        input_vector = array(input_vector).reshape(1,self.I)
        input_vector = input_vector[0]

        X = self.expandInput(input_vector, self.J)
        # print(X)

        D = (X - self.C_ji)
        # print(D)

        K = (D*D).sum(axis=1).reshape(len(D),1)
        # print(K)

        self.Phi = exp(-K/(2*self.var))
        # print(self.Phi)

        self.Y = self.W_kj @ self.Phi
        print("forward pass output",self.Y)



    def expandInput(self, input_vector, j):
        return array([input_vector,]*j)


a = Rbfnn(2,3,1,1,1,1)

a.forwardpass([1,2])
# a.forwardpass([[1],[2]])