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

        self.loss = []

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

        self.D = (X - self.C_ji)
        # print(D)

        K = (self.D*self.D).sum(axis=1).reshape(len(self.D),1)
        # print(K)

        self.Phi = exp(-K/(2*self.var))
        # print(self.Phi)

        self.Y = self.W_kj @ self.Phi
        print("forward pass output",self.Y)

    def backprop(self, target):

        target = array(target).reshape(self.K,1)

        self.e_k = target - self.Y
        print("error vec", self.e_k)

        self.E = sum(self.e_k * self.e_k)
        print("mean sq error", self.E)

        self.storeloss(self.E)

        # calculating change in weight

        self.delta_W_kj = self.e_k @ self.Phi.transpose()
        print("delta W", self.delta_W_kj)

        # calculating change in center matrix

        # calculating temporary marix T and constant c = eta2/var
        T = self.W_kj.transpose() @ self.e_k
        c = self.eta2/self.var

        self.delta_C_ji = (c * (T * self.Phi))*self.D
        print("change in center",self.delta_C_ji)


    def expandInput(self, input_vector, j):
        return array([input_vector,]*j)

    def storeloss(self, e):
        """
        Function to track mean squared errors occurring during training of the neural network

        :param e: mean squared error
        :return: none
        """
        self.loss.append(e)

    def plot_convergence_characteristics(self):
        '''
        Function to plot the convergence characteristics viz. plot of error v/s pattern number
        Convergence characteristics shows the rate of convergence of the error of output of NN v/s actual value, to 0.
        :param errors: list containing mean square error corresponding to each pattern
        :return: none
        '''
        plt.plot(self.loss)
        plt.xlabel('Pattern number')
        plt.ylabel('MSE')
        plt.show()



a = Rbfnn(2,3,2,1,1,1)

a.forwardpass([1,2])
a.backprop([1.5, 2.5])