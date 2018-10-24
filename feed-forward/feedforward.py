from numpy import *
import matplotlib.pyplot as plt
import scipy.linalg
import sys

class FeedForwardNN:

    def __init__(self, input_size, hidden_layer_size, output_layer_size, epslilon):
        """
        :param input_size: number of nodes in the input layer
        :param hidden_layer_size: number of neurons in the hidden layer
        :param output_layer_size: number of neurons in the output layer
        :param epslilon: learning parameter or learning rate
        """

        # Store sizes of input layer, hidden layer, and output layer
        self.I = input_size
        self.J = hidden_layer_size
        self.K = output_layer_size

        # store the learning parameter/ learning rate: epsilon
        self.epsilon = epslilon

        # randomly initialize all weight arrays
        self.W_ji = random.uniform(low = -0.5, high = 0.5, size=(self.J, self.I))   # matrix of dimension JxI
        self.W_kj = random.uniform(low = -0.5, high = 0.5, size=(self.K, self.J))   # matrix of dimension KxJ

        # initialize biases, fixing bias values to 1, if needed can be changed to some other value later on
        self.B_ji = identity(self.J, dtype=float)   # diagonal matrix of dimension JxI, diag(j,i) = bias of jth neuron
        self.Wb_ji = random.uniform(low=-0.5, high=0.5, size=(self.J, 1))   # weights corresponding to biases in B_ji
        self.B_kj = identity(self.K, dtype=float)   # diagonal matrix of dimension KxJ, diag(k,j) = bias of kth neuron
        self.Wb_kj = random.uniform(low=-0.5, high=0.5, size=(self.K, 1))   # weights corresponding to biases in B_kj

        # initialize an empty list for storing outputs of loss/cost function
        self.loss = []

    def feedforward(self, input_data):
        """
        Function to perform forward pass in the neural network

        :param input_data: list containing input to the neural network
        :return: list containing output of the neural network
        """

        # check if input vector matches input size pre-specified during initialization
        if len(input_data) != self.I:
            raise Exception("input size exceed the predefined input size")

        # prepare the input vector for further operations by storing it in an array a_i
        self.a_i = array(input_data).reshape(len(input_data),1)   # colum-matrix of dimension Ix1

        # calculate the net input - net_j - for each neuron in the hidden layer J
        self.net_j = (self.W_ji @ self.a_i) + (self.B_ji @ self.Wb_ji)  # colum-matrix of dimension Jx1

        # pass the net input to each neuron through the activation function, and produce a_j - output of hidden layer
        # self.a_j = 1/(1+exp(-self.net_j))
        self.a_j = self.activation(self.net_j)  # colum-matrix of dimension Jx1

        # the output a_j of hidden layer becomes input to output layer
        # calculate the net input - net_k - for each neuron in the output layer
        self.net_k = (self.W_kj @ self.a_j) + (self.B_kj @ self.Wb_kj)  # colum-matrix of dimension Kx1

        # now calculate a_k using activation function one sided sigmoid, you can change function here
        # self.a_k = 1/(1+exp(-self.net_k))
        self.a_k = self.activation(self.net_k)  # colum-matrix of dimension Kx1

        # return the output of the neural network - a_k
        # return self.a_k
        print("a_k", self.a_k)

    def backprop(self, target):
        """
        Implementation of the backpropagation algorithm for feed-forward neural network
        :param target: list containing expected outputs from the neural network
        :return:
        """

        # calculate error e_k corresponding to the output a_k
        self.e_k = self.errorcalc(target, self.a_k)  # colum-matrix of dimension Kx1

        # calculate gradient descent loss function or cost function and store it
        self.E = (1/2)*sum(self.e_k**2)
        self.storeloss(self.E)
        # print("loss function matix", self.E)

        # based on errors, calculate the change in weights

        # for weights corresponding to output layer, calculate delta_wkj and delta_wbkj

        # first calculate delta_k = e_k*d(a_k), where d(a_k) is differentiation, and all operations are elementwise
        self.delta_k = self.e_k * self.d(self.a_k)

        # now calculate delta_wkj = epsilon * delta_k * transpose(a_j)
        self.delta_W_kj = self.epsilon * (self.delta_k @ self.a_j.transpose())
        # now calculate delta_wbkj = epsilon * delta_k * b
        self.delta_Wb_kj = self.epsilon * (self.B_kj @ self.delta_k)

        # for weights corresponding to hidden layer, calculate delta_wji and delta_wbji now

        # first calculate temporary intermediate T matrix = transpose(w_kj)*delta_k
        self.T_mat = self.W_kj.transpose() @ self.delta_k

        # now calculate delta_j = T_mat*d(a_j) , where d(a_j) is differentiation
        self.delta_j = self.T_mat * self.d(self.a_j)

        # now calculate delta_wji = epsilon * delta_j * transpose(a_i)
        self.delta_W_ji = self.epsilon * (self.delta_j @ self.a_i.transpose())
        # now calculate delta_wbji = epsilon * delta_j * b
        self.delta_Wb_ji = self.epsilon * (self.B_ji @ self.delta_j)

    def fit(self, training_data, target_output):
        """
        Function for fitting the input data into the model

        This function will take in a dataset, i.e. input data and corresponding output dataset, i.e. target_output,
        and train the model.

        :param training_data: list of lists containing training patterns
        :param target_output: list of lists containing target patterns
        :return: none
        """
        if len(training_data) != len(target_output):
            raise Exception("Training data and Target output lengths mismatch")

        for i in range(len(training_data)):
            self.train(training_data[i], target_output[i])

        print("============================================")
        print("===== Updated weights ======================")
        print("============================================")
        print("Input-Hidden layer weights W_ji", self.W_ji)
        print("Hidden-Output layer weights W_kj", self.W_kj)
        print("Hidden layer bias weights W_ji", self.Wb_ji)
        print("Output layer bias weights W_kj", self.Wb_kj)


    def train(self, training_vector, target_vector):
        """
        Function to feed-forward the neural network, calculate offset of weights by backpropagating and update weights
        accordingly

        :param training_vector: list containing input to the neural network
        :param target_vector: list containing expected output from the neural network
        :return: none
        """
        self.feedforward(training_vector)
        self.backprop(target_vector)
        self.updateWeights()

    def test(self, testing_data):
        print("====== testing ======")
        self.feedforward(testing_data)

    def updateWeights(self):
        """
        Function to update the weights after error backpropagation
        :return: none
        """
        self.W_kj = self.W_kj + self.delta_W_kj
        self.W_ji = self.W_ji + self.delta_W_ji
        self.Wb_kj = self.Wb_kj + self.delta_Wb_kj
        self.Wb_ji = self.Wb_ji + self.delta_Wb_ji


    def errorcalc(self, target, actual):
        """
        Function to calculate the offset/error by subtracting the output of output neuron to the expected/target value

        :param target: matrix containing values that were expected to be output of the neural network
        :param actual: matrix containing actual output of the neural network
        :return: matrix containing difference of target and actual matrices
        """
        t = array(target).reshape(shape(actual))
        return t - actual

    def activation(self, x):
        """
        Implementation of activation function

        This function takes in net input to a neuron, and using the defined activation function calculates output of
        the neuron.

        I have used sigmoid function for activation in this implementation.

        :param x: colum-matrix whose each entry is an input to a neuron of some layer(hidden/output)
        :return: colum-matrix containing output of each neuron, after passing from sigmoid function
        """
        return (1/(1+exp(-x)))

    def d(self, x):
        """
        Implementation of the differentiation of the used activation function

        Since I have used the sigmoid function: s = 1/(1+exp(-x)) , the derivation of this function is simple: s*(1-s)

        :param x: colum-matrix containing output of some layer(hidden/output) whose differentiation is to be calculated
        :return: colum-matrix containing differentiation of input matrix x
        """
        return x*(1-x)

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
