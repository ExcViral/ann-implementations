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


    def feedforward(self, input):
        print("===== feeding-forward =====")
        # store input into array named a_i
        if(len(input) != self.I):
            raise Exception("input size exceed the predefined input size")

        self.a_i = array(input).reshape(len(input),1)
        print("input", self.a_i)

        # first calculate net_j
        self.net_j = (self.W_ji @ self.a_i) + (self.B_ji @ self.Wb_ji)
        print("net_j", self.net_j)

        # now calculate a_j using activation function one sided sigmoid, you can change function here
        self.a_j = 1/(1+exp(-self.net_j))
        print("a_j", self.a_j)

        # first calculate net_k
        self.net_k = (self.W_kj @ self.a_j) + (self.B_kj @ self.Wb_kj)
        print("net_k", self.net_k)

        # now calculate a_k using activation function one sided sigmoid, you can change function here
        self.a_k = 1/(1+exp(-self.net_k))
        print("a_k", self.a_k)

    def backprop(self, target):
        print("===== back-propagation =====")

        # calculate e_k's
        self.e_k = self.errorcalc(target)
        print("ek's", self.e_k)

        # calculate gradient descent loss function
        self.E = (1/2)*sum(self.e_k**2)
        print("loss function matix", self.E)

        # calculate delta_wkj and delta_wbkj now

        # first calculate delta_k = e_k*a_k*(1-a_k), all operations are elementwise
        self.delta_k = self.e_k * self.a_k * (1 - self.a_k)
        print("delta_k", self.delta_k)

        # now calculate delta_wkj = epsilon * delta_k * transpose(a_j)
        self.delta_W_kj = self.epsilon * (self.delta_k @ self.a_j.transpose())
        print("delta_W_kj", self.delta_W_kj)
        self.delta_Wb_kj = self.epsilon * (self.B_kj @ self.delta_k)
        print("delta_Wb_kj", self.delta_Wb_kj)

        # calculate delta_wji and delta_wbji now

        # first calculate temporary intermediate T matrix = transpose(w_kj)*delta_k
        self.T_mat = self.W_kj.transpose() @ self.delta_k
        print("T matrix, temporary intermediate", self.T_mat)

        # now calculate delta_j = T_mat*a_j*(1-a_j)
        self.delta_j = self.T_mat * self.a_j * (1 - self.a_j)
        print("delta_j", self.delta_j)

        # now calculate delta_wji = epsilon * delta_j * transpose(a_i)
        self.delta_W_ji = self.epsilon * (self.delta_j @ self.a_i.transpose())
        print("delta_W_ji", self.delta_W_ji)
        self.delta_Wb_ji = self.epsilon * (self.B_ji @ self.delta_j)
        print("delta_Wb_ji", self.delta_Wb_ji)

    def errorcalc(self, target):
        t = array(target).reshape(shape(self.a_k))
        return t - self.a_k

    def train(self, training_data, target_vector):
        print("===== training =====")
        self.feedforward(training_data)
        self.backprop(target_vector)

    def test(self, testing_data):
        print("testing")


testObj = FeedForwardNN(2,3,2,1)

testObj.train([1,2],[2,3])