## modules would include

## forward pass

## error calculation

## backward pass

## weight adjustment

## iterations

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetworkwithNumpy:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        ## initialize the weights and biases
        self.weight_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weight_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        ## hidden = X * weight_input_hidden + bias_hidden
        ## output = X * weight_hidden_output + bias_output
        self.hidden = sigmoid(np.dot(X, self.weight_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weight_hidden_output) + self.bias_output)
        return self.output

    def backward(self, X, y, output):
        ## --- propagating from output layer towards input ---

        ## calculating the error and error delta at the output layer
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)

        ## propagating back the error at the output layer to the hidden layer
        self.hidden_error = np.dot(self.output_delta, self.weight_hidden_output.T)
        self.hidden_delta = self.hidden_error * sigmoid_derivative(self.hidden)

        ## update the weights and the biases
        self.weight_hidden_output += np.dot(self.hidden.T, self.output_delta)
        self.bias_output += np.sum(self.output_delta, axis=0, keepdims=True)

        self.weight_input_hidden += np.dot(X.T, self.hidden_delta)
        self.bias_hidden += np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
