import random
import math


class Perceptron(object):

    def __init__(self, X, Y, learningRate=0.01):
        self.weights = [random.random() for _ in range(len(X[0]) + 1)]
        self.X = X
        self.Y = Y
        self.learningRate = learningRate
        print("Pesi iniziali: ", self.weights)

    def activationFunction(self, weightedSum):
        # funzione gradino
        if weightedSum > 0:
            ret = 1.0
        else:
            ret = 0

        return ret

    def output(self, inputs):
        weightedSum = self.weights[0]

        for i in range(len(inputs)):
            weightedSum += inputs[i] * self.weights[i + 1]

        return self.activationFunction(weightedSum)

    def learn(self, x, yAtteso):
        yCalcolato = self.output(x)
        error = yAtteso - yCalcolato

        print("input", x, "yAtteso", yAtteso, "error", error)

        # aggiorno peso del bias
        self.weights[0] += self.learningRate * error
        # aggiorno i pesi degli input del perceptron
        for i in range(len(self.weights) - 1):
            self.weights[i + 1] += self.learningRate * error * x[i]

    def train(self, epochs=100):
        for epoch in range(epochs):
            print("epoch:", epoch)

            for i, x in enumerate(self.X):
                yAtteso = self.Y[i]
                self.learn(x, yAtteso)
                # print("Weight: ", self.weights)

        # verifica dell'addestramento
        print("Risultati dell'addestramento")
        for i, x in enumerate(self.X):
            print("input:", x, "output:", self.output(x))
        print("Weight: ", self.weights)
