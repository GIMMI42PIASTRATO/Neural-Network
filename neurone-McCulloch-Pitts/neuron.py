import math


class NeuronMcCullochPitts(object):
    weights = [0.5, 0.5]

    def sigmoid_activation_fun(self, weightedSum):
        # sigmoid function
        return 1 / (1 + math.exp(-weightedSum))

    def step_activation_fun(self, weightedSum):
        if weightedSum >= 0:
            return 1
        else:
            return 0

    def output(self, inputs, activation_fun="sigmoid"):
        weightedSum = 0
        for i in range(len(inputs)):
            weightedSum += inputs[i] * self.weights[i]

        if activation_fun == "sigmoid":
            return self.sigmoid_activation_fun(weightedSum)
        elif activation_fun == "step":
            return self.step_activation_fun(weightedSum)


# Prima prova
neuron1 = NeuronMcCullochPitts()
print(neuron1.output([0, 0]))
print(neuron1.output([0, 1]))
print(neuron1.output([1, 1]))
print("--------------------")

# Input diversi da 0 e 1
print(neuron1.output([0.5, 0.5]))
print(neuron1.output([-0.2, -0.8]))
print(neuron1.output([0.9, -0.1]))
print(neuron1.output([10, 20]))
print(neuron1.output([-5, -10]))
print("--------------------")

# Cambio funzione di attivazione
print(neuron1.output([0, 0], "step"))
print(neuron1.output([-1, -1], "step"))
print(neuron1.output([0, 1], "step"))
print("--------------------")

# Output di due neuroni input del terzo
neuron2 = NeuronMcCullochPitts()
neuron3 = NeuronMcCullochPitts()

out1 = neuron1.output([0, 0])
out2 = neuron2.output([0, 1])
print(neuron3.output([out1, out2]))

out1 = neuron1.output([1, 0])
out2 = neuron2.output([0, 1])
print(neuron3.output([out1, out2]))

out1 = neuron1.output([-1, 0.3])
out2 = neuron2.output([-4, 1.5])
print(neuron3.output([out1, out2]))
