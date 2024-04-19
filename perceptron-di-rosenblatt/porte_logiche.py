from perceprtron import Perceptron


def AND():
    # Dati per addestramento
    # tabella della verità della AND
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 0, 0, 1]

    neuron = Perceptron(X, Y)
    neuron.train()


def OR():
    # Dati per addestramento
    # tabella della verità della OR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 1]

    neuron = Perceptron(X, Y)
    neuron.train()
    print(neuron.output([0, 0]))


def NOT():
    # Dati per addestramento
    # tabella della verità della NOT
    X = [[0], [1]]
    Y = [1, 0]

    neuron = Perceptron(X, Y)
    neuron.train()
    print(neuron.output([0]))


def EXOR():
    # Dati per addestramento
    # tabella della verità della EXOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 0]

    neuron = Perceptron(X, Y)
    neuron.train(epochs=500)
    print(neuron.output([0, 0]))


# AND()
print("------------------------------------------------")
# OR()
print("------------------------------------------------")
# NOT()
print("------------------------------------------------")
EXOR()
