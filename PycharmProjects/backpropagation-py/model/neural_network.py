import copy

from model.matrix import Matrix
from model.random_matrix import RandomMatrix


class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int, hidden_layers_sizes, is_copy=False):
        self.hidden_layers_count = len(hidden_layers_sizes)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_sizes = copy.copy(hidden_layers_sizes)

        self.biases: list[Matrix] = []
        self.weights: list[Matrix] = []

        if not is_copy:
            self.__set_random_parameters()

    def __set_random_parameters(self):
        self.biases = [
            Matrix(self.hidden_layers_sizes[i], 1) for i in range(0, self.hidden_layers_count)
        ] + [Matrix(self.output_size,1)]

        self.weights = [RandomMatrix(self.hidden_layers_sizes[0], self.input_size)] + [
            RandomMatrix(self.hidden_layers_sizes[i + 1], self.hidden_layers_sizes[i]) for i in range(0, self.hidden_layers_count - 1)
        ] + [RandomMatrix(self.output_size, self.hidden_layers_sizes[-1])]

    def calc_output_by(self, input_values):
        previous_layer_output = Matrix(self.input_size, 1, input_values)

        for i in range(0, self.hidden_layers_count + 1):
            previous_layer_output = Matrix.sigmoid_of(
                self.weights[i] * previous_layer_output + self.biases[i]
            )

        return [previous_layer_output.values[i][0] for i in range(0, self.output_size)]

    def calc_loss(self, input_: list[float], expected_output: list[float]) -> float:
        """Calculates MSE"""

        if len(input_) != self.input_size or len(expected_output) != self.output_size:
            pass

        output = self.calc_output_by(input_)

        loss = 0
        for i in range(0, self.output_size):
            loss += (output[i] - expected_output[i])**2
        loss /= self.output_size

        return loss

    def reset_parameters(self):
        self.__set_random_parameters()

    def __copy__(self):
        network_copy = NeuralNetwork(self.input_size, self.output_size, self.hidden_layers_sizes, is_copy=True)
        network_copy.weights = self.weights
        network_copy.biases = self.biases
        return network_copy
