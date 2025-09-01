import math
from dataclasses import dataclass, field

from model.matrix import Matrix
from model.neural_network import NeuralNetwork
from train.test import Test


@dataclass(kw_only=True)
class NetworkGradient:
    """Gradient storage structure

            W(0)     W(1)     W(2)    ...     W(n)
    A(0)  G(0)(0)     -        -      ...      -
    A(1)  G(1)(0)  G(1)(1)     -      ...      -
    A(2)  G(2)(0)  G(2)(1)  G(1)(2)   ...      -
      .      .        .        .       .       .
      .      .        .        .       .       .
      .      .        .        .       .       .
    A(n)  G(n)(0)  G(n)(1)  G(n)(2)   ...   G(n)(n)

    Where n = hiddenLayersQuantity,
    A(i) - nodes activations vector,
    W(j) - weights vector (weights matrices transform to vectors by indexing element (k, r) as [k * M + r], where M - matrix v-dimension),
    G(i)(j) = Gij - gradient matrix:

                   W(j)[0]         W(j)[1]         W(j)[2]      ...      W(j)[N * M - 1]
    A(i)[0]       Gij[0][0]       Gij[0][1]       Gij[0][2]     ...     Gij[0][N * M - 1]
    A(i)[1]       Gij[1][0]       Gij[1][1]       Gij[1][2]     ...     Gij[1][N * M - 1]
    A(i)[2]       Gij[2][0]       Gij[2][1]       Gij[2][2]     ...     Gij[2][N * M - 1]
       .              .               .               .          .              .
       .              .               .               .          .              .
       .              .               .               .          .              .
    A(i)[S - 1]   Gij[S - 1][0]   Gij[S - 1][1]   Gij[S - 1][2]   ...   Gij[S - 1][N * M - 1]

    Where N - W(j) v-dimension,
    M - W(j) h-dimension,
    Gij[u][v] - partial derivative of A(i)[u] by W(j)[v]

    """

    _network: NeuralNetwork
    _test: Test

    loss_gradient: Matrix = field(init=False)

    def __post_init__(self):
        self.__layers_gradients = []
        test_input = Matrix(self._network.input_size, 1, self._test.input)
        self.__calc_next_layer_gradient(0, self._network.input_size, test_input)
        self.loss_gradient = self.__calc_loss_gradient(self._test)

    @staticmethod
    def __calc_activation_value_derivative(node_charge: float):
        return 1 / (2 + math.e ** (-node_charge) + math.e ** node_charge)

    def _calc_linear_derivative(self, linear_coefficient: float) -> float:
        pass

    def _get_connected_parameters_count(self, previous_layer_size: int, current_layer_size: int) -> int:
        pass

    def __calc_next_layer_gradient(self, current_layer_index: int, previous_layer_size: int, previous_layer_output: Matrix):
        """Recursively calculates the gradient layer by layer"""

        # Create aliases for class fields.
        weights, biases = self._network.weights, self._network.biases
        previous_layer_gradient = Matrix(1, 1) if current_layer_index == 0 else self.__layers_gradients[current_layer_index - 1]

        # Formula for current layer input is W * A + B, where
        # W - weights between current and previous layer,
        # A - previous layer output,
        # B - biases.
        current_layer_input = weights[current_layer_index] * previous_layer_output + biases[current_layer_index]
        # Activate current layer input.
        current_layer_output = Matrix.sigmoid_of(current_layer_input)
        current_layer_size = current_layer_input.n
        current_connected_nodes_count: int = self._get_connected_parameters_count(previous_layer_size, current_layer_size)

        # Define empty gradient matrix and create alias.
        self.__layers_gradients.append(
            Matrix(
                current_layer_size,
                current_connected_nodes_count + (0 if current_layer_index == 0 else previous_layer_gradient.m)
            )
        )
        current_layer_gradient = self.__layers_gradients[current_layer_index]

        # Calculate and assign part of the gradient with partial derivatives by parameters from current layer,
        # i.e. by weights and biases connected to current layer.
        for i in range(0, current_layer_size):
            for j in range(0, current_connected_nodes_count // current_layer_size):
                index_in_gradient = i * current_connected_nodes_count // current_layer_size + j + (0 if current_layer_index == 0 else previous_layer_gradient.m)
                # Linear derivative component
                current_layer_gradient.values[i][index_in_gradient] = self._calc_linear_derivative(previous_layer_output.values[j][0])
                # Activation derivative component
                current_layer_gradient.values[i][index_in_gradient] *= NetworkGradient.__calc_activation_value_derivative(current_layer_input.values[i][0])

        # If the current layer is the first one, then there is no other gradient part
        # and the recursion call happens here.
        if current_layer_index == 0:
            self.__calc_next_layer_gradient(current_layer_index + 1, current_layer_size, current_layer_output)
            return

        # Calculate part of the gradient with partial derivatives by parameters from previous layers,
        # i.e. by weights and biases NOT connected to current layer.
        linear_deep_gradient = weights[current_layer_index] * previous_layer_gradient
        for i in range(0, current_layer_size):
            for j in range(0, previous_layer_gradient.m):
                # Linear derivative component.
                current_layer_gradient.values[i][j] = linear_deep_gradient.values[i][j]
                # Activation derivative component.
                current_layer_gradient.values[i][j] *= NetworkGradient.__calc_activation_value_derivative(current_layer_input.values[i][0])

        # If this statement is true, then previous layer is output layer and output gradient is
        # calculated, so recursion must be stopped.
        if current_layer_index >= self._network.hidden_layers_count:
            return

        # Main recursion call.
        self.__calc_next_layer_gradient(current_layer_gradient + 1, current_layer_size, current_layer_output)


    def __calc_loss_gradient(self, test):
        """Calculates loss function gradient by network output layer gradient"""

        network_output = self._network.calc_output_by(test.input)
        loss_gradient_matrix = self.__layers_gradients[-1].__copy__()

        # Every partial derivative multiply by respective loss function derivative
        for i in range(0, loss_gradient_matrix.n):
            for j in range(0, loss_gradient_matrix.m):
                loss_gradient_matrix.values[i][j] *= 2 * (network_output[i] - test.correct_output[i]) / self._network.output_size

        # Partial derivatives of output layer are added up to partial derivatives of loss function
        loss_gradient = [0] * loss_gradient_matrix.m
        for i in range(0, loss_gradient_matrix.m):
            for j in range(0, loss_gradient_matrix.n):
                loss_gradient[i] += loss_gradient_matrix.values[j][i]

        return Matrix(1, loss_gradient_matrix.m, loss_gradient)

    def add(self, other):
        self.loss_gradient.add(other.loss_gradient)