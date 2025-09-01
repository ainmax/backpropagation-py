from model.matrix import Matrix
from model.neural_network import NeuralNetwork
from train.gradient.biases_gradient import BiasesGradient
from train.gradient.weights_gradient import WeightsGradient
from train.test import Test
from train.test_set import TestSet
from train.trainer.trainer_options import TrainerOptions


class Trainer:
    def __init__(self, test_set: TestSet, options: TrainerOptions):
        self.test_set = test_set
        self.options = options

        self.last_train_average_losses = []
        self.last_weights_increments: list[Matrix] = []
        self.last_biases_increments: list[Matrix] = []

    def train_network(self, network: NeuralNetwork):
        network.reset_parameters()
        is_trained_enough = False

        while not is_trained_enough:
            self.last_train_average_losses = [0] * self.options.epochs_count

            for i in range(0, self.options.epochs_count):
                # Tweak network's parameters
                self.train_epoch_of(network)

                # Calculate average loss
                average_loss = 0
                self.test_set.reset_iterating()
                while self.test_set.has_next():
                    current_test = self.test_set.next()
                    current_loss = network.calc_loss(current_test.input, current_test.correct_output)
                    average_loss += current_loss
                average_loss /= self.test_set.size
                self.last_train_average_losses[i] = average_loss

                print(average_loss)

                if average_loss <= self.options.max_acceptable_average_loss:
                    is_trained_enough = True
                    break

            # If network isn't trained enough, then training restarts
            if not is_trained_enough:
                network.reset_parameters()

    def train_epoch_of(self, network: NeuralNetwork):
        self.last_weights_increments = [
            Matrix(
                network.weights[i].n,
                network.weights[i].m
            ) for i in range(0, len(network.weights))
        ]
        self.last_biases_increments = [
            Matrix(
                network.biases[i].n,
                network.biases[i].m
            ) for i in range(0, len(network.biases))
        ]
        self.test_set.reset_iterating()

        while self.test_set.has_next():
            self.train_batch(network)

    def train_batch(self, network: NeuralNetwork):
        # Save last parameters
        last_weights = [
            network.weights[i].__copy__() for i in range(0, len(network.weights))
        ]
        last_biases = [
            network.biases[i].__copy__() for i in range(0, len(network.biases))
        ]

        # Generate batch
        batch_size = min(
            self.options.batch_size,
            self.test_set.size - self.test_set.get_current_index()
        )
        batch_tests = [
            self.test_set.next() for _ in range(0, batch_size)
        ]

        # Get sums of gradients
        gradients = self.__GradientCalculationProcess(batch_tests, 0, batch_size, network).run()
        # Weights average gradient
        for i in range(0, gradients[0].m):
            gradients[0].values[0][i] /= batch_size
        # Biases average gradient
        for i in range(0, gradients[1].m):
            gradients[1].values[0][i] /= batch_size

        # Weights and biases update
        self.update_network_parameters(
            network,
            gradients[0].values[0],
            gradients[1].values[0]
        )

        # Calculate new increments
        self.last_weights_increments = [
            network.weights[i] - last_weights[i] for i in range(0, len(network.weights))
        ]
        self.last_biases_increments = [
            network.biases[i] - last_biases[i] for i in range(0, len(network.biases))
        ]

    def update_network_parameters(self, network: NeuralNetwork, weights_gradient: list[float], biases_gradient: list[float]):
        # Weights shift
        current_weight_index = 0
        for i in range(0, len(network.weights)):
            for j in range(0, network.weights[i].n):
                for k in range(0, network.weights[i].m):
                    network.weights[i].values[j][k] -= (
                            self.options.learn_speed * weights_gradient[current_weight_index]
                    )
                    network.weights[i].values[j][k] += (
                            self.options.inertia_coefficient * self.last_weights_increments[i].values[j][k]
                    )
                    current_weight_index += 1

        # Biases shift
        current_bias_index = 0
        for i in range(0, len(network.biases)):
            for j in range(0, network.biases[i].n):
                network.biases[i].values[j][0] -= (
                        self.options.learn_speed * biases_gradient[current_bias_index]
                )
                network.biases[i].values[j][0] += (
                        self.options.inertia_coefficient * self.last_biases_increments[i].values[j][0]
                )
                current_bias_index += 1

    class __GradientCalculationProcess:
        """This class represents calculations in one process of multiprocessing"""

        def __init__(self, tests: list[Test], first_test_index: int, last_test_index: int, network: NeuralNetwork):
            self.tests: list[Test] = tests
            self.first_test_index = first_test_index
            self.last_test_index = last_test_index
            self.network: NeuralNetwork = network

        def run(self) -> tuple[Matrix, Matrix]:
            if self.first_test_index == self.last_test_index:
                return Matrix(1, 1), Matrix(1, 1)

            current_test_index = self.first_test_index
            gradients = (
                WeightsGradient(
                    _network=self.network,
                    _test=self.tests[current_test_index]
                ),
                BiasesGradient(
                    _network=self.network,
                    _test=self.tests[current_test_index]
                )
            )

            while current_test_index < self.last_test_index:
                gradients[0].add(
                    WeightsGradient(
                        _network=self.network,
                        _test=self.tests[current_test_index]
                    )
                )
                gradients[1].add(
                    BiasesGradient(
                        _network=self.network,
                        _test=self.tests[current_test_index]
                    )
                )
                current_test_index += 1

            return (
                gradients[0].loss_gradient,
                gradients[1].loss_gradient
            )
