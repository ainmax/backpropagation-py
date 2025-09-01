from model.neural_network import NeuralNetwork
from train.test_set import TestSet
from train.trainer.trainer import Trainer
from train.trainer.trainer_options import TrainerOptions


network = NeuralNetwork(10, 11, [16])
trainer_options = TrainerOptions(
    learn_speed=0.6,
    inertia_coefficient=0.2,
    epochs_count=10000,
    batch_size=16,
    max_acceptable_average_loss=0.03
)
test_set = TestSet(network.input_size)
trainer = Trainer(test_set, trainer_options)

trainer.train_network(network)

network_losses = trainer.last_train_average_losses
print("--------------")
print(network_losses)
print("--------------")
for i in range(0, len(network.weights)):
    print(network.weights[i])
    print()
for i in range(0, len(network.biases)):
    print(network.biases[i])
    print()
print("--------------")
user_input = [0] * network.input_size
while True:
    try:
        input_str = input("Input:")
        network_input = [float(x) for x in input_str.split()]
    except RuntimeError:
        print()
        print("--------------")
        print("Enter 947 to terminate program")

        is_command_entered = False
        is_terminated = False

        while not is_command_entered:
            try:
                if input() == "947":
                    is_command_entered = True
                    is_terminated = True
                else:
                    is_command_entered = True
            except RuntimeError:
                break

        if is_terminated:
            break

        print("--------------")
        continue

    print()
    print("Output:")
    print(network.calc_output_by(user_input))
    print("--------------")





