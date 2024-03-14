import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Varying Learning Rate
# print("Testing different learning rates:")
# for learning_rate in [0.1, 1.0, 3.0, 5.0]:
#     net = network.Network([784, 30, 10])
#     print(f"Learning rate: {learning_rate}")
#     net.SGD(training_data, 30, 10, learning_rate, test_data=test_data)

# Varying Batch Size
# print("\nTesting different batch sizes:")
# for batch_size in [1, 10, 20, 50]:
#     net = network.Network([784, 30, 10])
#     print(f"Batch size: {batch_size}")
#     net.SGD(training_data, 30, batch_size, 3.0, test_data=test_data)

# Different Activation Functions
print("\nTesting different activation functions:")
for activation_function in ['sigmoid', 'tanh']:
    net = network.Network([784, 30, 10], activation_function=activation_function)
    print(f"Activation function: {activation_function}")
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Different Amount of Neurons in the Hidden Layer
# print("\nTesting different numbers of neurons in the hidden layer:")
# for hidden_neurons in [10, 30, 50, 100]:
#     net = network.Network([784, hidden_neurons, 10])
#     print(f"Hidden layer neurons: {hidden_neurons}")
#     net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
