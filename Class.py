import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """A class for producing and fitting a basic neural network to data."""

    def __init__(self, inputs, outputs, num_hidden_layer_neurons, lr=0.05):
        """Method that initializes and trains the neural network.

        Arguments:
            inputs {np.array or array} -- Input features for training.
            outputs {[type]} -- Labels for input features.
            num_hidden_layer_neurons {int} -- Number of neurons in the hidden layer of the network

        Keyword Arguments:
            lr {float} -- Learning rate, or how fast the algorithm adjusts its weights and biases. Too high of a learning rate would results in the algorithm overshooting the point of convergence, and a learning rate that is too low will cause the algorithm to not converge. (default: {0.05})
        """

        self.hidden_layer_neurons = num_hidden_layer_neurons

        self.train_inputs = np.array(inputs)
        self.train_outputs = np.array(outputs)

        self.weights_xh = np.random.rand(
            inputs.shape[1], self.hidden_layer_neurons)
        self.biases_hl = np.random.rand(self.hidden_layer_neurons)
        self.weights_hy = np.random.rand(self.hidden_layer_neurons)

        self.biases_out = np.random.rand(self.train_outputs.shape[1])
        self.backprop(3000, lr)

    def __str__(self):
        """Return str(self)."""

        return(f"Weights to hidden layer: {self.weights_xh} \nBiases to hidden layer: {self.biases_hl} \nWeights to output layer: {self.weights_hy} \nBiases to output layer {self.biases_out}")

    def relu(self, x, output_deriv=False):
        """Function that models the ReLu (Rectified Linear Unit) activation function.

        Arguments:
            x {np.array or float} -- Value or array of values that will be passed into the ReLu activation function.

        Keyword Arguments:
            output_deriv {bool} -- Argument that determines whether the function will output the derivative of the ReLu function, or its actual value. (default: {False})

        Returns:
            np.array or float -- Type of output is the same as input, which is the ReLu function or its derivative (based on output_deriv value).
        """
        if output_deriv:
            return np.array(x) > 0
        else:
            return list(map(lambda x: max(0, x), x)) if len(x) > 1 else max(0, x)

    def sigmoid(self, x, output_deriv=False):
        """Function that models the sigmoid activation function, which compresses input values into probabilities between 0 and 1.

        Arguments:
            x {np.array or float} -- Value or array of values that will be passed into the sigmoid function.

        Keyword Arguments:
            output_deriv {bool} -- Argument that determines whether the function will output the derivative of the sigmoid function, or its actual value. (default: {False})

        Returns:
            np.array or float -- Type of output is the same as input, which is the sigmoid or its derivative (based on output_deriv value).
        """
        outputs = 1 / (1 + np.exp(-x))
        if output_deriv:
            return outputs * (1-outputs)
        else:
            return outputs

    def feedforward(self, inputs, outputs_only=True):
        """Given input data, predict output value based on biases and weights.

        Arguments:
            inputs {np.array} -- Input features values into the neural network.

        Keyword Arguments:
            outputs_only {bool} -- Determines whether the function only returns outputs, or returns all activation and weighted sum values. (default: {True})

        Returns:
            np.array -- See explanation in "outputs_only" descriptions.
        """

        inputs = np.array(inputs)
        weightedsum1 = np.dot(self.weights_xh.T, inputs) + self.biases_hl
        activations_hl = self.relu(weightedsum1)
        weightedsum2 = np.dot(
            self.weights_hy.T, activations_hl) + self.biases_out
        output = self.sigmoid(weightedsum2)
        if outputs_only:
            return float(output)
        else:
            return activations_hl, float(output), weightedsum1, weightedsum2

    def backprop(self, iters, lr, cost_prog=False):
        """Backpropogation algorithm for updating values of the network weights and biases (based on partial derivatives of cost function).

        Arguments:
            iters {int} -- Number of times the backpropogation algorithm will repeat (how many iterations the model will be trained with).
            lr {float} -- Learning rate, or how fast the algorithm adjusts its weights and biases. Too high of a learning rate would results in the algorithm overshooting the point of convergence, and a learning rate that is too low will cause the algorithm to not converge.
            cost_prog {bool} -- Determines whether cost history graph will be displayed.
        """

        cost_history = []

        for _ in range(iters):

            dweights1 = 0
            dbiases1 = 0
            dweights2 = 0
            dbiases2 = 0

            sum_squared_error = 0
            for inputs, output in zip(self.train_inputs, self.train_outputs):

                hl_activations, prediction, _, weightedsum2 = self.feedforward(
                    inputs, outputs_only=False)

                error = prediction - output
                sum_squared_error += error ** 2

                sig_deriv_hl = self.sigmoid(output, output_deriv=True)
                relu_deriv_inputs = self.relu(
                    hl_activations, output_deriv=True)

                # Updating weights and bias going to output layer
                dweights2 += 2 * error * weightedsum2 * sig_deriv_hl
                dbiases2 += 2 * error * sig_deriv_hl

                # Updating weights and bias going to hidden layer
                dweights1 += 2 * error * sig_deriv_hl * \
                    np.matmul(
                        inputs[:, np.newaxis], (self.weights_hy * relu_deriv_inputs)[:, np.newaxis].T)
                dbiases1 += 2 * error * relu_deriv_inputs * sig_deriv_hl * self.weights_hy

            cost_history.append(sum_squared_error)

            self.weights_xh -= dweights1 * lr / len(self.train_inputs)
            self.weights_hy -= dweights2 * lr / len(self.train_inputs)
            self.biases_hl -= dbiases1 * lr / len(self.train_inputs)
            self.biases_out -= dbiases2 * lr / len(self.train_inputs)

        # Code to see if backprop is working
        if cost_prog:
            plt.plot(cost_history)
            plt.xlabel("Iteration")
            plt.ylabel("Sum Squa2red Error (SSE)")
            plt.title("Cost Progression over Gradient Descent")
            plt.style.use("fivethirtyeight")
            plt.show()
