import numpy as np
import json
import sys
import time

class NeuralNetwork:
    def __init__(self, architecture, learning_rate=0.1):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.weights = []
        self.cancel_training = False  # Flag utilisé pour l'annulation de l'entraînement
        self.setup_network()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def load_training_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        training_data = []
        classes = sorted(data.keys(), key=lambda x: int(x))
        num_classes = len(classes)
        for label in classes:
            examples = data[label]
            for example in examples:
                flattened = [pixel for row in example for pixel in row]
                flattened = np.array(flattened, dtype=np.float32)
                target = np.zeros(num_classes, dtype=np.float32)
                target[classes.index(label)] = 1.0
                training_data.append((flattened, target))
        input_dim = training_data[0][0].shape[0]
        return training_data, input_dim, num_classes

    def update_progress(self, progress, epoch, epochs, eta):
        bar_length = 100
        filled_length = int(bar_length * progress)
        bar = "[" + "#" * filled_length + "-" * (bar_length - filled_length) + "]"
        sys.stdout.write("\r" + bar + f" {epoch+1}/{epochs} - ETA: {eta:.1f} sec")
        sys.stdout.flush()

    def setup_network(self):
        self.weights = []
        for l in range(1, len(self.architecture)):
            W = np.random.uniform(0, 1, (self.architecture[l-1], self.architecture[l])).astype(np.float32)
            self.weights.append(W)

    def train_network(self, training_data, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            if self.cancel_training:
                break
            for inputs_sample, expected in training_data:
                activations = [inputs_sample]
                zs = []
                a = inputs_sample
                for W in self.weights:
                    z = np.dot(a, W)
                    zs.append(z)
                    a = self.sigmoid(z)
                    activations.append(a)
                delta = (activations[-1] - expected) * (activations[-1] * (1 - activations[-1]))
                deltas = [None] * len(self.weights)
                deltas[-1] = delta
                for l in range(len(self.weights) - 2, -1, -1):
                    delta = np.dot(self.weights[l+1], deltas[l+1]) * (activations[l+1] * (1 - activations[l+1]))
                    deltas[l] = delta
                for l in range(len(self.weights)):
                    grad = np.outer(activations[l], deltas[l])
                    self.weights[l] -= self.learning_rate * grad
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            remaining = avg_epoch_time * (epochs - (epoch + 1))
            self.update_progress((epoch + 1) / epochs, epoch, epochs, remaining)

    def compute_output(self, inputs):
        a = inputs
        for W in self.weights:
            a = self.sigmoid(np.dot(a, W))
        return a

    def save_weights(self, filename):
        weights_list = [W.tolist() for W in self.weights]
        with open(filename, 'w') as f:
            json.dump(weights_list, f)
    
    def load_weights(self, filename):
        with open(filename, 'r') as f:
            weights_list = json.load(f)
        self.weights = [np.array(W, dtype=np.float32) for W in weights_list]

if __name__ == "__main__":
    nn = NeuralNetwork(architecture=[64, 16, 10], learning_rate=0.1)

    training_data, input_dim, num_classes = nn.load_training_data("data/training_data.json")
    print(f"{len(training_data)} exemples chargés, dimension d'entrée: {input_dim}, nombre de classes: {num_classes}")
    print("")

    sample_input, sample_expected = training_data[0]

    print("Sans entrainement :")
    output = nn.compute_output(sample_input)
    print("Input fourni :", sample_input.tolist())
    print("Sortie attendue:", sample_expected.tolist())
    print("Sortie obtenue :", output.tolist())
    print("")

    epochs = 10000
    nn.train_network(training_data, epochs)

    print("")
    print("Avec entrainement :")
    output = nn.compute_output(sample_input)
    rounded_output = [round(x, 4) for x in output.tolist()]
    print("Input fourni :", sample_input.tolist())
    print("Sortie attendue:", sample_expected.tolist())
    print("Sortie obtenue :", rounded_output)

    nn.save_weights(f"weights/{epochs}_weights.json")
