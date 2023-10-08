import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

class BoltzmannMachine:
    def __init__(self, nv, nh):
        self.nv = nv
        self.nh = nh
        
        # Initialize weight matrix with random values
        self.weights = np.random.randn(nv, nh)
        
        # Initialize visible and hidden biases
        self.visible_bias = np.random.randn(nv)
        self.hidden_bias = np.random.randn(nh)

    def sample_hidden_units(self, visible_units):
        activation = np.dot(visible_units, self.weights) + self.hidden_bias
        prob_hidden_units = self.sigmoid(activation)
        hidden_units = np.random.binomial(1, prob_hidden_units)
        return hidden_units

    def sample_visible_units(self, hidden_units):
        activation = np.dot(hidden_units, self.weights.T) + self.visible_bias
        prob_visible_units = self.sigmoid(activation)
        visible_units = np.random.binomial(1, prob_visible_units)
        return visible_units

    def prob_hidden_units(self, visible_units):
        activation = np.dot(visible_units, self.weights) + self.hidden_bias
        prob_hidden_units = self.sigmoid(activation)
        return prob_hidden_units

    def prob_visible_units(self, hidden_units):
        activation = np.dot(hidden_units, self.weights.T) + self.visible_bias
        prob_visible_units = self.sigmoid(activation)
        return prob_visible_units

    def gibbs_sampling(self, visible_units_k):
        hidden_units_k = self.sample_hidden_units(visible_units_k)
        visible_units_k = self.sample_visible_units(hidden_units_k)
        return visible_units_k, hidden_units_k

    def update_weights(self, visible_units_0, hidden_units_0, visible_units_k, hidden_units_k, learning_rate):
        positive_grad = np.outer(visible_units_0, hidden_units_0)
        negative_grad = np.outer(visible_units_k, hidden_units_k)
        self.weights += learning_rate * (positive_grad - negative_grad)
        self.visible_bias += learning_rate * (visible_units_0 - visible_units_k)
        self.hidden_bias += learning_rate * (hidden_units_0 - hidden_units_k)

    def train(self, train_data, epochs, learning_rate, k_iterations):
        for epoch in range(epochs):
            for data in train_data:
                # Gibbs Sampling
                visible_units_0 = data
                hidden_units_0 = self.sample_hidden_units(visible_units_0)
                
                visible_units_k, hidden_units_k = self.gibbs_sampling(visible_units_0)
                
                # Update weights and biases
                self.update_weights(visible_units_0, hidden_units_0, visible_units_k, hidden_units_k, learning_rate)

    def recognize(self, test_data, k_iterations):
        recognized_chars = []
        for data in test_data:
            visible_units_0 = data
            hidden_units_0 = self.sample_hidden_units(visible_units_0)
            
            # Perform Gibbs Sampling to get visible units after k iterations
            for _ in range(k_iterations):
                visible_units_k, hidden_units_k = self.gibbs_sampling(visible_units_0)

            # Choose the most probable visible unit
            recognized_char = np.argmax(visible_units_k)
            recognized_chars.append(recognized_char)
        
        return recognized_chars

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Load MNIST training dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

# Create a list to store the training data
train_data = []
for _, (data, _) in enumerate(train_loader):
    train_data.append(torch.flatten(data).numpy())
train_data = np.array(train_data)

visible_units = train_data.shape[1]
hidden_units = 100
epochs = 10
learning_rate = 0.1
k_iterations = 10

# Create a Boltzmann Machine object
boltzmann_machine = BoltzmannMachine(visible_units, hidden_units)

# Run the training process
boltzmann_machine.train(train_data, epochs, learning_rate, k_iterations)

# Generate a random visible unit as input for testing
test_input = np.random.binomial(1, 0.5, visible_units)

# Perform Gibbs Sampling with k iterations on the test input
_, test_output = boltzmann_machine.gibbs_sampling(test_input)

# Reshape the test output into a 28x28 image
test_output_image = test_output.reshape((28, 28))

# Display the test output image
import matplotlib.pyplot as plt
plt.imshow(test_output_image)
