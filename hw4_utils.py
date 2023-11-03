
# Paste your code here to test it on autograder, this should include and_gate, or_gate,
# threshold_activation1, predict_output_v2, preprocess_data, MLP. This will create a file
# called hw4_utils.py. Note that even if some Errors show up in the autograder, it does
# not mean your code does not work. We will still look into your implementation manually.

import numpy as np
import torch
import torch.nn as nn
import random


def threshold_activation1(x):
        """
    TODO: Implement one activation function (unit step function)

    Args:
      x (np.ndarray): input array

    Returns (np.ndarray): output array (with the same shape as input array)

    """
    # TODO:
    threshold = 0
    output = np.zeros_like(x)
    output[x >= threshold] = 1

    return output

def and_gate(x):
    """
    TODO: Implement an "AND" gate

    Args:
      x (np.ndarray): array with shape (n, 1), representing n neurons as inputs.

    Returns: (int): scalar of 1 or 0
    """
    # TODO:
    output = 1
    for i in x:
      if i != 1:
        output = 0
        break
    return output

def or_gate(x):
    """
    TODO: Implement an "OR" gate

    Args:
      x (np.ndarray): array with shape (n, 1)

    Returns: (int): scalar of 1 or 0
    """
    # TODO:
    output = 0
    for i in x:
      if i == 1:
        output = 1
        break
    return output

def predict_output_v2(X, W, b):
    """
    #TODO: Update usage of the gates in this function
    """
    ## Cache of Predictions
    predictions = []
    ## Cycle Trhough Data Points
    for idx in range(data.shape[0]):
        x = np.reshape(X[idx, :], (2, 1))
        # First layer
        first_layer_output = np.matmul(W, x) + b
        first_layer_output = threshold_activation1(first_layer_output)
        # Second layer
        first_polygon = first_layer_output[0:5, :]
        second_polygon = first_layer_output[5:10, :]
        first_gate_output = and_gate(first_polygon)
        second_gate_output = or_gate(second_polygon)
        # Output layer
        input_to_final_gate = [first_gate_output, second_gate_output]
        prediction = and_gate(input_to_final_gate)
        predictions.append(prediction)
    return predictions

def preprocess_data(X, Y, test_split=1/6):
  num_samples = X.shape[0]
  num_train_samples = int(num_samples * (1-test_split))

  data = np.column_stack((X,Y))
  np.random.shuffle(data)

  X_shuffled = data[:,:-1]
  Y_shuffled = data[:, -1]

  # Split the data into training and testing sets
  X_train, X_test = X_shuffled[:num_train_samples], X_shuffled[num_train_samples:]
  y_train, y_test = Y_shuffled[:num_train_samples], Y_shuffled[num_train_samples:]

  # Standardize X by subtracting the mean and dividing by the standard deviation
  mean = np.mean(X_train, axis=0)
  std = np.std(X_train, axis=0)

  X_train = (X_train - mean) / (std + 1e-8)
  X_test = (X_test - mean) / (std + 1e-8)

  # Convert the data to torch.Tensor objects
  X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
  y_train, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

  return X_train, X_test, y_train, y_test

class MLP(nn.Module):
    def __init__(self, input_dim, layers_dims, output_dim, seed_value=None):
        """
        Initialize MLP.
        """
        super(MLP, self).__init__()

        ## TODO:
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layers_dims[0]))
        # self.layers.append(nn.Sigmoid())

        for i in range(1, len(layers_dims)):
          self.layers.append(nn.Linear(layers_dims[i-1], layers_dims[i]))
        #   self.layers.append(nn.Sigmoid())

        self.layers.append(nn.Linear(layers_dims[-1], output_dim))
        self.seed_value = seed_value
        self._initialize_weights()


    def _initialize_weights(self):
        """
        Initialize the weights and biases of the model.
        """

        ## TODO:
        for layer in self.layers:
          if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.001)
            # nn.init.normal_(layer.bias, mean=0, std=0.01)


    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        # TODO:
        for layer in self.layers:
          x = layer(x)
          x = torch.sigmoid(x)
          # print("Layer Output Shape:", x.shape)  # Print the shape of the output at each layer
        return x
