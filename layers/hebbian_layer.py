# ------------------------------
# BASIC IMPORTS
# ------------------------------

import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from numpy import outer
from layers.layer import NetworkLayer


class HebbianLayer(NetworkLayer):

    def __init__(self, input_dimension, output_dimension, device_id, lamb=2, heb_lr=0.001, gamma=0.99, eps=10e-5):

        # Calling super's constructor
        super().__init__(input_dimension, output_dimension, device_id, lamb, heb_lr, eps)


        # Extra hyperparameter set up
        self.gamma = gamma

        # Other useful things set up
        self.exponential_average = torch.zeros(self.output_dimension)   # Initializes self.exponential_average as a tensor of zeros with the size of output_dimension.
        self.id_tensor = self.create_id_tensors()                       # This function is specified in the layer.py



    """
    This function mimicks the behaviour of classical hopfield net!

    This function implements a form of lateral inhibition with a ReLU activation function

    Lateral inhibition is a process where an activated neuron suppresses the activity of its neighbors, enhancing the contrast in the activity pattern.
    
    This function ensures that stronger activations are further emphasized
    This can lead to more distinct and sparse activations, which can be beneficial

    @param
        x (torch.Tensor) = input to the ReLU function
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def relu_inhibition(self, x):

        # Step 1: ReLU Activation
        relu = nn.ReLU()
        x = relu(x)

        # Step 2: Find max element -> this step finds the maximum value in the tensor x after ReLU activation
        max_ele = torch.max(x).item()
            # The .item() method converts the tensor containing the maximum value to a Python scalar.


        # Step 3: Apply power function
        x = torch.pow(x, self.lamb)
            # Raises each element of the tensor x to the power of self.lamb.
            # This operation enhances the differences between the activated values, with higher values becoming more pronounced.


        # Step 4: Normalization
        x /= abs(max_ele) ** self.lamb
            # Normalizes the tensor x by dividing it by the absolute value of the maximum element raised to the power of self.lamb.
            # This ensures that the scaled values are within a consistent range, preventing them from becoming excessively large or small.

        return x
    



    """
    This function implements the divisive normalization inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def divisive_normalization_inhibition(x, epsilon=1e-6):
        sum_activity = torch.sum(x)
        x = x / (sum_activity + epsilon)
        return x
    



    """
    This function implements the gaussian inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def gaussian_inhibition(x, sigma=1.0):
        size = int(2 * sigma + 1)
        kernel = torch.tensor([torch.exp(-(i - size // 2) ** 2 / (2 * sigma ** 2)) for i in range(size)])
        kernel = kernel / torch.sum(kernel)

        x = F.conv1d(x.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=size//2).squeeze(0).squeeze(0)
        return x
    


    """
    This function implements the winner takes all inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def winner_take_all_inhibition(x, top_k=1):
        topk_values, _ = torch.topk(x, top_k)
        threshold = topk_values[-1]
        x = torch.where(x >= threshold, x, torch.tensor(0.0, device=x.device))
        return x


    """
    This function implements the softmax inhibition
    @param
        x (torch.Tensor) = input
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def softmax_inhibition(x):
        x = torch.exp(x - torch.max(x))
        return x / torch.sum(x)
        


    """
    Calculates lateral inhibition using an exponential function -> This function mimics MODERN HOPFIELD NET
    @param
        x (torch.Tensor)
    @return
        x (torch.Tensor) = activation after lateral inhibition
    """
    def modern_hopfield_inhibition(self, x):
        relu = nn.ReLU()
        x = relu(x)  # Apply ReLU activation to ensure non-negative values
        max_ele = torch.max(x).item()  # Find the maximum element in the tensor

        # Apply the exponential function to the activations
        x = torch.exp(self.lamb * x)

        # Normalize the activations by dividing by the exponential of the maximum element
        x /= torch.exp(self.lamb * max_ele)
        
        return x


































































































