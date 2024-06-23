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




# -----------------------------------------------------------------------------------------------------------
#  LATERAL INHIBITION SECTION
# -----------------------------------------------------------------------------------------------------------

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




# -----------------------------------------------------------------------------------------------------------
#  UPDATE WEIGHT SECTION
# -----------------------------------------------------------------------------------------------------------


    '''
    This function updates the weights of the layer based on Sanger’s Rule, which is a generalized version of Oja’s learning rule.
    
    •	Hebbian Learning: This method follows the Hebbian principle, where the connection between neurons that fire together is strengthened.
	•	Stabilized Learning: The lateral inhibition term ensures that weight updates are stabilized, preventing runaway growth.
	•	Efficient Representation: Promotes efficient and sparse coding by balancing excitatory and inhibitory influences.

    '''
    def update_weights(self, input, output, clamped_output=None):
        
    # STEP 1: Extract and prepare input and output vectors so that I can update the weights accordingly
        x = input.clone().detach().float().squeeze().to(self.device_id)
        x.requires_grad_(False)
        y = output.clone().detach().float().squeeze().to(self.device_id)
        y.requires_grad_(False)

        # What is done in the above lines of code is the following:
            # The input (x) and output (y) tensors are cloned, detached from the computation graph, converted to float, squeezed to remove any singleton dimensions, and moved to the specified device (self.device_id)
            # requires_grad_(False) ensures these tensors will not track gradients.


    # STEP 2: Calculate the outer product
        outer_prod = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy()))  # Move tensors to CPU before calling outer
        outer_prod = outer_prod.to(self.device_id)
        # Here, the outer product of the output (y) and input (x) vectors is computed and then moved back to the GPU


    # STEP 3: Retrieve and Prepare initial weights
        initial_weight = torch.transpose(self.fc.weight.clone().detach().to(self.device_id), 0, 1)
        # Here, the current weights of the fully connected layer (self.fc.weight) are cloned, detached, and transposed. This prepares the weights for subsequent calculations.

        
    # STEP 4: Ensure Tensors are on the same device
        self.id_tensor = self.id_tensor.to(self.device_id)
        self.exponential_average = self.exponential_average.to(self.device_id)
        # Here, I move the id_tensor and exponential_average tensors to the same device as the other tensors to ensure compatibility in calculations.


    # STEP 5: Compute lateral inhibition term
        A = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y)
        A = A * (y.unsqueeze(1))  # 'A' represents the inhibitory effect based on current weights and activations.

    # TODO -> UNDERSTAND THIS EINSUM!


    # STEP 6: Compute weight update
        delta_weight = self.alpha * (outer_prod - A)
        # Here, the weight update (delta_weight) is computed by subtracting the lateral inhibition term (A) from the outer product and scaling by the learning rate self.alpha.

    # STEP 7: Update the weights
        self.fc.weight = nn.Parameter(torch.add(self.fc.weight, delta_weight), requires_grad=False)
        # The weights of the fully connected layer are updated by adding delta_weight. 
        # The updated weights are wrapped in nn.Parameter with requires_grad=False to ensure they do not track gradients.


    # STEP 8: Update the exponential average
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        # The exponential average of the output activations is updated using a moving average formula with decay factor self.gamma.




    '''
    The weight_decay function adjusts the weights of the neural network layer by decaying overused weights and increasing underused weights using the hyperbolic tan function.
    
    '''
    def weight_decay(self):

        tanh = nn.Tanh()

    # STEP 1: Compute the average of exponential averages
        average = torch.mean(self.exponential_average).item()

    
    # STEP 2: Calculate the Ratio vs Mean
        ratio = self.exponential_average / average
        # Here, I have computed the ratio of each element in exponential_average to the average value. 
        # This ratio indicates how much each neuron’s activity deviates from the mean.


    # STEP 3: Calculate the growth factor
        growth_factor_positive = self.eps * tanh(-self.eps * (ratio - 1)) + 1
        growth_factor_negative = torch.reciprocal(growth_factor_positive)
        # Here, I compute the growth factors using the tanh function
            # This function decreases for values of  A  greater than 1 (overused weights) 
            # and this function increases for values less than 1 (underused weights).


    # STEP 4: 


 def weight_decay(self):
        tanh = nn.Tanh()

        # Step 1: Compute the average of exponential averages
            # This calculates the mean of the exponential_average tensor, which tracks the exponential moving average of the outputs of the layer.
            # .item() converts the result from a tensor to a Python scalar.
        average = torch.mean(self.exponential_average).item()

        # Step 2: Normalize the exponential_average tensor by dividing each element by the computed mean. 
            # This gives a ratio indicating how each element compares to the average.
        A = self.exponential_average / average

        # Step 3: calculate the growth factors
            # The difference between each ratio in A and 1 is scaled by eps, passed through the tanh function, and then scaled by eps again before adding 1. 
                # This growth factor is used to adjust weights that are greater than zero.
            # This is the reciprocal of growth_factor_positive and is used to adjust weights that are less than zero.
        growth_factor_positive = self.eps * tanh(-self.eps * (A - 1)) + 1
        growth_factor_negative = torch.reciprocal(growth_factor_positive)

        # Update the weights depending on growth factor
        positive_weights = torch.where(self.fc.weight > 0, self.fc.weight, 0.0)
        negative_weights = torch.where(self.fc.weight < 0, self.fc.weight, 0.0)
        positive_weights = positive_weights * growth_factor_positive.unsqueeze(1)
        negative_weights = negative_weights * growth_factor_negative.unsqueeze(1)
        self.fc.weight = nn.Parameter(torch.add(positive_weights, negative_weights), requires_grad=False)
        
        # Check if there are any NaN weights
        if (self.fc.weight.isnan().any()):
            print("NAN WEIGHT")



#~~~~~~~~~~~~~~ Reason behind using epsilon in weight decay~~~~~~~~~~~~~~~~

# Sensitivity Adjustment
    # The epsilon value is used to adjust the sensitivity of the growth factors calculated using the tanh function. 
    # By multiplying the difference between the normalized weight usage ratio (A) and 1 by epsilon, the model can fine-tune how aggressively the weights are adjusted.

    # A larger epsilon would make the tanh function’s output more responsive to changes in the weight usage ratio, leading to more significant adjustments. 
    # Conversely, a smaller epsilon would make the adjustments more gradual.

# Avoiding NaN Values
    # This is kinda straight forward

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






























# -----------------------------------------------------------------------------------------------------------
#  UPDATE BIAS SECTION       NOT REALLY USED
# -----------------------------------------------------------------------------------------------------------





    """
    Defines the way the biases will be updated at each iteration of the training
    It updates the biases of the classifier layer using a decay mechanism adjusted by the output probabilities.
    The method applies an exponential decay to the biases, which is modulated by the output probabilities,
    and scales the update by the learning rate. 
    The biases are normalized after the update.
    @param
        output (torch.Tensor) = The output tensor of the layer.
    @return
        ___ (void) = no returns
    """
    def update_bias(self, output):
        y = output.clone().detach().squeeze()
        exponential_bias = torch.exp(-1*self.fc.bias) # Apply exponential decay to biases

        # Compute bias update scaled by output probabilities.
        A = torch.mul(exponential_bias, y) - 1
        A = self.fc.bias + self.alpha * A

        # Normalize biases to maintain stability. (Divide by max bias value)
        bias_maxes = torch.max(A, dim=0).values
        self.fc.bias = nn.Parameter(A/bias_maxes.item(), requires_grad=False)
















































































