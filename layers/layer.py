# ------------------------------
# STEP 1: import classes/modules
# ------------------------------
from abc import ABC, abstractmethod # This imports Abstract Base Classes, used in python to create abstrac classes
import torch
import torch.nn as nn


class NetworkLayer (nn.Module, ABC):

    def __init__(self, input_dimension, output_dimension, device_id, lamb=1, learning_rate=0.005, eps=0.01):
        super().__init__()

        # Dimension of layer initialization
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # Device ID initialzation
        self.device_id = device_id

        # Hyperparameters initialziation
        self.lamb = lamb
        self.alpha = learning_rate

        # Linear layer initialziation
            # nn.Linear is a fully connected (linear) layer in PyTorch.
            # bias=True means that the layer will include a bias term (an additional parameter for each output node).
        self.fc = nn.Linear(self.input_dimension, self.output_dimension, bias=True)
            # Here, I am initializing weights in the fc layer 
        for param in self.fc.parameters():
            torch.nn.uniform_(param, a=0.0, b=1.0) # This function initializes the weights of the layer param uniformly between 0.0 and 1.0.
            param.requires_grad_(False) # This line sets the requires_grad attribute of the parameter to False, meaning that these weights will not be updated during the backpropagation process. 
            # The reason behind this is because I am trying to implement a Hebbian model, which requires no backprop!!!



    # TODO -> WHAT IS THE MEANING BEHIND THIS ID-TENSOR?
    '''
    Thie method generates a 3D tensor composed of identity matrices of increasing sizes, 
    padded to fit within a tensor of shape (self.output_dimension, self.output_dimension, self.output_dimension).
    '''
    def create_id_tensors(self):
        id_tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        
        for i in range(0, self.output_dimension):
            identity = torch.eye(i+1)
            padded_identity = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor



    # Method used to visualize weight/features learned in this layer
    # This is an abstract method, so MUST be implemented for each new layer
    @abstractmethod
    def visualize_weights(self, result_path, num, use):
        pass


    # Method is used to define the way the weights will be updated at each iteration of training
    # Another abstract method, so MUST ALSO be implemented for each new layer
    @abstractmethod
    def update_weights(self, input, output, clamped_output=None):
        pass



    # This updates the bias for each iteration of training 
    @abstractmethod
    def update_bias(self, output):
        pass


    
    # This method defines how the current layer(during training mode) processes the input forward and updates its weights
    '''
    @param
        x (torch.Tensor) = input data into the layer
        clamped_output (torch.Tensor) = one-hot encode of true labels
    @return
        data_input (torch.Tensor) = returns the data after passing it through the layer
    '''
    @abstractmethod
    def _train_forward(self, x, clamped_output=None):
        pass


    # This method defines how current layer(DURING EVAL MODE) processes input forward 
    @abstractmethod
    def _eval_forward(self,x):
        pass


    # This method gives me the number of active feature selectors based on a specific cutoff
    '''
    @param
        beta (float) = cutoff value determining which neuron is active
    @return
        ___ (void) = no returns
    '''
    @abstractmethod
    def active_weights(self, beta):
        pass

    # This method defines how an input 
    def forward(self, x, clamped_output=None):

        if self.training:
            x = self._train_forward(x, clamped_output)


















































































