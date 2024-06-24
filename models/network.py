# ------------------------------
# Import classes/modules
# ------------------------------

from abc import ABC, abstractmethod
import torch.nn as nn


"""
Interface for all networks to be created
"""
class Network(nn.Module, ABC):

    def __init__(self, device):
        super().__init__()
        self.device_id = device


    '''
    This function provides a mechanism to retrieve a specific layer from a neural network model based on the layer name
    '''
    def get_module(self, name):

    # STEP 1: loop through the network layeres
        for module_name, module in self.named_children():       
            # self.named_children() is a method provided by PyTorch’s nn.Module class 
            # It yields pairs of layer names (module_name) and the corresponding layer objects (module)

    # STEP 2: Layer matching and retrieval
            if name == module_name:
                return module


    '''
    This method aims to set or reset the scheduler for all layers within a network. 
    A scheduler typically adjusts the learning rate or other parameters over time or based on certain criteria, such as the number of epochs or the loss values.
    '''
    def set_scheduler(self):

    # STEP 1: Iterate through child modules
        for module in self.children():
            # self.children() is a method from PyTorch’s nn.Module that iterates over all direct children modules (layers) of the current network module.
            
            module.set_scheduler()
            # The line above calls set_scheduler on each child module


    '''
    This function sets a specific scheduler for a specific layer within the network. 
    
    This allows for customized control over how parameters are adjusted in individual layers, which might have different training dynamics.
    '''
    def set_layer_scheduler(self, name, scheduler):
        layer = self.get_module(name)
        # Uses the get_module method to retrieve the layer by its name. 
        # This assumes the layer names are uniquely defined within the network.

        layer.set_scheduler(scheduler)
        # This calls the set_scheduler method on the retrieved layer, passing the scheduler as an argument. 
        # This sets the scheduler specifically for this layer, allowing it to update its parameters according to the scheduler’s logic.




    '''
    This method is designed to visualize the weights or features learned by each neuron during training. 
    '''
    def visualize_weights(self, path, num, use):
        for module in self.children():
            module.visualize_weights(path, num, use)



    '''
    This method aims to determine and report the number of active neurons in each layer of the network.
    '''
    def active_weights(self, beta):

        # Iterate and shows back active weights
        for name, module in self.name_children():
            # For each module, return the count of active weights, and prints this count along with the module’s name.
            print(f"{name}: {module.active_weights(beta)}")



    '''
    This method is intended to process input data (x) through the network. 
    The optional clamped_output parameter suggests some form of output control or restriction, possibly used for training or specific types of inference.
    '''
    @abstractmethod  
    def forward(self, x, clamped_output=None):
        pass




























































