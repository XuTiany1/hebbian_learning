from abc import ABC
import torch
import torch.nn as nn

from interfaces.layer import NetworkLayer
from utils.experiment_constants import LayerNames


class Network(nn.Module, ABC):
    """
    INTERFACE
    Defines an ANN that uses 1 or more NetworkLayers -> Every ANN for experiment must implement interface
    @instance attr.
        device (str): device to which calculations will be made
    """
    def __init__(self, 
                 device: str) -> None:
        """
        CONSTRUCTOR METHOD
        @attr.
            device: device that will be used for CUDA
        @return
            None
        """
        super().__init__()
        self.device = device


    def get_module(self, 
                   lname: LayerNames) -> NetworkLayer:
        """
        METHOD
        Returns layer with given name
        @param
            name: name of layer to get
        @return
            layer: layer of the network with searched name
        """
        for layer_name, layer in self.named_children():       
            if lname.name.upper() == layer_name.upper():
                return layer


    def visualize_weights(self, 
                          path: str, 
                          num: int, 
                          use: str) -> None:
        """
        METHOD
        Visualizes the weights/features learned by each neuron during training
        @param
            path: path to print out result
            num: which iteration is the visualization happening
            use: after what step is the visualization happening
        @return
            None
        """
        for name, module in self.named_children():
            module.visualize_weights(path, num, use, name.lower())


    def active_weights(self, 
                       beta: float) -> dict[str:int]:
        """
        METHOD
        Returns number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            module_active_weights: dictionary {str:int}
        """
        module_active_weights: dict[str:int] = {}
        
        for name, module in self.name_children():
            module_active_weights[name.lower()] = module.active_weights(beta)
        
        return module_active_weights


    def forward(self, 
                input: torch.Tensor, 
                clamped_output: torch.Tensor = None) -> torch.Tensor: 
        raise NotImplementedError("This method is not implemented.")