from abc import ABC

import torch
from interfaces.layer import NetworkLayer
from torch.utils.data import TensorDataset


class InputLayer(NetworkLayer, ABC):
    """
    INTERFACE
    Defines the input layer of an ANN -> Every input layer should implement this class
    @instance attr.
        PARENT ATTR.
            * Not used for this layer *
        OWN ATTR.
    """
    def __init__(self) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            None
        @return
            None
        """
        super().__init__(0, 0)


    @staticmethod
    def setup_data(data: str, 
                   label: str, 
                   filename: str, 
                   size: int
                   ) -> TensorDataset:
        raise NotImplementedError("This method has yet to be implemented.")
        

    @staticmethod
    def convert(img_file: str, 
                label_file: str, 
                out_file: str, 
                data_size: int, 
                img_size: int
                ) -> None:   
        raise NotImplementedError("This method has yet to be implemented.")
        

    # List of all methods from NetworkLayer that are not needed for this layer
    # TODO: find a better way to implement the logic of having an input processing layer that still extends the layer.NetworkLayer interface
    def create_id_tensors(self): pass
    def update_weights(self, input: torch.Tensor, output: torch.Tensor, clamped_output: torch.Tensor = None): pass
    def update_bias(self, output: torch.Tensor): pass
    def forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None, freeze: bool = False): pass
    def _train_forward(self, input: torch.Tensor, clamped_output: torch.Tensor = None): pass
    def _eval_forward(self, input: torch.Tensor): pass
    def visualize_weights(self, result_path: str, num: int, use: str, fname: str): pass
    def active_weights(self, beta: float): pass