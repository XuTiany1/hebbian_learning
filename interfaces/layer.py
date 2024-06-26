from abc import ABC
import math
import matplotlib
import matplotlib.figure
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn



class NetworkLayer (nn.Module, ABC):
    """
    INTERFACE

    This defines a single layer of an ANN
    And so every layer must implemente this interface
    """

    def __init__(self, 
                 input_dimension: int, 
                 output_dimension: int, 
                 device: str = 'cpu',
                 learning_rate: float = 0.005
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from layer
            device: device that will be used for CUDA
            learning_rate: how fast model learns at each iteration
        @return
            None
        """
        super().__init__()
        self.input_dimension: int = input_dimension
        self.output_dimension: int = output_dimension
        self.device: str = device
        self.lr: float = learning_rate
        self.fc: nn.Linear = nn.Linear(self.input_dimension, self.output_dimension, bias=True)


        # Setup linear activation
        for param in self.fc.parameters():
            torch.nn.init.uniform(param, a=0.0, b=1.0)
            # Now, also since I am doing hebbian learning, I will disable backprop
            param.requires_grad_(False)



    def create_id_tensors(self) -> torch.Tensor:   
        """
        METHOD
        Creates an identity tensor
        @param
            None
        @return
            id_tensor: 3D tensor with increasing size of identify matrices
        """
        id_tensor: torch.Tensor = torch.zeros(self.output_dimension, self.output_dimension, self.output_dimension, dtype=torch.float)
        for i in range(0, self.output_dimension):
            identity: torch.tensor = torch.eye(i+1)
            padded_identity: torch.Tensor = torch.nn.functional.pad(identity, (0, self.output_dimension - i-1, 0, self.output_dimension - i-1))
            id_tensor[i] = padded_identity
        return id_tensor



    def update_weights(self,
                       input: torch.Tensor,
                       output: torch.Tensor,
                       clamped_output: torch.Tensor = None
                       ) -> None:
        """
        METHOD 
        Not implemented in interface level so each layer, depending on its configuration, will have its own way 
        of updating weights
        """
        raise NotImplementedError("UPDATE WEIGHT METHOD IS NOT IMPLEMENTED")



    def update_bias(self,
                    output: torch.Tensor
                    ) -> None:
        """
        METHOD
        Not implemented in interface level so each layer, depending on its configuration, will have its own way 
        of updating bias
        """
        raise NotImplementedError("UPDATE BIAS METHOD IS NOT IMPLEMENTED")

    
    def _train_forward(self,
                       input: torch.Tensor,
                       clamped_output: torch.Tensor = None
                       ) -> torch.Tensor:
        """
        METHOD

        This method will define how a layer forward propagate an input and then updates its weights based on output

        Not implemented in the interface level
        Hence, for each layer, depending on its configuration, will have its own way of its forward training pass
        """
        raise NotImplementedError("TRAIN-FORWARD METHOD IS NOT IMPLEMENTED")


    def _eval_forward(self,
                       input: torch.Tensor,
                       ) -> torch.Tensor:
        """
        METHOD

        This method just defines how a layer forward propagates an input

        Not implemented in the interface level
        Hence, for each layer, depending on its configuration, will have its own way of its forward EVALUATION pass
        """
        raise NotImplementedError("EVAL-FORWARD METHOD IS NOT IMPLEMENTED")



    def forward(self,
                input: torch.Tensor,
                clamped_output: torch.Tensor = None,
                freeze: bool = False
                ) -> torch.Tensor:
        """
        METHOD
        This method defines how input data flows through the network

        @param
            input: input data into the layer
            clamped output: one-hot encoded vectors of the true labels
            freeze: determine if the layer is frozen or not

        @return
            input: returns the data after passing it through the layer, so this will be the input to the next layer
        """

        # As long as:
            # 1. my model is in training mode and 
            # 2. not in freeze mode
        # Then, I will update my weights by running this train-forward function
        if (self.training) and (not freeze):
            input = self._train_forward(input, clamped_output)
        else:
        # Any other case, then my model will be in running the evaluation forward function, which is designed to just 
        # pass forward the data and not update any weights
            input = self._eval_forward(input)

        return input




    def visualize_weights(self, result_path: str, num: int, use: str, fname: str) -> None:
        """
        METHOD
        Vizualizes the weights/features learned by neurons in this layer using a heatmap
        @param
            result_path: path to folder where results will be printed
            num: integer representing certain property (for file name creation purposes)
            use: the use that called this method (for file name creation purposes)
            fname: name to be used for folder/file
        @return
            None
        """
        # Name of saved plot
        plot_name: str = f'/{fname}/{fname.lower()}layerweights-{num}-{use}.png'
        
        # Find value for row and column
        row: int = 0
        col: int = 0

        root: int = int(math.sqrt(self.output_dimension))
        for i in range(root, 0, -1):
            if self.output_dimension % i == 0:
                row = min(i, self.output_dimension // i)
                col = max(i, self.output_dimension // i)
                break
        
        # Get the weights and create heatmap
        weight: nn.parameter.Parameter = self.fc.weight
        fig: matplotlib.figure.Figure = None
        axes: np.ndarray = None
        fig, axes = plt.subplots(row, col, figsize=(16, 16))
        for ele in range(row*col):  
            random_feature_selector: torch.Tensor = weight[ele]
            # Move tensor to CPU, convert to NumPy array for visualization
            heatmap: torch.Tensor = random_feature_selector.view(int(math.sqrt(self.fc.weight.size(1))), int(math.sqrt(self.fc.weight.size(1)))).cpu().numpy()

            ax = axes[ele // col, ele % col]
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'Weight {ele}')
            
            # Move the tensor back to the GPU if needed
            random_feature_selector = random_feature_selector.to(self.device)
        
        file_path: str = result_path + plot_name
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
    
    
    def active_weights(self, beta: float) -> int:
        """
        METHOD
        Returns number of active feature selectors
        @param
            beta: cutoff value determining which neuron is active
        @return
            number of active weights
        """
        weights: nn.parameter.Parameter = self.fc.weight
        active: torch.Tensor = torch.where(weights > beta, weights, 0.0)
        return active.nonzero().size(0)


























