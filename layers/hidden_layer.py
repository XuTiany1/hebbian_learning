from abc import ABC
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces.layer import NetworkLayer


class HiddenLayer(NetworkLayer, ABC):
    """
    INTERFACE
    Defines a single hidden layer in ANN -> Every hidden layer should implement this class
    @instance attr.
        PARENT ATTR.
            input_dimension (int): number of inputs into the layer
            output_dimension (int): number of outputs from layer
            device (str): device that will be used for CUDA
            lr (float): how fast model learns at each iteration
            fc (nn.Linear): fully connected layer using linear transformation
        OWN ATTR.
            exponential_average (torch.Tensor): 0 tensor to keep track of exponential averages
            gamma (float): decay factor -> factor to decay learning rate
            lamb (float): lambda hyperparameter for lateral inhibition
            eps (float): to avoid division by 0
            id_tensor (torch.Tensor): id tensor of layer
    """

    def __init__(self,
                 input_dimension: int,
                 output_dimension: int,
                 device: str,
                 learning_rate: float = 0.005,
                 lamb: float = 1,
                 gamma: float = 0.99,
                 eps: float = 0.01
                 ) -> None:
        """
        CONSTRUCTOR METHOD
        @param
            input_dimension: number of inputs into the layer
            output_dimension: number of outputs from the layer
            device: the device that will be used
            learning_rate: how fast model learns at each iteration
            lamb: lambda hyperparameter for lateral inhibition
            gamma: this param affects exponential average updates
            eps: this affects weight decay updates

        @return
            None
        """
        # Use constructor from its super class, the layer interface
        super().__init__(
            input_dimension,
            output_dimension,
            device,
            learning_rate
        )

        # Here, I am initializing some param that are specific to the hidden layer, ARGUMENTS ARE PROVIDED 
        self.gamma: float = gamma
        self.lamb: float = lamb
        self.eps: float = eps

        # Here, I am initializing some other param that are specific to hidden layer,
            # However, in this case, the arguments are not needed from calling the constructor
        self.exponential_average: torch.Tensor = torch.zeros(self.output_dimension).to(self.device)
        self.id_tensor: torch.Tensor = self.create_id_tensors().to(self.device)


    #################################################################################################
    # Different Inhibition Methods
    
    # Lateral inhibition is a process used to enhance the contrast between the activations of neurons, 
    # ensuring that the most active neurons suppress the activity of their neighbors.
    #################################################################################################


    def _relu_inhibition(self,
                         input: torch.Tensor
                         ) -> torch.Tensor:
        """
        METHOD
        This method calculates ReLU lateral inhibitiom

        @param
            input: input to the layer
        @return
            output: this is the activation values after applying ReLU lateral inhibition
        """

        # First, I get the ReLU activation function
        relu: nn.ReLU = nn.ReLU()

        # Second, I compute ReLU and lateral inhibition
            # Steps are as follows:
                # 1. Pass input through relu       -> (This step zeroes out any negative activation values, ensuring that only positively activated neurons are considered for further processing)
                # 2. Find maximum element from all
                # 3. Raise all values to the power of lambda
                # 4. Normalize to ensure that the activations remain within a manageable range
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        input_copy = relu(input_copy)
        max_ele: int = torch.max(input_copy).item()
        input_copy = torch.pow(input_copy, self.lamb)
        output: torch.Tensor =  (input_copy / abs(max_ele) ** self.lamb).to(self.device)

        return output
    

    
    def _softmax_inhibition(self, 
                            input: torch.Tensor
                            ) -> torch.Tensor:
        """
        METHOD
        This method calculates the softmax lateral inhibition

        @param
            input: input to the layer
        @return
            output: this is the activation values after applying SOFTMAX lateral inhibition
        """

        # All this method does is to apply softmax to the input
            # So as opposed to linear ReLU, softmax has a ceiling and floor
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        output: torch.Tensor = F.softmax(input_copy, dim=-1).to(self.device)
        return output


    def _exp_inhibition(self, input: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Calculates exponential (Modern Hopfield) lateral inhibition
        @param
            input: input to layer
        @return
            output: activation after lateral inhibition
        """

        # The steps for exponential inhibition is as follows:
            # First, I find the largest element
            # Then, the denominator, which will be the normalization term, 
                # will be the SUMMATION over all 
                # the exponential of (lambda * (current_input_value - max_input_value))
            # Lastly, the numerator will be different for every single input term
                # This will be the exponential of (lambda * (current_input_value - max_input_value))
        input_copy: torch.Tensor = input.clone().detach().float().to(self.device)
        max_ele: int = torch.max(input_copy).item()
        output: torch.Tensor = F.softmax((input_copy - max_ele) * self.lamb, dim=-1).to(self.device)
        return output
    


    #################################################################################################
    # Different Weight Updates Methods

    # Here, I am just implementing different methods to update weights
    #################################################################################################
    def _hebbian_rule(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        METHOD
        Computes Hebbian Leanring Rule -> this is the vanilla hebbian learning rule
        @param
            input: the input of the layer
            output: the output of the layer
        @return
            computed_rule: this is the delta_weight value
        """
        # Copy both input and output for future calculation
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Calculate outer product of output and input
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)

        # Calculate Hebbian Learning Rule
        computed_rule: torch.Tensor = outer_prod.to(self.device)

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule



    def _sanger_rule(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        METHOD
        Computes Sanger's Rule
            This rule not only update weights in a Hebbian manner but also ENSURE ORTHOGONALITY among the weight vectors
            This is particularly useful for principal component analysis (PCA), where the goal is to extract the principal components of the input data
            From a computational perspective, Sanger rule ensures that the weight vectors evolve to become the principal components of the input data

            So, in essence, Sanger rule extends traditional Hebbian learning by adding a correction term that enforces orthogonality among the weight vectors

            
            The function below does the following:
                	•	Step 1: Copy input and output tensors for processing.
                    •	Step 2: Calculate the outer product of the output and input vectors.
                    •	Step 3: Retrieve and transpose the initial weights.
                    •	Step 4: Calculate the correction term for each output neuron to ensure orthogonality.
                    •	Step 5: Compute the final change in weights by subtracting the correction term from the outer product.
                    •	Step 6: Update exponential averages if necessary.

        @param
            input: the input of the layer
            output: the output of the layer
        @return
            computed_rule: this is the delta_weight value
        """
        # Copy both input and output to be used in Sanger's Rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)
        
        # Step 1: Calculate outer product of output and input
        # O = y ⊗ x
        outer_prod: torch.Tensor = torch.tensor(outer(y.cpu().numpy(), x.cpu().numpy())).to(self.device)
            # This results in a matrix where each element is the product of the corresponding elements of y and x.

        # Step 2: Retrieve initial weights (transposed) 
        # W_init = W^T
        initial_weight: torch.Tensor = torch.transpose(self.fc.weight.clone().detach().to(self.device), 0, 1)
            # This ensures that the weight update does not interfere with the original weights during computation.

        # Step 3: Calculate Sanger's Rule
        A: torch.Tensor = torch.einsum('jk, lkm, m -> lj', initial_weight, self.id_tensor, y).to(self.device)
        A = A * (y.unsqueeze(1))
        '''
        Note, the above is sort of complex, but it basically does the same thing as the following:

        # Note the FIRST I initialize the correction tensor
        correction: torch.Tensor = torch.zeros_like(initial_weight)           # Here, I have a tensor correction is initialized with the same shape as initial_weight, filled with zeros.
        for i in range(len(y)):
            correction[:, i] = y[i] * (initial_weight[:, :i+1] @ y[:i+1])     # HERE, I loop through each element in the output vector y.
                                                                              # Now, for each output neuron i, compute the correction term as follows:
                                                                                        # •	initial_weight[:, :i+1] extracts the first i+1 columns of the weight matrix.
                                                                                          •	y[:i+1] extracts the first i+1 elements of the output vector.
                                                                                          •	The matrix-vector multiplication initial_weight[:, :i+1] @ y[:i+1] computes the sum of the products of the weights and corresponding outputs for the first i+1 neurons.
                                                                                          •	The result is then scaled by y[i].
        '''
        computed_rule: torch.Tensor = (outer_prod - A).to(self.device)
            # The final weight update computed_rule is obtained by subtracting the correction term A from the outer product.
            # This ensures that the weight update aligns with Sanger’s rule.
            # The final weight update computed_rule is obtained by subtracting the correction term correction from the outer product outer_prod.

        # Update exponential averages
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule



    def _fully_orthogonal_rule(self, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Update weights using Fully Orthogonal Rule.
        """
        # Copy both input and output to be used in the rule
        x: torch.Tensor = input.clone().detach().float().squeeze().to(self.device)
        x.requires_grad_(False)
        y: torch.Tensor = output.clone().detach().float().squeeze().to(self.device)
        y.requires_grad_(False)

        # Step 1: Calculate outer product of output and input
        # O = y ⊗ x
        outer_prod: torch.Tensor = torch.ger(y, x)
            # This results in a matrix where each element is the product of the corresponding elements of y and x.

        # Step 2: Retrieve initial weights
        # W_init = W
        initial_weight: torch.Tensor = self.fc.weight.clone().detach().to(self.device)
            # This ensures that the weight update does not interfere with the original weights during computation.

        # Step 3: Calculate the dot product between output and initial weights
        # y^T W = ytw
        ytw = torch.matmul(y.unsqueeze(0), initial_weight).to(self.device)
            # ytw is computed as the dot product of y (as a row vector) and initial_weight. 
            # This results in a vector that represents the projection of the output vector y onto the initial weights.

        # Step 4: Calculate the norm term
        # N = y ⊗ (y^T W)
        norm_term = torch.outer(y.squeeze(0), ytw.squeeze(0)).to(self.device)
            # norm_term is calculated as the outer product of y and ytw. 
            # This term represents the orthogonal projection adjustment that ensures the weights maintain orthogonality.

        # Step 5: Compute the change in weights
        # ΔW = O - N
        computed_rule: torch.Tensor = (outer_prod - norm_term).to(self.device)
            # The final weight update computed_rule is obtained by subtracting norm_term from outer_prod. 
            # This ensures that the updated weights maintain orthogonality.


        """
        Step 4 and 5 explained further:

        In essense...
            Step 4: Calculate the orthogonal projection adjustment (norm term) by computing the dot product of y and the initial weights, and then computing the outer product of y and this dot product result.
            Step 5: Compute the final weight update by subtracting the norm term from the outer product of y and x.
        
        """


        # Update exponential averages if necessary
        self.exponential_average = torch.add(self.gamma * self.exponential_average, (1 - self.gamma) * y)
        
        return computed_rule




