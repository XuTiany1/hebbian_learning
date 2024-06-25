# ------------------------------
# Import classes/modules
# ------------------------------

# Built-in imports
import os
import shutil
import time
from abc import ABC
import argparse
from typing import Tuple, List

# Pytorch imports
import torch
from torch.utils.data import DataLoader, TensorDataset

# Custom defined model imports
from models.network import Network

# Utils imports
from utils.experiment_logger import *
from utils.experiment_parser import *
from utils.experiment_timer import *



'''
Abstract class for for setting up, running, and logging different types of neural network experiments
'''
class Experiment(ABC):


    '''
        @param
            model: model to be trained and tested in experiment
            args: all arguments passed for experiment
            name: name of experiment
        @return
            None
    '''
    def __init__(self, model: Network, args: argparse.Namespace, name: str) -> None:

        # Initialize model and parameters
        self.model: Network = model.to(args.device_id).float()
        self.ARGS: argparse.Namespace = args


        # Set up timers
        self.START_TIME: float = None
        self.END_TIMER: float = None
        self.DURATION: float = None
        self.TRAIN_TIME: float = 0
        self.TEST_ACC_TIME: float = 0
        self.TRAIN_ACC_TIME: float = 0
             # Here, I Initialize various time-tracking attributes to measure different phases of the experiment

        # Set up result directories
        self.EXP_NAME: str = name
        self.RESULT_PATH: str = f"results/experiment-{self.EXP_NAME}"               # Defines the directory path for storing experiment results.
        
        # If the result path does not exist, it creates the necessary directories. 
        if not os.path.exists(self.RESULT_PATH):
            os.makedirs(self.RESULT_PATH, exist_ok=True)
            os.makedirs(f"{self.RESULT_PATH}/classification", exist_ok=True)
            os.makedirs(f"{self.RESULT_PATH}/hebbian", exist_ok=True)
            print(f"Experiment '{self.EXP_NAME}' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/classification' result folder created successfully.")
            print(f"Experiment '{self.EXP_NAME}/hebbian' result folder created successfully.")
        
        # If the path exists, it removes the old directory and creates a new one, ensuring a clean state for the new experiment.
        else:
            try:
                shutil.rmtree(self.RESULT_PATH)
                print(f"Removed {self.RESULT_PATH}.")
                os.makedirs(self.RESULT_PATH, exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/classification", exist_ok=True)
                os.makedirs(f"{self.RESULT_PATH}/hebbian", exist_ok=True)
                print(f"Experiment {self.EXP_NAME} result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/classification' result folder re-created successfully.")
                print(f"Experiment '{self.EXP_NAME}/hebbian' result folder re-created successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}")

        # Loggers for experiment
        self.loggers: List[logging.Logger] = []
        self.PRINT_LOG: logging.Logger = get_print_log("Print Log", self.RESULT_PATH) # Replace print statements (for debugging purposes)
        self.TEST_LOG: logging.Logger = get_test_acc_log("Test Log", self.RESULT_PATH) # Testing accuracy
        self.TRAIN_LOG: logging.Logger = get_train_acc_log("Train Log", self.RESULT_PATH) # Training accuracy
        self.PARAM_LOG: logging.Logger = get_parameter_log("Parameter Log", self.RESULT_PATH) # Experiment parameters
        self.DEBUG_LOG: logging.Logger = get_debug_log("Debug Log", self.RESULT_PATH) # Debugging stuff
        self.EXP_LOG: logging.Logger = get_experiment_log("Experiment Log", self.RESULT_PATH) # Logs during experiment
        self.loggers.append(self.PRINT_LOG)
        self.loggers.append(self.TEST_LOG)
        self.loggers.append(self.TRAIN_LOG)
        self.loggers.append(self.PARAM_LOG)
        self.loggers.append(self.DEBUG_LOG)
        self.loggers.append(self.EXP_LOG)
        # Here, I sets up multiple loggers to handle different aspects of logging for the experiment

        self.EXP_LOG.info("Completed imports.")
        self.EXP_LOG.info("Completed log setups.")
        self.EXP_LOG.info("Completed arguments parsing.")
        self.EXP_LOG.info(f"Experiment '{self.EXP_NAME}' result folder created successfully.")


    # This method should be implemented in its children classes
    def training(self, train_data_loader: DataLoader, epoch: int, visualize: bool = True) -> None:
        raise NotImplementedError("This method was not implemented.")
    
    # Similarly, this method should be implemented in its children classes
    def testing(self, test_data_loader: DataLoader, set_name: str, epoch: int, visualize: bool = True) -> float:
        raise NotImplementedError("This method was not implemented.")

    

    '''
    This method basically runs the experiment
    '''
    def run(self) -> Tuple[float, float]:


        # Start timer
        self.START_TIME = time.time()
        self.EXP_LOG.info("Start of experiment.")

        torch.device(self.ARGS.device_id)                                   # NOTE: Should this line be here or used where we create the experiment itself
        self.PRINT_LOG.info(f"local_machine: {self.ARGS.local_machine}.")


        # Logging training parameters
        self.EXP_LOG.info("Started logging of experiment parameters.")
        self.PARAM_LOG.info(f"Input Dimension: {self.ARGS.input_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Dimension: {self.ARGS.heb_dim}")
        self.PARAM_LOG.info(f"Outout Dimension: {self.ARGS.output_dim}")
        self.PARAM_LOG.info(f"Hebbian Layer Lambda: {self.ARGS.heb_lamb}")
        self.PARAM_LOG.info(f"Hebbian Layer Gamma: {self.ARGS.heb_gam}")
        self.PARAM_LOG.info(f"Classification Layer Lambda: {self.ARGS.cla_lamb}")
        self.PARAM_LOG.info(f"Network Learning Rate: {self.ARGS.lr}")
        self.PARAM_LOG.info(f"Epsilon: {self.ARGS.eps}")
        self.PARAM_LOG.info(f"Number of Epochs: {self.ARGS.epochs}")
        self.PARAM_LOG.info(f"Start time of experiment: {time.strftime('%Y-%m-%d %Hh:%Mm:%Ss', time.localtime(self.START_TIME))}")
        
        self.EXP_LOG.info("Completed logging of experiment parameters.")
            # Here, I logs all relevant experiment parameters for future reference and reproducibility


        # Dataset and dataloader setup
        train_data_set: TensorDataset = self.model.get_module("Input Layer").setup_data(dataset_type='train')
        train_data_loader: DataLoader = DataLoader(train_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for training dataset and dataloader.")

        # Testing dataset
        test_data_set: TensorDataset = self.model.get_module("Input Layer").setup_data(dataset_type='test')
        test_data_loader: DataLoader = DataLoader(test_data_set, batch_size=self.ARGS.batch_size, shuffle=True)
        self.EXP_LOG.info("Completed setup for testing dataset and dataloader.")


        # Training and Testing loops

        self.EXP_LOG.info("Started training and testing loops.")
        for epoch in range(0, self.ARGS.epochs):
            self.training(train_data_loader, epoch, visualize=True)


            # At specified intervals (self.ARGS.test_epochs), 
            #  I tests the model using both the test and training datasets to log accuracy and visualize results.
            if epoch % self.ARGS.test_epochs == 0:
                # Testing accuracy
                self.testing(test_data_loader, 'test', epoch, visualize=True)
                # Training accuracy
                self.testing(train_data_loader, 'train', epoch, visualize=True)


        # Post training steps
            # This includes:
                # 1. Visualize the weights of the model after training.
                # 2. Compute final testing and training accuracies, logging the results.
        self.EXP_LOG.info("Completed training of model.")        
        self.model.visualize_weights(self.RESULT_PATH, self.ARGS.epochs, 'final')
        self.EXP_LOG.info("Visualize weights of model after training.")
        test_acc = self.testing(test_data_loader, 'test', self.ARGS.epochs, visualize=True)
        train_acc = self.testing(train_data_loader, 'train', self.ARGS.epochs, visualize=True)
        self.EXP_LOG.info("Completed final testing methods.")
        self.PARAM_LOG.info(f"Training accuracy of model after training for {self.ARGS.epochs} epochs: {train_acc}")
        self.PARAM_LOG.info(f"Testing accuracy of model after training for {self.ARGS.epochs} epochs: {test_acc}")


        # End timer
        self.END_TIME = time.time()
        self.DURATION = self.END_TIME - self.START_TIME
        self.EXP_LOG.info(f"The experiment took {time_to_str(self.DURATION)} to be completed.")
        self.PARAM_LOG.info(f"End time of experiment: {time_to_str(self.END_TIME)}")
        self.PARAM_LOG.info(f"Runtime of experiment: {time_to_str(self.DURATION)}")
        self.PARAM_LOG.info(f"Train time of experiment: {time_to_str(self.TRAIN_TIME)}")
        self.PARAM_LOG.info(f"Test time (test acc) of experiment: {time_to_str(self.TEST_ACC_TIME)}")
        self.PARAM_LOG.info(f"Test time (train acc) of experiment: {time_to_str(self.TRAIN_ACC_TIME)}")
        self.EXP_LOG.info("The experiment has been completed.")
        
        return (test_acc, train_acc)



    def cleanup(self) -> None:
        """
        METHOD
        Cleanup used resources
        @param
            None
        @return
            None        
        """
        for logger in self.loggers:
            close_logger(logger)

















































