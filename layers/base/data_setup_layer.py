import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
from layers.input_layer import InputLayer
from typing import IO, List



class DataSetupLayer(InputLayer):
    """
    CLASS
    Defines how the input dataset will be processed before feeding it to the base network
    @instance attr.
        NetworkLayer ATTR.
            * Not used for this layer *
        InputLayer ATTR.
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
        super().__init__()


    @staticmethod
    def setup_data(data: str, label: str, filename: str, size: int) -> TensorDataset:
        """
        STATIC METHOD
        Function to setup requested dataset
        @param
            data: data filename
            label: label filename
            filename: data (img + label) filename
            size: number of data
        @return
            tensor dataset containing (data, label)
        """
        # Converting to .csv file if needed
        if not os.path.exists(filename):
            DataSetupLayer.convert(data, label, filename, size, 28)
         
        # Setup dataset   
        data_frame: pd.DataFrame = pd.read_csv(filename, header=None, on_bad_lines='skip')
        labels: torch.Tensor = torch.tensor(data_frame[0].values)
        data_tensor: torch.Tensor = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_tensor /= 255
        
        return TensorDataset(data_tensor, labels)



    @staticmethod
    def convert(img_file: str, 
                label_file: str, 
                out_file: str, 
                data_size: int, 
                img_size: int) -> None:
        """
        STATIC METHOD
        Convert .ubyte files into a .csv file for ease of use
        @param
            img_file: path to image files
            label_file: path to file with labels
            out_file: path to .cvs file that merges data and label
            data_size: number of data inputs
            img_size: size of image
        @return
            None
        """    
        # Get absolute path of all the necessary files (img, label, out)
        project_root: str = os.getcwd()
        img_file: str = os.path.join(project_root, img_file)
        label_file: str = os.path.join(project_root, label_file)
        out_file: str = os.path.join(project_root, out_file)

        # Open all necessary files
        imgs: IO = open(img_file, "rb")
        out: IO = open(out_file, "w")
        labels: IO = open(label_file, "rb")
        
        # Skip header bytes
        imgs.read(16)
        labels.read(8)
        
        # Create a 2D list of images where each image is a 1D list where the first element is the label
        img_area: int = img_size ** 2
        images: List[List[int]] = []

        for _ in range(data_size):
            image: List[int] = [int.from_bytes(labels.read(1), byteorder='big')]
            for _ in range(img_area):
                image.append(int.from_bytes(imgs.read(1), byteorder='big'))
            images.append(image)

        # Convert each image from 1D list to a comma-seperated str and write it into out file
        for image in images:
            out.write(",".join(str(pix) for pix in image) + "\n")
        
        # Close files
        imgs.close()
        out.close()
        labels.close()


























































































