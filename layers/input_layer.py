# ------------------------------
# BASIC IMPORTS
# ------------------------------
import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd
from layers.layer import NetworkLayer


'''
This class will define how the input dataset will be processed before feeding it to the network
'''
class InputLayer(NetworkLayer):

    def __init__(self, train_data, train_label, train_filename, test_data, test_label, test_filename, device_id):
        super().__init__(0, 0, device_id)

        # Set up training data
        self.train_data = train_data
        self.train_label = train_label
        self.train_filename = train_filename

        # Set up testing data
        self.test_data = test_data
        self.test_label = test_label
        self.test_filename = test_filename



    '''
    Class method used to convert .ubyte files(which is commonly used for MNIST) into a .csv file for ease of use
    
    @param
        cls = an instance of the InputLayer class
        img_file 
    @return
        ___ (void) = no returns

    '''
    @classmethod
    def convert(cls, img_file, label_file, out_file, data_size, img_size):

        # These lines construct the absolute paths for the image, label, and output files by combining the current working directory (project_root) with the relative paths provided as arguments.
        project_root = os.getcwd()
        img_file = os.path.join(project_root, img_file)
        label_file = os.path.join(project_root, label_file)
        out_file = os.path.join(project_root, out_file)

        # These lines:
            # 1. opens the image and label files in binary read mode
            # 2. open output file in write mode
        imgs = open(img_file, "rb")
        out = open(out_file, "w")
        labels = open(label_file, "rb")

        # Skip header bytes
            # The .ubyte files for MNIST contain header information. The image file header is 16 bytes, and the label file header is 8 bytes. These lines skip over the header bytes.
        imgs.read(16)
        labels.read(8)


        # Initialize image list
        img_size = img_size**2  # Here, the image size is squared to get the total number of pixels in each image.
        images = []

        
        # Read image and label data
        # Here, I am looping through each image
        for i in range(data_size):

            image = [int.from_bytes(labels.read(1), byteorder='big')]   # For each image, the corresponding label is read as a single byte
                                                                        # then, this is converted to an integer and stored
            
            # Here, I am looping through each pixel of each image
            for j in range(img_size):

                image.append(int.from_bytes(imgs.read(1), byteorder='big'))     # For this, each pixel value of the image is read, one byte at a time
                                                                                # then, this is appended to the image list 
            # Notice -> the 'image' is a list storing the label at the first index and the pixels for the rest of the bytes

            images.append(image) # Lastly, I am appending the current image to the images list


        # WRITE TO CSV FILE!
        for image in images:
            out.write(",".join(str(pix) for pix in image) + "\n")
        # This loop basically converts each image(which is now a list of integers) INTO
        # a comma-separated string and writes it to the output file

        # Close all opened files
        imgs.close()
        out.close()
        labels.close()
        












    def setup_train_data(self):







































































































