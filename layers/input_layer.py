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

    # Static method to log bad lines
    @staticmethod
    def log_bad_lines(line, line_number):
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(log_dir, "bad_lines.log")
        with open(log_file_path, "a") as log_file:                  # Open log file in append mode
            log_file.write(f"Line {line_number}: {line}\n")         # Write bad line and its line number





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
        

    # Combined function to setup the dataset
    def setup_data(self, dataset_type='train'):
        if dataset_type == 'train':
            filename = self.train_filename
            data_file = self.train_data
            label_file = self.train_label
            data_size = 60000
        else:
            filename = self.test_filename
            data_file = self.test_data
            label_file = self.test_label
            data_size = 10000

        if not os.path.exists(filename):
            InputLayer.convert(data_file, label_file, filename, data_size, 28)
        
        data_frame = pd.read_csv(filename, header=None, on_bad_lines=InputLayer.log_bad_lines if dataset_type == 'train' else 'skip', engine='python')
        labels = torch.tensor(data_frame[0].values)
        data_frame = torch.tensor(data_frame.drop(data_frame.columns[0], axis=1).values, dtype=torch.float)
        data_frame /= 255
        return TensorDataset(data_frame, labels)



    # List of all methods from layers.NetworkLayer that are not needed for this layer
    # TODO: find a better way to implement the logic of having an input processing layer that still extends the layer.NetworkLayer interface
    def create_id_tensors(self): pass
    def set_scheduler(self): pass
    def visualize_weights(self, path, num, use): pass
    def update_weights(self): pass
    def update_bias(self): pass
    def forward(self): pass
    def active_weights(self): pass
    def _train_forward(self, x, clamped_output): pass
    def _eval_forward(self, x, clamped_output): pass








































































































