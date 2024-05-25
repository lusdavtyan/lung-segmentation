import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import paths


def save_model_checkpoint(model, model_path, cp_name):
    """
    Saves the state of a PyTorch model to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        model_path (str): The directory to save the model in.
        cp_name (str): The name of the checkpoint file.
    """
    torch.save(model.state_dict(), os.path.join(model_path, cp_name))


def get_device():
    """
    Returns the device to use for computations (either a GPU if one is available, or the CPU).

    Returns:
        torch.device: The device to use for computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(x):
    """
    Moves a tensor to the device that is used for computations.

    Args:
        x (torch.Tensor): The tensor to move.

    Returns:
        torch.Tensor: The tensor on the device used for computations.
    """
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def plot_histogram(image, log=False):
    """
    Plots a histogram of the intensities in an image.

    Args:
        image (np.array): The image to plot a histogram for.
        log (bool, optional): Whether to use a logarithmic scale. Defaults to False.
    """
    if log:
        plt.hist(image.flatten(), bins=80, color='c', log=True)
    else:
        plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Intensities")
    plt.ylabel("Frequency")
    plt.show()

def visualize_data():
    """
    Visualizes CT scan data and corresponding labels.

    This function generates a figure with subplots to display CT scan slices and their corresponding labels.
    It randomly selects a folder and a slice index from the provided PATH_LIST and retrieves the input data
    and label data using the `get_input_and_label` function. The input data is displayed in the first column,
    the background label is displayed in the second column, and the lungs label is displayed in the third column.

    Returns:
        None

    """
    fig, ax = plt.subplots(5, 2, figsize=(3, 7))
    for i in range(5):
        folder_index = random.randint(0, len(paths.PATH_LIST) - 1)
        slice_index = random.randint(0, len(paths.PATH_LIST[folder_index]) - 1)
        input_data, label_data = get_input_and_label(paths.PATH_LIST, folder_index, slice_index)

        ax[i, 0].imshow(input_data)
        ax[i, 1].imshow(np.where(label_data == 1, 1, 0))

        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    ax[0, 0].set_title('CT Slice')
    ax[0, 1].set_title('Lungs')
    plt.show()

def get_volume_sizes(path_list):
    """
    Returns the sizes of the volumes in a list of paths.

    Args:
        path_list (list): The list of paths to the volumes.

    Returns:
        list: The sizes of the volumes.
    """
    sizes = []
    for l in path_list:
        sizes.append(len(l))
    return sizes


def get_img(path):
    """
    Loads an image from a file and returns it as a numpy array.

    Args:
        path (str): The path to the image file.

    Returns:
        np.array: The loaded image.
    """
    image = Image.open(path)
    image = np.array(image)
    image = image[:, :, 0:1]
    return image


def get_input_and_label(path_list, volume_index, slice_index):
    """
    Returns the input and label data for a specific slice of a specific volume.

    Args:
        path_list (list): The list of paths to the volumes.
        volume_index (int): The index of the volume.
        slice_index (int): The index of the slice.

    Returns:
        tuple: The input and label data for the specified slice of the specified volume.
    """
    return get_img(path_list[volume_index][slice_index][0]), get_img(path_list[volume_index][slice_index][1]).astype(bool)

    
class CTScansDataset(Dataset):
    """
    A custom dataset class for CT scans.

    Args:
        volume_sizes (list): A list of volume sizes for each scan.
        transform (callable, optional): A function/transform to apply to the image and label.
        mode (str, optional): The mode of the dataset. Can be 'train', 'valid', or 'test'.

    Attributes:
        transform (callable): A function/transform to apply to the image and label.
        mode (str): The mode of the dataset.
        volume_sizes (list): A list of volume sizes for each scan.
        data_indices (list): A list of indices corresponding to the selected data.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Returns the image and label at the given index.

    """

    def __init__(self, volume_sizes, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.volume_sizes = volume_sizes

        if self.mode == 'train':
            self.data_indices = [i for i in range(14)]
        elif self.mode == 'valid':
            self.data_indices = [i for i in range(14, 17)]
        elif self.mode == 'test':
            self.data_indices = [i for i in range(17, 20)]

        self.volume_sizes = [self.volume_sizes[i] for i in self.data_indices]
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.

        """
        return sum(self.volume_sizes)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.

        """
        image_i = 0
        size_sum = 0
        for i in range(len(self.data_indices)):
            if idx < size_sum + self.volume_sizes[i]:
                break
            size_sum += self.volume_sizes[i]
            image_i += 1

        image = Image.open(paths.PATH_LIST[self.data_indices[image_i]][idx - size_sum][0])
        label = Image.open(paths.PATH_LIST[self.data_indices[image_i]][idx - size_sum][1])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image = image[0:1, :, :]
        label = label[0:1, :, :]
        label = label.bool()
        return image, label