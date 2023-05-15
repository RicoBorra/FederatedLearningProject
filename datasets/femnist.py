import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import os

class Femnist(Dataset):
    '''
    Original femnist dataset with 62 classes. It can be either iid or niid,
    depending on the simulation arguments. Normally each client has a `Subset`
    of a `Femnist` dataset instance associated to its user images and labels.
    '''

    def __init__(self, frame: pd.DataFrame, normalize: bool = True):
        '''
        Initializes a whole femnist dataset from a dataframe.

        Parameters
        ----------
        frame: pd.DataFrame
            Pandas dataframe
        normalize: bool = True
            Whether to normalize (standardize) images (True by default)
        
        Notes
        -----
        Dataframe columns should be as follows.
        * frame.iloc[:, 0] should contain the user identifier
        * frame.iloc[:, 1] should contain the class label 1...62
        * frame.iloc[:, 2:786] should contain pixel features
        '''

        self.frame = frame
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]) if normalize else transforms.ToTensor()


    def __len__(self) -> int:
        '''
        Length of dataset.

        Returns
        -------
        int
            Number of samples within the dataset
        '''
        
        return self.frame.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        '''
        Gets image and label at `index`.

        Parameters
        ----------
        index: int
            Image index

        tuple[torch.Tensor, int]
            Image and label
        '''

        x, y = self.frame.iloc[index, 2:786].values, self.frame.iloc[index, 1]
        # reshape images in 2D as [WIDTH, HEIGHT, CHANNELS]
        x = np.reshape(x, newshape = (28, 28, 1)).astype(np.float32)
        # apply transformation, maybe a standardization, and put the
        # number of channels as first dimension
        if self.transform is not None:
            x = self.transform(x)
        # image and label
        return x, y

def load(directory: str) -> dict[str, list[tuple[str, Subset]]]:
    '''
    Loads local clients `Femnist` datasets divided into three macro groups,
    which are `training`, `validation` and `testing`.

    Parameters
    ----------
    directory: str
        Directory from which the dataframes `training.parquet` and `testing.parquet` are loaded

    Returns
    -------
    dict[str, list[tuple[str, Subset]]]
        Dictonary with groups of clients' datasets

    Notes
    -----
    Each entry of the dictionary is the list of subsets within the selected group,
    specifically each dataset of the list is a tuple of the client name and its subset 
    of data from the original `Femnist` dataset of the same group.
    '''

    # whole datasets are stored in a compressed and efficient format
    training_frame = pd.read_parquet(os.path.join(directory, 'training.parquet'))
    testing_frame = pd.read_parquet(os.path.join(directory, 'testing.parquet'))
    # this operation ensures to have unique indices for each entry and not the 
    # 'user' column
    training_frame.reset_index(inplace = True)
    testing_frame.reset_index(inplace = True)
    # builds whole datasets and three groups of clients (and corresponding subsets)
    training_data = Femnist(training_frame)
    testing_data = Femnist(testing_frame)
    user_datasets = { 'training': [], 'validation': [], 'testing': [] }
    # first group is 'training' for clients on whose datasets training of central
    # model is performed
    for name, group in training_frame.groupby('user'):
        subset = Subset(training_data, group.index.values)
        user_datasets['training'].append((name, subset))
    # sample 20% of 'training' clients datasets and move them to 'validation' group
    training_users_count = math.floor(0.8 * len(user_datasets['training']))
    user_datasets['validation'] = user_datasets['training'][training_users_count:]
    user_datasets['training'] = user_datasets['training'][:training_users_count]
    # construct 'testing' group of clients datasets
    for name, group in testing_frame.groupby('user'):
        subset = Subset(testing_data, group.index.values)
        user_datasets['testing'].append((name, subset))
    # three groups of clients datasets
    return user_datasets
