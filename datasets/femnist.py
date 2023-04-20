import numpy as np
import os
import pandas as pd
import torch
import torchvision
from typing import List, Tuple

import datasets.np_transforms as tr


class Femnist(torch.utils.data.Dataset):
    '''
    This class holds a portion of the femnist dataset held by a specific client on the network.
    '''
    
    def __init__(self, client_name: str, frame: pd.DataFrame, transform: tr.Compose):
        '''
        Initializes the shared portion of the dataset.
        
        Parameters
        ----------
        client_name: str
            Name of user, namely the transmitting client
        frame: pd.DataFrame
            Dataframe holding the images and corresponding labels
        transform: tr.Compose
            Transformation to be applied on features
        '''
        
        super().__init__()

        self.client_name = client_name
        self.frame = frame
        self.transform = transform

    def __len__(self) -> int:
        return self.frame.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[np.array, int]:
        image, label = self.frame.iloc[index, 1:785].values, self.frame.iloc[index, 0]
        image = np.reshape(image, newshape = (28, 28, 1))

        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_femnist(
        directory: str, 
        transforms: Tuple[torchvision.transforms.Compose, torchvision.transforms.Compose],
        as_csv: bool = False
) -> Tuple[List[Femnist], List[Femnist]]:
    '''
    This loads both training and testing set from specified directory.

    Parameters
    ----------
    directory: str
        Directory where to read dataset files
    transforms: Tuple[torchvision.transforms.Compose, torchvision.transforms.Compose]
        Transformations to be applied on both datasets' features
    as_csv: bool
        Enable parsing CSV file instead of PARQUET (False by default)

    Notes
    -----
    Files are expected to be 'training.parquet' and 'testing.parquet' in case
    of compressed format, otherwise 'training.csv' and 'testing.csv'. PARQUET
    files are extremely efficient in terms of storage and speed (up to ~80 times).
    '''

    def load_femnist_dataset(
            path: str, 
            transform: torchvision.transforms.Compose, 
            as_csv: bool = False
    ) -> List[Femnist]:
        # loads by default parquet file
        frame = pd.read_csv(path, index_col = 0) if as_csv else pd.read_parquet(path)
        # construct dataset portions assigned to many users
        return [
            Femnist(client, group, transform)
            for client, group in frame.groupby('user')
        ]
    
    return (
        load_femnist_dataset(os.path.join(directory, 'training.parquet'), transforms[0], as_csv),
        load_femnist_dataset(os.path.join(directory, 'testing.parquet'), transforms[1], as_csv)
    )
