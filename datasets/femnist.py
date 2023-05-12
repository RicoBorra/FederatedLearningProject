import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from typing import Any, List, Tuple, Union

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
    
    def split(self, split_size: float, shuffle: bool = True, stratified: bool = False) -> Tuple[Any, Any]:
        if stratified:
            return train_test_split(self.frame, train_size = split_size, shuffle = shuffle, stratify = self.frame.iloc[:, 0])
        else:
            return train_test_split(self.frame, train_size = split_size, shuffle = shuffle)
        

def load_femnist(
        directory: str, 
        transforms: Tuple[torchvision.transforms.Compose, torchvision.transforms.Compose],
        as_csv: bool = False
) -> Tuple[List[Femnist], List[Femnist], List[Femnist], Femnist, Femnist, Femnist]:
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
    merged: bool
        If true then returns two datasets, otherwise splits them across clients (false is default)

    Returns
    -------
    Tuple[List[Femnist], List[Femnist], Femnist, Femnist]
        Returns list of train clients, validation clients, test clients, train set, validation set and test set

    Notes
    -----
    Files are expected to be 'training.parquet' and 'testing.parquet' in case
    of compressed format, otherwise 'training.csv' and 'testing.csv'. PARQUET
    files are extremely efficient in terms of storage and speed (up to ~80 times).
    '''

    def load_femnist_dataset(
            path: str, 
            transform: torchvision.transforms.Compose, 
            as_csv: bool = False,
            split: bool = False
    ) -> Union[Tuple[List[Femnist], Femnist, List[Femnist], Femnist], Tuple[List[Femnist], Femnist]]:
        # loads by default parquet file
        frame = pd.read_csv(path, index_col = 0) if as_csv else pd.read_parquet(path)
        # split is used to denote validation split
        if split:
            training_data, validation_data = Femnist('*', frame, transform).split(split_size = 0.80, shuffle = False, stratified = False)
            return (
                # training sets spread across training clients
                [
                    Femnist(client, group, transform)
                    for client, group in training_data.groupby('user')
                ], 
                # whole training data
                Femnist('train', training_data, transform),
                # validation sets spread across validation clients
                [
                    Femnist(client, group, transform)
                    for client, group in validation_data.groupby('user')
                ], 
                # whole validation data
                Femnist('validation', validation_data, transform)
            )

        else:
            # construct dataset portions assigned to many users
            return [
                Femnist(client, group, transform)
                for client, group in frame.groupby('user')
            ], Femnist('*', frame, transform)
    
    training_clients, training_data, validation_clients, validation_data = load_femnist_dataset(os.path.join(directory, 'training.parquet'), transforms[0], as_csv, split = True)
    testing_clients, testing_data = load_femnist_dataset(os.path.join(directory, 'testing.parquet'), transforms[1], as_csv)
    return training_clients, validation_clients, testing_clients, training_data, validation_data, testing_data
