import math
import pandas as pd
import numpy as np
from typing import Sequence
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as transforms
import os

class Femnist(Dataset):
    '''
    Original femnist dataset with 62 classes. It can be either iid or niid,
    depending on the simulation arguments. Normally each client has a `Subset`
    of a `Femnist` dataset instance associated to its user images and labels.
    '''

    def __init__(self, frame: pd.DataFrame, normalize: bool = True, dtype=torch.float32):
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
        self.dtype = dtype

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
        return x.type(dtype=self.dtype), y
    
class Transformed_Femnist(Dataset):
    def __init__(self, frame: pd.DataFrame):

        self.frame = frame
        self.num_features = frame.shape[1] - 2


    def __len__(self) -> int:

        return self.frame.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:

        x, y = self.frame.iloc[index, 2:].values, self.frame.iloc[index, 1]
        
        x = torch.Tensor(x.astype(np.float32))
        # image and label
        return x, y
    
class RotatedFemnistSubset(Subset):
    '''
    This class represents a single client subset of images and labels subject to
    a counter clockwise rotation of some angle.
    '''

    def __init__(self, dataset: Dataset, indices: Sequence, angle: int, normalize: bool = True):
        '''
        Constructs a client subset whose images are rotated counter clockwise of `angle`.
        
        Parameters
        ----------
        dataset: Dataset
            Original femnist dataset
        indices: Sequence
            Indices of original dataset corresponding to current client
        angle: int
            Rotation angle for this client
        normalize: bool = True
            Whether to normalize (standardize) images (True by default)
        '''
        
        super().__init__(dataset, indices)
        # rotation angle
        self.angle = angle
        # apply transformation (rotation) and fills around spaces with whites
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, angle = float(self.angle), fill = 255)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]) if normalize else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, angle = float(self.angle), fill = 255)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        '''
        Selects a rotated image and its label.

        Parameters
        ----------
        index: int
            Index of retrieved image

        Returns
        -------
        tuple[torch.Tensor, int]
            Rotated image and its label
        '''

        import matplotlib.pyplot as plt
        
        # calls subset super class
        x, y = super().__getitem__(index)
        # applies rotation
        x = self.transform(x)
        # image and label
        return x, y

def load(directory: str, transformed: bool = False, training_fraction = 0.80) -> dict[str, list[tuple[str, Subset]]]:
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
    training_data = Femnist(training_frame) if not transformed else Transformed_Femnist(training_frame)
    testing_data = Femnist(testing_frame) if not transformed else Transformed_Femnist(testing_frame)
    user_datasets = { 'training': [], 'validation': [], 'testing': [] }
    # first group is 'training' for clients on whose datasets training of central
    # model is performed
    for name, group in training_frame.groupby('user'):
        subset = Subset(training_data, group.index.values)
        user_datasets['training'].append((name, subset))
    # sample 20% of 'training' clients datasets and move them to 'validation' group
    training_users_count = math.floor(training_fraction * len(user_datasets['training']))
    user_datasets['validation'] = user_datasets['training'][training_users_count:]
    user_datasets['training'] = user_datasets['training'][:training_users_count]
    # construct 'testing' group of clients datasets
    for name, group in testing_frame.groupby('user'):
        subset = Subset(testing_data, group.index.values)
        user_datasets['testing'].append((name, subset))
    # three groups of clients datasets
    return user_datasets

def load_with_rotated_domains(
        directory: str, 
        n_rotated: int, 
        angles: Sequence[int], 
        validation_domain_angle: int = None
) -> dict[str, list[tuple[str, Subset]]]:
    '''
    Loads local clients `Femnist` datasets divided into three macro groups,
    which are `training`, `validation` and `testing`. This time `n_rotated`
    random clients are divided in `len(angles)` groups and accordingly their
    images are counter clockwise rotated. When `validation_domain_angle` is
    not null, then validation clients are uniquely those rotated with such
    an angle.

    Parameters
    ----------
    directory: str
        Directory from which the dataframes `training.parquet` and `testing.parquet` are loaded
    n_rotated: int
        Number of clients rotated (in total)
    angles: Sequence[int]
        Sequence of rotation angles applied to `n_rotated` random clients, propely divided
    validation_domain_angle: int
        Rotation angle for clients whose group is employed for validation (none by default)
    

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
    training_data = Femnist(training_frame, normalize = False)
    testing_data = Femnist(testing_frame, normalize = False)
    user_datasets = { 'training': [], 'validation': [], 'testing': [] }
    # clients sampled for being rotated
    rotated_clients_names = np.random.choice(training_frame['user'].unique(), size = n_rotated, replace = False)
    # rotated clients angles is a map client -> angle
    rotated_clients_angles = {
        rotated_client: angle
        for angle, rotated_clients in zip(angles, np.array_split(rotated_clients_names, len(angles)))
        for rotated_client in rotated_clients
    }
    # checks if one domain is exclusively employed for validation instead of a 80:20 split
    if validation_domain_angle is None:
        # first group is 'training' for clients on whose datasets training of central model is performed
        for name, group in training_frame.groupby('user'):
            if name in rotated_clients_angles:
                # client belongs to a rotated domain
                subset = RotatedFemnistSubset(training_data, group.index.values, angle = rotated_clients_angles[name], normalize = True)
            else:
                # client has not been selected for rotation
                subset = Subset(training_data, group.index.values)
            # rotated or not, anyway client is added to training clients group
            user_datasets['training'].append((name, subset))
        # sample 20% of 'training' clients datasets and move them to 'validation' group
        training_users_count = math.floor(0.8 * len(user_datasets['training']))
        user_datasets['validation'] = user_datasets['training'][training_users_count:]
        user_datasets['training'] = user_datasets['training'][:training_users_count]
    else:
        # whenever evaluation domain angle exists, then only that domain is employed for validation
        # instead of a simple 80:20 division among training and validation clients
        for name, group in training_frame.groupby('user'):
            if name in rotated_clients_angles:
                # angle is the rotation angle for this specifc client
                angle = rotated_clients_angles[name]
                # rotated subset
                subset = RotatedFemnistSubset(training_data, group.index.values, angle = angle, normalize = True)
                # if the rotation anfle of this client corresponds to the validation domain angle
                # then this client is moved to validation set instead of training set
                if angle == validation_domain_angle:
                    user_datasets['validation'].append((name, subset))
                else:
                    user_datasets['training'].append((name, subset))
            else:
                # client has not been selected for rotation
                subset = Subset(training_data, group.index.values)
                user_datasets['training'].append((name, subset))
    # construct 'testing' group of clients datasets
    for name, group in testing_frame.groupby('user'):
        subset = Subset(testing_data, group.index.values)
        user_datasets['testing'].append((name, subset))
    # three groups of clients datasets
    return user_datasets

def centralized(
        clients_groups: dict[str, list[tuple[str, Subset]]]
) -> dict[str, Dataset]:
    '''
    Constructs centralized `training`, `validation` and `testing`
    datasets from multiple federated clients, each one beloging
    to one of the previously mentioned groups. 

    Parameters
    ----------
    clients_groups: dict[str, list[tuple[str, Subset]]]
        Groups of clients' datasets

    Returns
    -------
    dict[str, Dataset]
        Dictionary with groups of centralized datasets
    '''

    return {
        # concatenate all clients' datasets from one group to centralize
        # 'training', 'validation' or 'testing' dataset
        group: ConcatDataset([ dataset for _, dataset in clients ])
        for group, clients in clients_groups.items()
    }