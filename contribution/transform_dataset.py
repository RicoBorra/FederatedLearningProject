import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.femnist import Femnist
from feature_extractors.vgg_19_bn import VGG_19_BN
from feature_extractors.rocket2d import Rocket2D
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pyarrow.parquet as pq
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PCA_SIZE = 20_000


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

# before executing this file make sure that the parquet generating pipeline has been run, including the merge_parquet file

def get_transformed(directory, filename, pca=None):
    training_frame = pd.read_parquet(os.path.join(directory,  filename))
    training_frame.reset_index(inplace=True)
    print(training_frame)
    print(training_frame.shape)
    if pca is not None:
        dataset = Transformed_Femnist(training_frame)
    else:
        dataset = Femnist(training_frame)
    dataloader = DataLoader(dataset, batch_size=128 if extractor == 'vgg' else 1024)

    
    if filename == "training.parquet" and pca is not None:
        print("PCA training... ", end="")
        data_for_pca = random.choices(dataset, k=PCA_SIZE)
        pca_dataloader = DataLoader(data_for_pca, batch_size=PCA_SIZE)
        for x, y in pca_dataloader:
            pca.fit(x.numpy())
        print("done")

    progress = tqdm(total=len(dataloader), desc=filename)
    transformed_x_list = []
    for x, y in dataloader:
        if pca is not None:
            transformed_x = pca.transform(x.numpy())
            frame = pd.DataFrame(transformed_x, columns=[f"x_{i}" for i in range(transformed_x.shape[1])])
        else:
            x = x.to(device=device)
            transformed_x = model(x)
            frame = pd.DataFrame(transformed_x.to(device='cpu'), columns=[f"x_{i}" for i in range(transformed_x.shape[1])])
        transformed_x_list.append(frame)
        progress.update(1)
    progress.close()

    del dataloader
    del dataset
    training_frame = training_frame[['user', 'y']]

    return transformed_x_list, training_frame

def transform_file(directory, out_dir, filename, pca=None):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, filename)
    if not os.path.exists(out_file):
        transformed_x_list, training_frame = get_transformed(directory, filename, pca)
        transformed_frame = pd.concat(transformed_x_list, axis=0)
        transformed_frame.reset_index(drop=True, inplace=True)
        transformed_frame.insert(loc=0, column='user', value=training_frame['user'])
        transformed_frame.insert(loc=1, column='y', value=training_frame['y'])
        transformed_frame.set_index('user').to_parquet(out_file)
        print(transformed_frame)

if __name__ == "__main__":
    
    directory = os.path.join("..", "data", "femnist", "compressed")
    extractors = ['vgg', 'rocket2d']

    for extractor in extractors:
        if extractor == 'vgg':
            model = VGG_19_BN().to(device=device)
        elif extractor == 'rocket2d':
            model = Rocket2D(h=28, w=28).to(device=device)
        else:
            raise NotImplementedError()
        
        for t in ["iid", 'niid']:

            base_directory = os.path.join(directory, t)

            out_dir_extr = os.path.join(directory, f"{t}_{extractor}")

            transform_file(directory=base_directory, out_dir=out_dir_extr, filename='training.parquet')
            transform_file(directory=base_directory, out_dir=out_dir_extr, filename='testing.parquet')

            # with PCA
            out_dir_pca = os.path.join(directory, f"{t}_{extractor}_pca")
            pca = Pipeline([
                ('pca', PCA(int(0.9 * model.number_of_features))),
                ('scaler', StandardScaler())
            ])

            transform_file(directory=out_dir_extr, out_dir=out_dir_pca, filename='training.parquet', pca=pca)
            transform_file(directory=out_dir_extr, out_dir=out_dir_pca, filename='testing.parquet', pca=pca)
