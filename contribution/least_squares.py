import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.femnist import Femnist
from models.ridge_classifier import RidgeClassifier
from feature_extractors.vgg_19_bn import VGG_19_BN
from feature_extractors.rocket2d import Rocket2D
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import torch.nn as nn
import random
from sklearn.decomposition import PCA
import pyarrow.parquet as pq
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PCA_SIZE = 20_000

# before executing this file make sure that the parquet generating pipeline has been run, including the merge_parquet file

def train_least_squares(feature_extractor, classifier, directory, training_filename, testing_filename):
    training_frame = pd.read_parquet(os.path.join(directory,  training_filename))
    training_frame = training_frame[:int(0.05*len(training_frame))]
    training_frame.reset_index(inplace=True)
    dataset = Femnist(training_frame)

    pipeline = Pipeline([
        ('PCA', PCA(9000)),
        ('scaler', StandardScaler())
    ])
    dataloader = DataLoader(dataset, batch_size=9000)
    x, y = next(iter(dataloader))
    x = x.to(device=device)
    x = feature_extractor(x)
    pipeline.fit(x.to(device='cpu').numpy())

    dataloader = DataLoader(dataset, batch_size=128 if extractor == 'vgg' else 1024)

    progress = tqdm(total=len(dataloader), desc=training_filename)
    num_features = feature_extractor.number_of_features

    XTX = torch.zeros((num_features+1, num_features+1)).to(device=device)
    XTY = torch.zeros((num_features+1, 62)).to(device=device)
    for x, y in dataloader:
        y = y.to(device=device)
        Y = (F.one_hot(y, num_classes=62)*2)-1
        Y = Y.type(torch.float32)
        x = x.to(device=device)
        X = feature_extractor(x)
        print("X: ", torch.linalg.cond(X))
        print("XTX: ", torch.linalg.cond(X.T @ X))
        X = torch.tensor(pipeline(X.to(device='cpu').numpy()))
        print("pipe X: ", torch.linalg.cond(X))
        print("pipe XTX: ", torch.linalg.cond(X.T @ X))
        ones = torch.unsqueeze(torch.ones(X.shape[0]), 1).to(device=device)
        X = torch.cat((ones, X), dim=-1)
        print("ones X: ", torch.linalg.cond(X))
        print("ones XTX: ", torch.linalg.cond(X.T @ X))
        XTX += X.T @ X
        XTY += X.T @ Y
        #frame = pd.DataFrame(transformed_x.to(device='cpu'), columns=[f"x_{i}" for i in range(transformed_x.shape[1])])
        #transformed_x_list.append(frame)
        progress.update(1)
    progress.close()

    del dataloader
    del dataset

    alpha = 10
    #B = torch.linalg.inv(XTX + alpha * torch.eye(n=num_features+1).to(device=device)) @ XTY
    regularizedXTX = XTX + alpha * torch.eye(n=num_features+1).to(device=device)
    B = torch.linalg.solve(regularizedXTX, XTY)
    classifier.weight = nn.Parameter(B)

    #testing_frame = pd.read_parquet(os.path.join(directory,  testing_filename))
    #testing_frame = testing_frame[:int(0.1*len(training_frame))]
    #testing_frame.reset_index(inplace=True)
    testing_frame = training_frame
    dataset = Femnist(testing_frame)
    dataloader = DataLoader(dataset, batch_size=128 if extractor == 'vgg' else 1024)

    progress = tqdm(total=len(dataloader), desc=training_filename)
    classifier = classifier.to(device=device)

    training_loss, training_acc = 0, 0
    for x, y in dataloader:
        y = y.to(device=device)
        x = x.to(device=device)
        X = feature_extractor(x)
        X = torch.tensor(pipeline(X.to(device='cpu').numpy()))
        ones = torch.unsqueeze(torch.ones(X.shape[0]), 1).to(device=device)
        X = torch.cat((ones, X), dim=-1)

        _, _, loss, acc = classifier.evaluate(X, y)
        training_loss += loss
        training_acc += acc
        print(acc)
        #frame = pd.DataFrame(transformed_x.to(device='cpu'), columns=[f"x_{i}" for i in range(transformed_x.shape[1])])
        #transformed_x_list.append(frame)
        progress.update(1)
    training_loss /= len(dataloader)
    training_acc /= len(dataloader)

    print(training_acc)
    #print(training_loss)
    progress.close()

    del dataloader
    del dataset

    return


if __name__ == "__main__":
    
    directory = os.path.join("..", "data", "femnist", "compressed")
    extractors = ['vgg', 'rocket2d']

    for extractor in extractors:
        if extractor == 'vgg':
            feature_extractor = VGG_19_BN().to(device=device)
        elif extractor == 'rocket2d':
            feature_extractor = Rocket2D(h=28, w=28).to(device=device)
        else:
            raise NotImplementedError()
        
        classifier = RidgeClassifier(num_inputs=feature_extractor.number_of_features+1, num_classes=62)
        for t in ["iid", 'niid']:

            base_directory = os.path.join(directory, t)

            train_least_squares(feature_extractor,
                                classifier,
                                directory=base_directory, 
                                training_filename='training.parquet', 
                                testing_filename='testing.parquet')
            #transform_file(directory=base_directory, filename='testing.parquet')
