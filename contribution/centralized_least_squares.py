import argparse
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import tqdm

import wandb

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractors.rocket2d import Rocket2D
from feature_extractors.vgg_19_bn import VGG_19_BN
from models.ridge_classifier import RidgeClassifier
from models import cnn
from utils.reduction import MeanReduction
import datasets.femnist as femnist
 
def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_arguments():
    epilog = "note: argument \'--scheduler\' accept different learning rate scheduling choices (\'exp\' or \'step\') followed by decaying factor and decaying period\n\n" \
        "examples:\n\n" \
        ">>> python3 experiments/centralized_baseline.py --batch_size 256 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --epochs 10 --scheduler exp 0.5\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 256\n" \
        " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5\n" \
        " [+] SGD momentum: 0.9\n" \
        " [+] SGD weight decay penalty: 0.0001\n" \
        " [+] running epoch for training and validation: 10\n\n" \
        ">>> python3 experiments/centralized_baseline.py --batch_size 512 --learning_rate 0.01 --epochs 100 --scheduler step 0.75 3\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 epochs\n" \
        " [+] running epoch for training and validation: 100\n\n" \
        ">>> python3 experiments/centralized_baseline.py --batch_size 512 --learning_rate 0.01 --epochs 100 --scheduler linear 0.1 8\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 is starting factor of linear growth for 8 epochs\n" \
        " [+] running epoch for training and validation: 100\n\n" \
        ">>> python3 experiments/centralized_baseline.py --batch_size 512 --learning_rate 0.01 --epochs 100 --scheduler onecycle 0.1\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 with one cycle cosine annealing rising up to a peak of 0.1 and then decreasing\n" \
        " [+] running epoch for training and validation: 100"
    parser = argparse.ArgumentParser(
        usage = 'run experiment on baseline EMNIST dataset (centralized) with a CNN architecture',
        description = 'This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        epilog = epilog,
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    #parser.add_argument('--epochs', default = 1, type = int, help = 'number of local epochs')
    #parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--model', type = str, default = 'rocket2d', help = 'model')
    parser.add_argument('--weight_decay', type = float, default = 1, help = 'weight decay')
    #parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    #parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'learning rate decay scheduling')
    #parser.add_argument('--kfold', type=int, default=None, help='kfold cross validation')
    parser.add_argument('--log', action='store_true', default=False, help='whether or not to log to weights & biases')
    return parser.parse_args()


def load_emnist(batch_size: int):
    return (
        DataLoader(
            datasets.EMNIST(root = 'data/emnist', train = True, download = True, split = 'byclass', transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])),
            batch_size = batch_size
        ),
        DataLoader(
            datasets.EMNIST(root = 'data/emnist', train = False, download = True, split = 'byclass', transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])),
            batch_size = batch_size
        )
    )

def load_model(args, n_batches: int, model_type: str = 'rocket2d'):
    if model_type == 'rocket2d':
        feature_extractor = Rocket2D()
    else:
        feature_extractor = VGG_19_BN()
    ridge = RidgeClassifier(num_inputs=feature_extractor.number_of_features, num_classes=62)

    
    # yields built model
    return feature_extractor, ridge

def run(feature_extractor: torch.nn.Module, ridge: nn.Module, training_loader: DataLoader, validation_loader: DataLoader, num_classes: int = 62, alpha: int = 1) -> tuple[float, float]:
    '''
    ...

    Returns
    -------
    tuple[float, float]
        Validation loss and score at last epoch
    '''

    # resulting values from fold validation
    validation_loss, validation_score = None, None
    # running device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move model to device
    feature_extractor = feature_extractor.to(device)
    ridge = ridge.to(device)
    # log
    wandb.watch(ridge, log = 'all')
    wandb.define_metric('epoch')
    wandb.define_metric('loss/training', step_metric = 'epoch')
    wandb.define_metric('loss/validation', step_metric = 'epoch')
    wandb.define_metric('accuracy/training', step_metric = 'epoch')
    wandb.define_metric('accuracy/validation', step_metric = 'epoch')
    # compile if possible
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    print(f'[+] running on device \'{device}\'')
    
    training_progress = tqdm.tqdm(total = len(training_loader), desc = '  [-] training')
    validation_progress = tqdm.tqdm(total = len(validation_loader), desc = '  [-] validation')
    # metrics
    training_loss, training_score = 0, 0
    validation_loss, validation_score = 0, 0
    # training
    # train()
    # train loop

    # for now no intercept
    num_features = feature_extractor.number_of_features
    XTX = torch.zeros((num_features, num_features))
    XTY = torch.zeros((num_features, num_classes))
    for X, y in training_loader:
        X, y = X.to(device), y.to(device)
        Y = (F.one_hot(y, num_classes=num_classes)*2)-1
        Y = Y.type(torch.float32)
        #print("before extracting features")
        X = feature_extractor(X)
        #print("Before XTX... ", end="")
        #XTX += X.T @ X
        #print("done")
        #print("before XTY... ", end="")
        #XTY += X.T @ Y
        #print("done")
        #training_batch_logits, training_batch_loss = model.step(X, y, model.optimizer)
        #training_batch_score = (training_batch_logits.argmax(1) == y).sum().item()
        #training_loss, training_score = training_loss + training_batch_loss, training_score + training_batch_score
        training_progress.update(1)

    B = torch.linalg.inv(XTX + alpha * torch.eye(n=num_features)) @ XTY
    ridge.weight = nn.Parameter(B)
    # enable validation mode
    
    with torch.no_grad():
        for X, y in validation_loader:
            X, y = X.to(device), y.to(device)
            _, validation_batch_predicted, validation_batch_loss, _ = ridge.evaluate(feature_extractor(X), y)
            validation_batch_score = (validation_batch_predicted == y).sum().item()
            validation_loss, validation_score = validation_loss + validation_batch_loss, validation_score + validation_batch_score
            validation_progress.update(1)
    # execute scheduler for learning rate
    # stop progress bars
    training_progress.close()
    validation_progress.close()
    # adjust metrics
    training_loss, training_score = training_loss / len(training_loader), training_score / len(training_loader.dataset)
    validation_loss, validation_score = validation_loss / len(validation_loader), validation_score / len(validation_loader.dataset)
    # remote log
    wandb.log({
        'epoch': 1,
        'loss/training': training_loss,
        'loss/validation': validation_loss,
        'accuracy/training': training_score,
        'accuracy/validation': validation_score
    })
    # log metrics
    print(f'  [-] loss: {training_loss:.3f}, score: {training_score:.3f} (training)')
    print(f'  [-] loss: {validation_loss:.3f}, score: {validation_score:.3f} (validation)')
    
    # these values are returned to represent fold validation
    return validation_loss, validation_score

if __name__ == '__main__':
    # command line arguments
    args = get_arguments()
    # set seed
    #set_seed(args.seed)
    # simulation identifier
    identifier = f"leastsquares_emnist_s{args.seed}_{args.model}_bs{args.batch_size}_wd{args.weight_decay}"
    # log
    wandb.init(
        mode = 'online' if args.log else 'disabled',
        project = 'centralized',
        name = identifier,
        config = {
            'seed': args.seed,
            'dataset': 'emnist',
            'model': 'rocket2d',
            #'epochs': args.epochs,
            #'learning_rate': args.learning_rate,
            #'scheduler': ':'.join(args.scheduler),
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay
            #'momentum': args.momentum
        }
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] id: {identifier}')
    print(f'  [-] seed: {args.seed}')
    print(f'  [-] batch size: {args.batch_size}')
    #print(f'  [-] learning rate: {args.learning_rate}')
    #print(f'  [-] momentum: {args.momentum}')
    print(f'  [-] weight decay L2: {args.weight_decay}')
    #print(f'  [-] epochs: {args.epochs}')
    #print(f"  [-] learning rate scheduling: {' ' .join(args.scheduler)}")
    print(f'  [-] model: {args.model}')
    print(f'  [-] remote log enabled: {args.log}')
    
    # data loaders
    #training_loader, validation_loader = load_emnist(batch_size = args.batch_size)
    training_frame = pd.read_parquet("../data/femnist/compressed/iid/training.parquet")
    testing_frame = pd.read_parquet("../data/femnist/compressed/iid/testing.parquet")
    training_frame.reset_index(inplace = True)
    testing_frame.reset_index(inplace = True)

    training_loader = DataLoader(femnist.Femnist(training_frame), batch_size=args.batch_size)
    validation_loader = DataLoader(femnist.Femnist(testing_frame), batch_size=args.batch_size)
    # load model
    feature_extractor, ridge = load_model(args, n_batches = len(training_loader))
    # train and validate model as expeced
    run(feature_extractor, ridge, training_loader, validation_loader, num_classes = 62, alpha = args.weight_decay)
    # terminate weights & biases session by sincynchronizing
    wandb.finish()
