import argparse
from functools import partial
import numpy as np
import os
import random
import sys
import torch
from typing import Any
import wandb

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets.femnist as femnist
import models.ridge as ridge
import utils.algorithm as algorithm
import client
import server

def set_seed(seed: int, deterministic: bool = True):
    '''
    Initializes all random generators with same seed and may force the usage of 
    deterministc algorithms.

    Parameters
    ----------
    seed: int
        Seed number
    deterministc: bool
        Tells to (or not to) force usage of deterministic algorithms (True by default)
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

def initialize_learning_rate_scheduler(args: Any) -> torch.optim.lr_scheduler.LRScheduler:
    '''
    Constructs a scheduler decaying learning rate over multiple training rounds of clients.

    Parameters
    ----------
    args: Any
        Command line arguments of the simulation

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    '''
    
    scheduling = args.scheduler[0].lower()
    # dummy optimizer
    dummy = torch.optim.SGD([ torch.nn.Parameter(torch.zeros([])) ], lr = args.learning_rate)
    # constructs learning rate scheduler of right kind
    # decaying every central round of 'gamma' factor
    if scheduling == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(dummy, gamma = float(args.scheduler[1]))
    # decaying every 'step_size' central rounds of 'gamma' factor
    elif scheduling == 'step':
        return torch.optim.lr_scheduler.StepLR(dummy, step_size = int(args.scheduler[2]), gamma = float(args.scheduler[1]))
    # linear scheduling from 'start_factor' up to 'learning_rate' in 'total_iters' rounds
    elif scheduling == 'linear':
        return torch.optim.lr_scheduler.LinearLR(dummy, start_factor = float(args.scheduler[1]), total_iters = int(args.scheduler[2]))
    # default, no scheduling
    return torch.optim.lr_scheduler.LambdaLR(dummy, lambda round: 1)

def initialize_federated_algorithm(args: Any, num_features: int, num_classes: int, device: torch.device) -> algorithm.FedAlgorithm:
    '''
    Initializes a federated learning algorithm.

    Parameters
    ----------
    args: Any
        Command line simulation arguments

    Returns
    -------
    algorithm.FedAlgorithm
        Federated learning algorithm
    '''

    # FIXME: scheduler useless
    # scheduler = initialize_learning_rate_scheduler(args)
    # binds scheduler
    if not args.algorithm or args.algorithm[0] == 'fedlsq':
        return partial(
            algorithm.FedLeastSquares,
            num_features = num_features,
            num_classes = num_classes,
            lmbda = args.lmbda,
            device = device
        )
    # federated algorithm not recognized
    raise RuntimeError(f'unrecognized federated algorithm \'{args.algorithm[0]}\', expected \'fedlsq\'')

def get_arguments() -> Any:
    '''
    Parses command line arguments.

    Returns
    -------
    Any
        Command line arguments
    '''
    
    parser = argparse.ArgumentParser(
        usage = 'run experiment on baseline FEMNIST dataset (federated, so decentralized) with rocket 2d data transformation and ridge resolution',
        description = 'This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        formatter_class = argparse.RawDescriptionHelpFormatter
        # TODO: write epilog for R2D2
    )
    
    parser.add_argument('--lmbda', type = float, default = 1.0, help = 'regularization parameter of federated least squares')
    parser.add_argument('--training_fraction', type = float, default = 1.0, choices = [ .80, .90, .95, 1.0 ], help = 'fraction of clients used for training (left ones are for validation)')
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--dataset', type = str, choices = ['femnist_rocket2d', 'femnist_vgg', 'femnist_rocket2d_pca', 'femnist_vgg_pca'], default = 'femnist_rocket2d', help = 'dataset name')
    parser.add_argument('--niid', action = 'store_true', default = False, help = 'run the experiment with the non-IID partition (IID by default), only on FEMNIST dataset')
    parser.add_argument('--selected', type = int, default = None, help = 'number of clients trained per round')
    parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'learning rate decay scheduling, like \'step\' or \'exp\' or \'linear\'')
    parser.add_argument('--algorithm', metavar = ('algorithm', 'params'), nargs = '+', type = str, default = ['fedlsq'], help = 'federated learning algorithm, like \'fedlsq\'')
    parser.add_argument('--evaluation', type = int, default = None, help = 'evaluation interval of training and validation set')
    parser.add_argument('--evaluators', type = float, default = None, help = 'fraction (if < 1.0) or number (if >= 1) of clients to be evaluated from training and validation set')
    parser.add_argument('--testing', action = 'store_true', default = True, help = 'run final evaluation on unseen testing clients')
    parser.add_argument('--save', type = str, default = 'checkpoints', help = 'save state dict after training in path')
    parser.add_argument('--log', action = 'store_true', default = False, help = 'whether or not to log to weights & biases')

    return parser.parse_args()

if __name__ == '__main__':
    # parameters of the simulation
    args = get_arguments()
    # random seed and more importantly deterministic algorithm versions are set
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # federated datasets are shared among training and testing clients
    print('[+] loading datasets... ', end = '', flush = True)
    dataset_type = args.dataset.removeprefix('femnist_')
    datasets = femnist.load(
        directory = os.path.join(
            'data', 
            'femnist', 
            'compressed', 
            f'niid_{dataset_type}' if args.niid else f'iid_{dataset_type}'
        ),
        transformed = True,
        training_fraction = args.training_fraction
    )
    print('done')
    # client construction by dividing them in three groups (training, validation, testing)
    # each client of each group has its own private user dataset
    print('[+] constructing clients... ', end = '', flush = True)
    clients = client.construct(datasets, device, args)
    print('done')
    # simulation identifier
    identifier = f"r2d2_{'niid' if args.niid else 'iid'}_s{args.seed}_d{args.dataset}_a{':'.join(args.algorithm)}_c{args.selected}_lr{args.learning_rate}_lrs{':'.join(args.scheduler)}_bs{args.batch_size}"
    # server uses training clients and validation clients when fitting the central model
    # clients from testing group should be used at the very end
    num_features = datasets['training'][0][1].dataset.num_features
    num_classes = 62
    server = server.Server(
        algorithm = initialize_federated_algorithm(args, num_features = num_features, num_classes = num_classes, device = device), 
        model = ridge.RidgeRegression(num_features, num_classes, device = device),
        clients = clients,
        args = args,
        id = identifier
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] id: {identifier}')
    print(f'  [-] seed: {args.seed}')
    print(f'  [-] dataset: {args.dataset}')
    print(f"  [-] distribution: {'niid' if args.niid else 'iid'}")
    print(f'  [-] lambda: {args.lmbda}')
    print(f'  [-] batch size: {args.batch_size}')
    print(f'  [-] learning rate: {args.learning_rate}')
    print(f"  [-] learning rate scheduling: {' '.join(args.scheduler)}")
    print(f"  [-] federated algorithm: {' '.join(args.algorithm)}")
    print(f"  [-] clients selected: {args.selected if args.selected else len(clients['training'])}")
    print(f'  [-] fraction or number of clients evaluated: {args.evaluators if args.evaluators else 1.0}')
    print(f'  [-] final testing: {args.testing}')
    print(f'  [-] remote log enabled: {args.log}')
    # initialize configuration for weights & biases log
    wandb.init(
        mode = 'online' if args.log else 'disabled',
        project = 'federated',
        name = identifier,
        config = {
            'seed': args.seed,
            'dataset': args.dataset,
            'niid': args.niid,
            'selected': args.selected,
            'learning_rate': args.learning_rate,
            'scheduler': ':'.join(args.scheduler),
            'batch_size': args.batch_size,
            'algorithm': ':'.join(args.algorithm)
        }
    )
    # execute training and validation epochs
    server.run()
    # terminate weights & biases session by synchronizing
    wandb.finish()