import argparse
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
import utils.algorithm as algorithm
import utils.reduction as reduction
from models.cnn import CNN
from models.logistic_regression import LogisticRegression
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

def initialize_model(args: Any) -> torch.nn.Module:
    '''
    Initializes the central model to be trained.

    Parameters
    ----------
    args: Any
        Command line arguments

    Returns
    -------
    torch.nn.Module
        Model

    Notes
    -----
    Utilizes `args` to extract `reduction` parameter for loss reduction
    and the `model` for the type of model.
    '''

    loss_reduction = None
    # loss reduction is applied to aggregate the samples losses within each batch
    if args.reduction == 'hnm':
        loss_reduction = reduction.HardNegativeMining()
    elif args.reduction == 'sum':
        loss_reduction = reduction.SumReduction()
    else:
        loss_reduction = reduction.MeanReduction()
    # logreg is a plain logistic regression algorithm optimized using SGD
    # cnn is a 2D convolutional neural network
    if args.model == 'cnn':
        return CNN(
            num_classes = 62,
            loss_reduction = loss_reduction
        ) 
    elif args.model == 'logreg':
        LogisticRegression(
            num_inputs = 784, 
            num_classes = 62,
            loss_reduction = loss_reduction
        )
    # no other models are implemented for such setting
    raise RuntimeError(f'unrecognized model \'{args.model}\', expected \'cnn\' or \'logreg\'')

def initialize_federated_algorithm(args: Any) -> algorithm.FedAlgorithm:
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

    from functools import partial

    if not args.algorithm or args.algorithm[0] == 'fedavg':
        return algorithm.FedAvg
    elif args.algorithm[0] == 'fedprox':
        return partial(algorithm.FedProx, mu = float(args.algorithm[1])) if len(args.algorithm) > 1 else algorithm.FedProx
    # federated algorithm not recognized
    raise RuntimeError(f'unrecognized federated algorithm \'{args.algorithm[0]}\', expected \'fedavg\' or \'fedprox\'')


def get_arguments() -> Any:
    '''
    Parses command line arguments.

    Returns
    -------
    Any
        Command line arguments
    '''
    
    parser = argparse.ArgumentParser(
        usage = 'run experiment on baseline FEMNIST dataset (federated, so decentralized) with a CNN architecture',
        description = 'This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = "note: argument \'--scheduler\' accept different learning rate scheduling choices (\'exp\', \'onecycle\' or \'step\') followed by decaying factor and decaying period\n\n" \
            "examples:\n\n" \
            ">>> python3 experiments/script.py --batch_size 256 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --rounds 1000 --epochs 1 --scheduler exp 0.5\n\n" \
            "This command executes the experiment using\n" \
            " [+] algorithm: fedavg\n" \
            " [+] batch size: 256\n" \
            " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5 every central round\n" \
            " [+] SGD momentum: 0.9\n" \
            " [+] SGD weight decay penalty: 0.0001\n" \
            " [+] running server rounds for training and validation: 1000\n" \
            " [+] running local client epoch: 1\n\n" \
            ">>> python3 experiments/script.py --batch_size 512 --learning_rate 0.01 --epochs 5 --rounds 1000 --scheduler step 0.75 3 --algorithm fedprox 0.25\n\n" \
            "This command executes the experiment using\n" \
            " [+] algorithm: fedprox with mu parameter equal to 0.25\n" \
            " [+] batch size: 512\n" \
            " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 central rounds\n" \
            " [+] running server rounds for training and validation: 1000\n" \
            " [+] running local client epoch: 5\n\n" \
            ">>> python3 experiments/script.py --niid --batch_size 512 --learning_rate 0.01 --epochs 5 --rounds 500 --scheduler onecycle 0.1 --algorithm fedavg --checkpoint 5 ./saved\n\n" \
            "This command executes the experiment using\n" \
            " [+] algorithm: fedavg\n" \
            " [+] checkpoint: saves model parameters every 5 rounds in ./saved\n" \
            " [+] dataset distribution: niid (unbalanced across clients)\n" \
            " [+] batch size: 512\n" \
            " [+] learning rate: 0.1 with one cycle cosine annealing rising up to a peak of 0.1 and then decreasing\n" \
            " [+] running server rounds for training and validation: 500\n" \
            " [+] running local client epoch: 5"
    )
    
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--dataset', type = str, choices = ['femnist'], default = 'femnist', help = 'dataset name')
    parser.add_argument('--niid', action = 'store_true', default = False, help = 'run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type = str, choices = ['logreg', 'cnn'], default = 'cnn', help = 'model name')
    parser.add_argument('--rounds', type = int, help = 'number of rounds')
    parser.add_argument('--epochs', type = int, help = 'number of local epochs')
    parser.add_argument('--selected', type = int, help = 'number of clients trained per round')
    parser.add_argument('--selection', choices = ['uniform', 'hybrid', 'poc'], default = 'uniform', type = str, help = 'criterion for selecting partecipating clients each round')
    parser.add_argument('--reduction', type = str, default = 'mean', choices = ['mean', 'sum', 'hnm'], help = 'Hard negative mining or mean or sum loss reduction')
    parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'Learning rate decay scheduling, like \'step\' or \'exp\' or \'onecycle\'')
    parser.add_argument('--algorithm', metavar = ('algorithm', 'params'), nargs = '+', type = str, default = ['fedavg'], help = 'Federated learning algorithm, like \'fedavg\' (default) or \'fedprox\'')
    parser.add_argument('--evaluation', type = int, default = 10, help = 'evaluation interval of training and validation set')
    parser.add_argument('--checkpoint', metavar = ('interval', 'params'), type = str, nargs = '+', default = None, help = 'Checkpoint after rounds interval and directory')
    parser.add_argument('--log', action = 'store_true', default = False, help = 'whether or not to log to weights & biases')

    return parser.parse_args()

if __name__ == '__main__':
    # parameters of the simulation
    args = get_arguments()
    # random seed and more importantly deterministic algorithm versions are set
    set_seed(args.seed)
    # model is compiled to CPU and operates on GPU
    print(f'[+] initializing model... ', end = '', flush = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # FIXME no scheduling implemented in federated model
    model = initialize_model(args)
    model = model.to(device)
    print('done')
    # federated datasets are shared among training and testing clients
    print('[+] loading datasets... ', end = '', flush = True)
    datasets = femnist.load(
        directory = os.path.join(
            'data', 
            'femnist', 
            'compressed', 
            'niid' if args.niid else 'iid'
        )
    )
    print('done')
    # client construction by dividing them in three groups (training, validation, testing)
    # each client of each group has its own private user dataset
    print('[+] constructing clients... ', end = '', flush = True)
    clients = client.construct(datasets, device, args)
    print('done')
    # server uses training clients and validation clients when fitting the central model
    # clients from testing group should be used at the very end
    server = server.Server(
        algorithm = initialize_federated_algorithm(args), 
        model = model,
        clients = clients,
        args = args
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] seed: {args.seed}')
    print(f"  [-] distribution: {'niid' if args.niid else 'iid'}")
    print(f'  [-] batch size: {args.batch_size}')
    print(f'  [-] learning rate: {args.learning_rate}')
    print(f'  [-] momentum: {args.momentum}')
    print(f'  [-] weight decay L2: {args.weight_decay}')
    print(f"  [-] learning rate scheduling: {' '.join(args.scheduler)}")
    print(f"  [-] local loss reduction: {args.reduction}")
    print(f"  [-] federated algorithm: {' '.join(args.algorithm)}")
    print(f'  [-] rounds: {args.rounds}')
    print(f'  [-] epochs: {args.epochs}')
    print(f'  [-] clients selected: {args.selected}')
    print(f'  [-] selection strategy: {args.selection}')
    print(f'  [-] checkpoint: {args.checkpoint}')
    print(f'  [-] remote log enabled: {args.log}')
    # initialize configuration for weights & biases log
    wandb.init(
        mode = 'online' if args.log else 'disabled',
        project = 'federated_learning',
        name = f"FEMNIST{'_NIID' if args.niid else '_IID'}_S{args.seed}_BS{args.batch_size}_LR{args.learning_rate}_M{args.momentum}_WD{args.weight_decay}_NR{args.rounds}_NE{args.epochs}_LRS{','.join(args.scheduler)}_C{args.selected}_S{args.selection}_R{args.reduction}_A{args.algorithm}",
        config = {
            'seed': args.seed,
            'dataset': args.dataset,
            'niid': args.niid,
            'model': args.model,
            'rounds': args.rounds,
            'epochs': args.epochs,
            'selected': args.selected,
            'selection': args.selection,
            'reduction': args.reduction,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    )
    # execute training and validation epochs
    server.run()
    # terminate weights & biases session by synchronizing
    wandb.finish()