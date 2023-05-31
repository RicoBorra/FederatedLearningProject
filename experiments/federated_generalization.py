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
import utils.algorithm as algorithm
import utils.reduction as reduction
from models.cnn import CNN
from models.logistic_regression import LogisticRegression
from models.sr import SimpleRepresentationClassifier
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
    # variational cnn is instantiated in case of fedsr
    # logreg is a plain logistic regression algorithm optimized using SGD
    # cnn is a 2D convolutional neural network
    if args.algorithm[0] == 'fedsr':
        return SimpleRepresentationClassifier(
            num_classes = 62,
            num_latent_features = 64 if len(args.algorithm) < 2 else int(args.algorithm[1]),
            beta_l2 = 1e-3 if len(args.algorithm) < 3 else float(args.algorithm[2]),
            beta_kl = 1e-3 if len(args.algorithm) < 4 else float(args.algorithm[3])
        )
    elif args.model == 'cnn':
        return CNN(
            num_classes = 62,
            loss_reduction = loss_reduction
        ) 
    elif args.model == 'logreg':
        return LogisticRegression(
            num_inputs = 784, 
            num_classes = 62,
            loss_reduction = loss_reduction
        )
    # no other models are implemented for such setting
    raise RuntimeError(f'unrecognized model \'{args.model}\', expected \'cnn\' or \'logreg\'')

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

    scheduler = initialize_learning_rate_scheduler(args)
    # binds scheduler
    if not args.algorithm or args.algorithm[0] in [ 'fedavg', 'fedsr' ]:
        return partial(algorithm.FedAvg, scheduler = scheduler)
    elif args.algorithm[0] == 'fedprox':
        return partial(algorithm.FedProx, scheduler = scheduler, mu = float(args.algorithm[1])) if len(args.algorithm) > 1 else partial(algorithm.FedProx, scheduler = scheduler)
    elif args.algorithm[0] == 'fedyogi':
        return partial(
            algorithm.FedYogi, 
            scheduler = scheduler,
            beta_1 = 0.9 if len(args.algorithm) < 2 else float(args.algorithm[1]),
            beta_2 = 0.99 if len(args.algorithm) < 3 else float(args.algorithm[2]),
            tau = 1e-4 if len(args.algorithm) < 4 else float(args.algorithm[3]),
            eta = 10 ** (-2.5) if len(args.algorithm) < 5 else float(args.algorithm[4])
        )
    # federated algorithm not recognized
    raise RuntimeError(f'unrecognized federated algorithm \'{args.algorithm[0]}\', expected \'fedavg\', \'fedsr\', \'fedyogi\' or \'fedprox\'')


def get_arguments() -> Any:
    '''
    Parses command line arguments.

    Returns
    -------
    Any
        Command line arguments
    '''
    
    parser = argparse.ArgumentParser(
        usage = 'run experiment on baseline FEMNIST dataset (federated, so decentralized) with a CNN architecture in a domain generalization setting',
        description = 'This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = "note: argument \'--scheduler\' accept different learning rate scheduling choices (\'exp\', \'linear\' or \'step\') followed by decaying factor and decaying period\n\n" \
            "examples:\n\n" \
            ">>> python3 experiments/script.py --validation_domain_angle 75 --batch_size 256 --learning_rate 0.1 --momentum 0.9 --weight_decay 0.0001 --rounds 1000 --epochs 1 --scheduler exp 0.5 --algorithm fedyogi 0.9 0.99 1e-4 1e-2\n\n" \
            "This command executes the experiment using\n" \
            " [+] validation domain angle: among rotated domains (0, 15, ..., 75) the one of 75 degrees counter clockwise rotation is selected for validation\n" \
            " [+] algorithm: fedyogi with beta_1 = 0.9, beta_2 = 0.99, tau = 1e-4 and eta = 1e-2\n" \
            " [+] batch size: 256\n" \
            " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5 every central round\n" \
            " [+] SGD momentum: 0.9\n" \
            " [+] SGD weight decay penalty: 0.0001\n" \
            " [+] running server rounds for training and validation: 1000\n" \
            " [+] running local client epoch: 1\n\n" \
            ">>> python3 experiments/script.py --batch_size 512 --learning_rate 0.01 --epochs 5 --rounds 1000 --scheduler step 0.75 3 --algorithm fedsr 512 1e-3 1e-2\n\n" \
            "This command executes the experiment using\n" \
            " [+] validation domain angle: none, thus simple 80:20 split for training and validation clients without target domain validation\n" \
            " [+] algorithm: fedsr with latent size 512 and regularizations coefficents 1e-3 and 1e-2, respectively for L2 norm and CMI\n" \
            " [+] batch size: 512\n" \
            " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 central rounds\n" \
            " [+] running server rounds for training and validation: 1000\n" \
            " [+] running local client epoch: 5\n\n" \
            ">>> python3 experiments/script.py --validation_domain_angle 30 --niid --batch_size 512 --learning_rate 0.01 --epochs 5 --rounds 500 --algorithm fedavg --selection poc 30\n\n" \
            "This command executes the experiment using\n" \
            " [+] validation domain angle: among rotated domains (0, 15, ..., 75) the one of 30 degrees counter clockwise rotation is selected for validation\n" \
            " [+] algorithm: fedavg\n" \
            " [+] dataset distribution: niid (unbalanced across clients)\n" \
            " [+] batch size: 512\n" \
            " [+] selection strategy: power of choice with candidate set of 30 clients\n" \
            " [+] running server rounds for training and validation: 500\n" \
            " [+] running local client epoch: 5\n\n" \
            ">>> python3 experiments/script.py --validation_domain_angle 30 --niid --batch_size 512 --learning_rate 0.01 --epochs 5 --rounds 500 --algorithm fedsr 64 1e-2 5e-4 --selection poc 30\n\n" \
            "This command executes the experiment using\n" \
            " [+] validation domain angle: among rotated domains (0, 15, ..., 75) the one of 30 degrees counter clockwise rotation is selected for validation\n" \
            " [+] algorithm: fedsr using 64 as simple representation dimension and beta_l2n = 1e-2 and beta_cmi = 5e-4\n" \
            " [+] dataset distribution: niid (unbalanced across clients)\n" \
            " [+] batch size: 512\n" \
            " [+] selection strategy: power of choice with candidate set of 30 clients\n" \
            " [+] running server rounds for training and validation: 500\n" \
            " [+] running local client epoch: 5"
    )
    
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--dataset', type = str, choices = ['femnist'], default = 'femnist', help = 'dataset name')
    parser.add_argument('--niid', action = 'store_true', default = False, help = 'run the experiment with the non-IID partition (IID by default), only on FEMNIST dataset')
    parser.add_argument('--model', type = str, choices = ['logreg', 'cnn'], default = 'cnn', help = 'model name')
    parser.add_argument('--validation_domain_angle', type = int, choices = [ None, 0, 15, 30, 45, 60, 75 ], default = None, help = 'rotated domain of clients which is selected for validation')
    parser.add_argument('--rounds', type = int, help = 'number of rounds')
    parser.add_argument('--epochs', type = int, help = 'number of local epochs')
    parser.add_argument('--selected', type = int, help = 'number of clients trained per round')
    parser.add_argument('--selection', metavar = ('selection', 'params'), type = str, nargs = '+', default = ['uniform'], help = 'criterion for selecting partecipating clients each round, like \'uniform\' or \'hybrid\' or \'poc\'')
    parser.add_argument('--reduction', type = str, default = 'mean', choices = ['mean', 'sum', 'hnm'], help = 'hard negative mining or mean or sum loss reduction')
    parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'learning rate decay scheduling, like \'step\' or \'exp\' or \'linear\'')
    parser.add_argument('--algorithm', metavar = ('algorithm', 'params'), nargs = '+', type = str, default = ['fedavg'], help = 'federated learning algorithm, like \'fedavg\' (default), \'fedprox\' or \'fedsr\'')
    parser.add_argument('--evaluation', type = int, default = 10, help = 'evaluation interval of training and validation set')
    parser.add_argument('--evaluators', type = float, default = float(250), help = 'fraction (if < 1.0) or number (if >= 1) of clients to be evaluated from training and validation set')
    parser.add_argument('--testing', action = 'store_true', default = True, help = 'run final evaluation on unseen testing clients')
    parser.add_argument('--save', type = str, default = 'checkpoints', help = 'save state dict after training in path')
    parser.add_argument('--log', action = 'store_true', default = False, help = 'whether or not to log to weights & biases')

    return parser.parse_args()

if __name__ == '__main__':
    # parameters of the simulation
    args = get_arguments()
    # random seed and more importantly deterministic algorithm versions are set
    set_seed(args.seed)
    # model is compiled to CPU and operates on GPU
    print(f'[+] initializing experiment in a domain generalization setting (rotated images)')
    print(f'[+] initializing model... ', end = '', flush = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # FIXME no scheduling implemented in federated model
    model = initialize_model(args)
    model = model.to(device)
    print('done')
    # federated datasets are shared among training and testing clients
    print('[+] loading datasets... ', end = '', flush = True)
    angles = [ 0, 15, 30, 45, 60, 75 ]
    datasets = femnist.load_with_rotated_domains(
        directory = os.path.join(
            'data', 
            'femnist', 
            'compressed', 
            'niid' if args.niid else 'iid'
        ),
        n_rotated = 1000,
        angles = angles,
        validation_domain_angle = args.validation_domain_angle
    )
    print('done')
    # client construction by dividing them in three groups (training, validation, testing)
    # each client of each group has its own private user dataset
    print('[+] constructing clients... ', end = '', flush = True)
    clients = client.construct(datasets, device, args)
    print('done')
    # simulation identifier
    identifier = f"dg_{'niid' if args.niid else 'iid'}{'' if args.validation_domain_angle is None else f'_va{args.validation_domain_angle}'}_s{args.seed}_a{':'.join(args.algorithm)}_r{args.rounds}_e{args.epochs}_c{args.selected}_cs{':'.join(args.selection)}_lr{args.learning_rate}_lrs{':'.join(args.scheduler)}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}_rd{args.reduction}"
    # server uses training clients and validation clients when fitting the central model
    # clients from testing group should be used at the very end
    server = server.Server(
        algorithm = initialize_federated_algorithm(args), 
        model = model,
        clients = clients,
        args = args,
        id = identifier
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] id: {identifier}')
    print(f'  [-] seed: {args.seed}')
    print(f"  [-] distribution: {'niid' if args.niid else 'iid'}")
    print(f"  [-] angles: {' '.join([ str(a) for a in angles])}")
    print(f"  [-] validation domain angle: {args.validation_domain_angle if args.validation_domain_angle else 'none'}")
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
    print(f"  [-] selection strategy: {' '.join(args.selection)}")
    print(f'  [-] fraction or number of clients evaluated: {args.evaluators}')
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
            'validation_domain_angle': args.validation_domain_angle,
            'niid': args.niid,
            'model': args.model,
            'rounds': args.rounds,
            'epochs': args.epochs,
            'selected': args.selected,
            'selection': ':'.join(args.selection),
            'reduction': args.reduction,
            'learning_rate': args.learning_rate,
            'scheduler': ':'.join(args.scheduler),
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'algorithm': ':'.join(args.algorithm)
        }
    )
    # execute training and validation epochs
    server.run()
    # terminate weights & biases session by synchronizing
    wandb.finish()