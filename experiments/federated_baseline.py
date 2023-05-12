import json
import numpy as np
import os
import random
import sys
import torch
from torch.utils.data import DataLoader

import wandb

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from client import Client
from datasets.femnist import load_femnist, Femnist
from server import Server
from utils.args import get_parser
import models.cnn as cnn
from utils.evaluation import ModelEvaluator


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(args):
    scheduling = args.lrs[0].lower()
    model = cnn.Network(
        n_classes = 62,
        learning_rate = args.lr,
        momentum = args.m,
        weight_decay = args.wd
    )
    # constructs learning rate scheduler
    if scheduling == 'none':
        model.scheduler = None
    elif scheduling == 'exp':
        model.scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma = float(args.lrs[1]))
    elif scheduling == 'step':
        model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size = int(args.lrs[2]), gamma = float(args.lrs[1]))
    elif scheduling == 'onecycle':
        model.scheduler = torch.optim.lr_scheduler.OneCycleLR(model.optimizer, max_lr = float(args.lrs[1]), epochs = int(args.num_rounds), steps_per_epoch = 1)
    else:
        print('[*] unrecognized learning rate scheduling, set to \'none\'')
        model.scheduler = None
    # yields built model
    return model


def load_datasets(args):
    # yields a tuple of training and testing datasets
    return load_femnist(
        directory = os.path.join(
            'data', 'femnist', 'compressed', 'niid' 
            if args.niid else 'iid'
        ),
        transforms = (
            # training data trasformation
            nptr.Compose([
                nptr.ToTensor(),
                nptr.Normalize((0.5,), (0.5,)),
            ]),
            # testing data trasformation
            nptr.Compose([
                nptr.ToTensor(),
                nptr.Normalize((0.5,), (0.5,)),
            ])
        ),
        as_csv = False
    )


def load_evaluators(args, training_data: Femnist, validation_data: Femnist, testing_data: Femnist):
    # constructs respective evaluators
    return {
        'train': ModelEvaluator(
            DataLoader(training_data, batch_size = 1024, shuffle = False)
        ),
        'validation': ModelEvaluator(
            DataLoader(validation_data, batch_size = 1024, shuffle = False)
        ),
        'test': ModelEvaluator(
            DataLoader(testing_data, batch_size = 1024, shuffle = False)
        ),
    }


def load_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=(i == 1)))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args()
    # random seed and more importantly deterministic algorithm versions are set
    set_seed(args.seed)
    # model is compiled to CPU and operates on GPU for
    # algebraic operations
    print(f'[+] initializing model... ', end = '', flush = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args)
    # model = torch.compile(model)
    model = model.to(device)
    print('done')
    # federated datasets are shared among training and testing clients
    print('[+] loading datasets... ', end = '', flush = True)
    (
        training_clients_data, 
        validation_clients_data,
        testing_clients_data, 
        training_data,
        validation_data,
        testing_data 
    ) = load_datasets(args)
    print('done')
    # metric objects holds aggregated mean/overall/classes accuracies both for training and testing clients
    # metrics = load_metrics(args)
    # performance evaluators
    print('[+] loading datasets evaluators... ', end = '', flush = True)
    evaluators = load_evaluators(args, training_data, validation_data, testing_data)
    print('done')
    # initialize configuration for weights & biases log
    wandb.init(
        project='federated_learning',
        name=f"FEMNIST{'_NIID' if args.niid else '_IID'}_S{args.seed}_BS{args.bs}_LR{args.lr}_M{args.m}_WD{args.wd}_NR{args.num_rounds}_NE{args.num_epochs}_LRS{','.join(args.lrs)}_C{args.clients_per_round}_S{args.selection}",
        config={
            'seed': args.seed,
            'dataset': args.dataset,
            'niid': args.niid,
            'model': args.model,
            'num_rounds': args.num_rounds,
            'num_epochs': args.num_epochs,
            'clients_per_round': args.clients_per_round,
            'selection': args.selection,
            'hnm': args.hnm,
            'learning_rate': args.lr,
            'batch_size': args.bs,
            'weight_decay': args.wd,
            'momentum': args.m
        }
    )
    # build clients partitioning
    train_clients, validation_clients = load_clients(args, training_clients_data, validation_clients_data, model)
    server = Server(args, train_clients, validation_clients, model, evaluators)
    # initial log
    print('[+] running with configuration')
    print(f'  [-] seed: {args.seed}')
    print(f"  [-] distribution: {'niid' if args.niid else 'iid'}")
    print(f'  [-] batch size: {args.bs}')
    print(f'  [-] learning rate: {args.lr}')
    print(f'  [-] momentum: {args.m}')
    print(f'  [-] weight decay L2: {args.wd}')
    print(f'  [-] learning rate scheduling: {args.lrs}')
    print(f'  [-] rounds: {args.num_rounds}')
    print(f'  [-] epochs: {args.num_epochs}')
    print(f'  [-] clients selected: {args.clients_per_round}')
    print(f'  [-] selection strategy: {args.selection}')
    # execute training and validation epochs
    server.train()
    # terminate weights & biases session by sincynchronizing
    wandb.finish()


if __name__ == '__main__':
    main()
