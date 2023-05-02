import os
import json

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from datasets.femnist import load_femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
import models.cnn as cnn
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

import wandb


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(
            in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'cnn':
        return cnn.Network(
            n_classes=get_dataset_num_classes(args.dataset),
            learning_rate=args.lr,
            momentum=args.m,
            weight_decay=args.wd,
            learning_rate_decay=1.0,
            learning_rate_decay_period=5
        )
    raise NotImplementedError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


def get_datasets(args):
    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]
    elif args.dataset == 'femnist':
        directory = os.path.join(
            'data', 'femnist', 'compressed', 'niid' if args.niid else 'iid')
        train_transforms, test_transforms = get_transforms(args)
        train_datasets, test_datasets = load_femnist(
            directory,
            transforms=(train_transforms, test_transforms),
            as_csv=False
        )
    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=(i == 1)))
    return clients[0], clients[1]


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    # model is compiled to CPU and operates on GPU for
    # algebraic operations
    print('[+] initializing model... ', end = '', flush = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_init(args)
    # model = torch.compile(model)
    model = model.to(device)
    print('done')
    # federated datasets are shared among training and testing clients
    print('[+] loading datasets... ', end = '', flush = True)
    train_datasets, test_datasets = get_datasets(args)
    print('done')
    # initialize configuration for weights & biases log
    wandb.init(
        project='mldl23-fl',
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
    # metric objects holds aggregated mean/overall/classes accuracies both for training and testing clients
    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    server = Server(args, train_clients, test_clients, model, metrics)
    server.train()
    # terminate weights & biases session by sincynchronizing
    wandb.finish()


if __name__ == '__main__':
    main()
