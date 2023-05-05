import argparse
import numpy as np
import os
import random
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import wandb

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import cnn


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_arguments():
    epilog = "note: argument \'--lrs\' accept different learning rate scheduling choices (\'exp\' or \'step\') followed by decaying factor and decaying period\n\n" \
        "examples:\n\n" \
        ">>> python3 experiments/centralized.baseline.py --bs 256 --lr 0.1 --m 0.9 --wd 0.0001 --num_epochs 10 --lrs exp 0.5\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 256\n" \
        " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5\n" \
        " [+] SGD momentum: 0.9\n" \
        " [+] SGD weight decay penalty: 0.0001\n" \
        " [+] running epoch for training and validation: 10\n\n" \
        ">>> python3 experiments/centralized.baseline.py --bs 512 --lr 0.01 --num_epochs 100 --lrs step 0.75 3\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 epochs\n" \
        " [+] running epoch for training and validation: 100"
    parser = argparse.ArgumentParser(
        usage='run experiment on baseline EMNIST dataset (centralized) with a CNN architecture',
        description='This program is used to log to Weights & Biases training and validation results\nevery epoch of training on the EMNIST dataset. The employed architecture is a\nconvolutional neural network with two convolutional blocks and a fully connected layer.\nStochastic gradient descent is used by default as optimizer along with cross entropy loss.',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_epochs', default=10, type=int, help='number of local epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=512, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--lrs', metavar=('scheduler', 'params'), nargs='+', type=str, default=['none'], help='Learning rate decay scheduling')
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


def load_model(args):
    scheling = args.lrs[0].lower()
    model = cnn.Network(
        n_classes = 62, 
        learning_rate = args.lr, 
        momentum = args.m, 
        weight_decay = args.wd
    )
    # constructs learning rate scheduler
    if scheling == 'none':
        model.scheduler = None
    elif scheling == 'exp':
        model.scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma = float(args.lrs[1]))
    elif scheling == 'step':
        model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size = int(args.lrs[2]), gamma = float(args.lrs[1]))
    else:
        print('[*] unrecognized learning rate scheduling, set to \'none\'')
        model.scheduler = None
    # yields built model
    return model


def training(model: cnn.Network, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    # compute predictions
    predicted = model(X)
    # compute loss
    loss = model.criterion(predicted, y)
    # update weights after gradient computation
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    # return loss and score
    return loss.item(), (predicted.argmax(1) == y).to(dtype = torch.float).sum().item()

def validation(model: cnn.Network, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    # compute predictions
    predicted = model(X)
    # compute loss
    loss = model.criterion(predicted, y)
    # return loss and score
    return loss.item(), (predicted.argmax(1) == y).to(dtype = torch.float).sum().item()

def run(model: torch.nn.Module, training_loader: DataLoader, validation_loader: DataLoader, epochs: int):
    # running device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # move model to device
    model = model.to(device)
    # log
    wandb.watch(model, log = 'all')
    wandb.define_metric('epoch')
    wandb.define_metric('loss/training', step_metric = 'epoch')
    wandb.define_metric('loss/testing', step_metric = 'epoch')
    wandb.define_metric('accuracy/training', step_metric = 'epoch')
    wandb.define_metric('accuracy/testing', step_metric = 'epoch')
    # compile if possible
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    print(f'[+] running on device \'{device}\'')
    # epochs of training
    for epoch in range(epochs):
        print(f'[+] training epoch {epoch + 1}/{epochs}...')
        training_progress = tqdm.tqdm(total = len(training_loader), desc = '  [-] training')
        validation_progress = tqdm.tqdm(total = len(validation_loader), desc = '  [-] validation')
        # metrics
        training_loss, training_score = 0, 0
        validation_loss, validation_score = 0, 0
        # training
        model.train()
        # train loop
        for X, y in training_loader:
            training_batch_loss, training_batch_score = training(model, X.to(device), y.to(device))
            training_loss, training_score = training_loss + training_batch_loss, training_score + training_batch_score
            training_progress.update(1)
        # enable validation mode
        model.eval()
        with torch.no_grad():
            for X, y in validation_loader:
                validation_batch_loss, validation_batch_score = validation(model, X.to(device), y.to(device))
                validation_loss, validation_score = validation_loss + validation_batch_loss, validation_score + validation_batch_score
                validation_progress.update(1)
        # execute scheduler for learning rate
        if model.scheduler is not None:
            model.scheduler.step()
        # stop progress bars
        training_progress.close()
        validation_progress.close()
        # adjust metrics
        training_loss, training_score = training_loss / len(training_loader), training_score / len(training_loader.dataset)
        validation_loss, validation_score = validation_loss / len(validation_loader), validation_score / len(validation_loader.dataset)
        # remote log
        wandb.log({
            'epoch': epoch + 1,
            'loss/training': training_loss,
            'loss/testing': validation_loss,
            'accuracy/training': training_score,
            'accuracy/testing': validation_score
        })
        # log metrics
        print(f'  [-] loss: {training_loss:.3f}, score: {training_score:.3f} (training)')
        print(f'  [-] loss: {validation_loss:.3f}, score: {validation_score:.3f} (validation)')

if __name__ == '__main__':
    # command line arguments
    args = get_arguments()
    # set seed
    set_seed(args.seed)
    # data loaders
    training_loader, validation_loader = load_emnist(batch_size = args.bs)
    # load model
    model = load_model(args)
    # log
    wandb.init(
        project='federated_learning',
        name=f"EMNIST_BS{args.bs}_LR{args.lr}_M{args.m}_WD{args.wd}_NE{args.num_epochs}_LRS{','.join(args.lrs)}",
        config={
            'seed': args.seed,
            'dataset': 'emnist',
            'model': 'cnn',
            'num_epochs': args.num_epochs,
            'learning_rate': args.lr,
            'scheduling': args.lrs,
            'batch_size': args.bs,
            'weight_decay': args.wd,
            'momentum': args.m
        }
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] batch size: {args.bs}')
    print(f'  [-] learning rate: {args.lr}')
    print(f'  [-] momentum: {args.m}')
    print(f'  [-] weight decay L2: {args.wd}')
    print(f'  [-] epochs: {args.num_epochs}')
    print(f'  [-] learning rate scheduling: {args.lrs}')
    # train and validate model as expeced
    run(model, training_loader, validation_loader, epochs = args.num_epochs)
    # terminate weights & biases session by sincynchronizing
    wandb.finish()
