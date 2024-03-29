import argparse
import numpy as np
import os
import random
import sys
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

import wandb

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import cnn
from utils.reduction import MeanReduction

class KFold(object):

    def __init__(self, batch_size: int, k: int = 5) -> None:
        self.k = k
        self.batch_size = batch_size

    def split(self, dataset: Dataset) -> list[tuple[DataLoader, DataLoader]]:
        indices = torch.arange(len(dataset))
        subsize = len(dataset) // self.k
        folds = torch.split(indices, subsize)
        partitions = []

        for i in range(self.k):
            training = torch.hstack([ fold for j, fold in enumerate(folds) if j != i ]).view(-1)
            validation = folds[i]
            partitions.append((
                DataLoader(Subset(dataset, training), batch_size = self.batch_size),
                DataLoader(Subset(dataset, validation), batch_size = self.batch_size)
            ))

        return partitions

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
    parser.add_argument('--epochs', default = 10, type = int, help = 'number of local epochs')
    parser.add_argument('--learning_rate', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'learning rate decay scheduling')
    parser.add_argument('--kfold', type=int, default=None, help='kfold cross validation')
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

def load_k_fold_emnist(k: int, batch_size: int) -> tuple[list[tuple[DataLoader, DataLoader]], DataLoader]:
    return (
        KFold(batch_size = batch_size, k = k).split(
            datasets.EMNIST(root = 'data/emnist', train = True, download = True, split = 'byclass', transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ]))
        ),
        DataLoader(
            datasets.EMNIST(root = 'data/emnist', train = False, download = True, split = 'byclass', transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])),
            batch_size = batch_size
        )
    )


def load_model(args, n_batches: int):
    scheduling = args.scheduler[0].lower()
    model = cnn.CNN(
        num_classes = 62,
        loss_reduction = MeanReduction(),
    )
    model.optimizer = torch.optim.SGD(
        model.parameters(),
        lr = args.learning_rate, 
        momentum = args.momentum, 
        weight_decay = args.weight_decay
    )
    # constructs learning rate scheduler
    if scheduling == 'none':
        model.scheduler = None
    elif scheduling == 'exp':
        model.scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma = float(args.scheduler[1]))
    elif scheduling == 'step':
        model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size = int(args.scheduler[2]), gamma = float(args.scheduler[1]))
    elif scheduling == 'onecycle':
        model.scheduler = torch.optim.lr_scheduler.OneCycleLR(model.optimizer, max_lr = float(args.scheduler[1]), epochs = int(args.epochs), steps_per_epoch = n_batches)
    elif scheduling == 'linear':
        model.scheduler = torch.optim.lr_scheduler.LinearLR(model.optimizer, start_factor = float(args.scheduler[1]), total_iters = int(args.scheduler[2]))
    else:
        print('[*] unrecognized learning rate scheduling, set to \'none\'')
        model.scheduler = None
    # yields built model
    return model

def run(model: torch.nn.Module, training_loader: DataLoader, validation_loader: DataLoader, epochs: int) -> tuple[float, float]:
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
    model = model.to(device)
    # log
    wandb.watch(model, log = 'all')
    wandb.define_metric('epoch')
    wandb.define_metric('loss/training', step_metric = 'epoch')
    wandb.define_metric('loss/validation', step_metric = 'epoch')
    wandb.define_metric('accuracy/training', step_metric = 'epoch')
    wandb.define_metric('accuracy/validation', step_metric = 'epoch')
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
            X, y = X.to(device), y.to(device)
            training_batch_logits, training_batch_loss = model.step(X, y, model.optimizer)
            training_batch_score = (training_batch_logits.argmax(1) == y).sum().item()
            training_loss, training_score = training_loss + training_batch_loss, training_score + training_batch_score
            training_progress.update(1)
            # single batch update regarding onecycle annealing
            if model.scheduler in [ 'onecycle' ]:
                model.scheduler.step()
        # enable validation mode
        model.eval()
        with torch.no_grad():
            for X, y in validation_loader:
                X, y = X.to(device), y.to(device)
                _, validation_batch_predicted, validation_batch_loss, _ = model.evaluate(X, y)
                validation_batch_score = (validation_batch_predicted == y).sum().item()
                validation_loss, validation_score = validation_loss + validation_batch_loss, validation_score + validation_batch_score
                validation_progress.update(1)
        # execute scheduler for learning rate
        if model.scheduler in [ 'step', 'exp' ]:
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
    set_seed(args.seed)
    # simulation identifier
    identifier = f"emnist_s{args.seed}_e{args.epochs}_lr{args.learning_rate}_lrs{':'.join(args.scheduler)}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}"
    # log
    wandb.init(
        mode = 'online' if args.log else 'disabled',
        project = 'centralized',
        name = identifier,
        config = {
            'seed': args.seed,
            'dataset': 'emnist',
            'model': 'cnn',
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'scheduler': ':'.join(args.scheduler),
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] id: {identifier}')
    print(f'  [-] seed: {args.seed}')
    print(f'  [-] batch size: {args.batch_size}')
    print(f'  [-] learning rate: {args.learning_rate}')
    print(f'  [-] momentum: {args.momentum}')
    print(f'  [-] weight decay L2: {args.weight_decay}')
    print(f'  [-] epochs: {args.epochs}')
    print(f"  [-] learning rate scheduling: {' ' .join(args.scheduler)}")
    print(f'  [-] remote log enabled: {args.log}')
    # execution has two modes, kfold or none
    if args.kfold is not None:
        # data loaders (training and validation for each fold) and unique testing loader
        loaders, testing_loader = load_k_fold_emnist(k = 5, batch_size = args.batch_size)
        # load model
        model = load_model(args, n_batches = len(loaders[0][0]))
        print('[+] running cross validation')
        # k fold progress bar
        progress = tqdm.tqdm(total = len(loaders), desc = f'  [-] evaluating fold')
        # k fold results
        losses = []
        scores = []
        # kfold validation multiple runs
        for training_loader, validation_loader in loaders:
            # train and validate model as expected
            loss, score = run(model, training_loader, validation_loader, epochs = args.epochs)
            # utilizes results for averaging later
            losses.append(loss)
            scores.append(score)
            progress.update(1)
        # compute averages
        progress.close()
        print(f'[+] mean validation loss: {np.mean(losses):.3f} ± {np.std(losses):.3f}')
        print(f'[+] mean validation score: {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    else:
        # data loaders
        training_loader, validation_loader = load_emnist(batch_size = args.batch_size)
        # load model
        model = load_model(args, n_batches = len(training_loader))
        # train and validate model as expeced
        run(model, training_loader, validation_loader, epochs = args.epochs)
    # terminate weights & biases session by sincynchronizing
    wandb.finish()
