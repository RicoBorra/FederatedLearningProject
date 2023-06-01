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

import datasets.femnist as femnist
from models import cnn
from utils.reduction import MeanReduction

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
        ">>> python3 experiments/script.py --batch_size 256 --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --epochs 10 --scheduler exp 0.5\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 256\n" \
        " [+] learning rate: 0.1 decaying exponentially with multiplicative factor 0.5\n" \
        " [+] SGD momentum: 0.9\n" \
        " [+] SGD weight decay penalty: 0.0001\n" \
        " [+] running epoch for training and validation: 10\n\n" \
        ">>> python3 experiments/script.py --batch_size 512 --lr 0.01 --epochs 100 --scheduler step 0.75 3\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 decaying using step function with multiplicative factor 0.75 every 3 epochs\n" \
        " [+] running epoch for training and validation: 100\n\n" \
        ">>> python3 experiments/script.py --batch_size 512 --lr 0.01 --epochs 100 --scheduler linear 0.1 8\n\n" \
        "This command executes the experiment using\n" \
        " [+] batch size: 512\n" \
        " [+] learning rate: 0.1 is starting factor of linear growth for 8 epochs\n" \
        " [+] running epoch for training and validation: 100\n\n" \
        ">>> python3 experiments/script.py --validation_domain_angle 45 --batch_size 512 --lr 0.01 --epochs 100 --scheduler onecycle 0.1\n\n" \
        "This command executes the experiment using\n" \
        " [+] validation domain angle: uses datasets from client whose image are rotated of 45 degrees for validation\n" \
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
    parser.add_argument('--lr', type = float, default = 0.05, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('--scheduler', metavar = ('scheduler', 'params'), nargs = '+', type = str, default = ['none'], help = 'learning rate decay scheduling')
    parser.add_argument('--log', action='store_true', default=False, help='whether or not to log to weights & biases')
    parser.add_argument('--validation_domain_angle', type = int, choices = [ None, 0, 15, 30, 45, 60, 75 ], default = None, help = 'rotated domain of clients which is selected for validation')
    return parser.parse_args()


def load_rotated_emnist(
        batch_size: int, 
        angles: list[int], 
        validation_domain_angle: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # here datasets of each group are fragment across multiple groups
    federated_datasets = femnist.load_with_rotated_domains(
        directory = os.path.join(
            'data', 
            'femnist', 
            'compressed', 
            # NOTE since we are dealing with domain generalization,
            # we adopt a niid client division strategy from the beginning,
            # so that we can establish a baseline
            'niid'
        ),
        n_rotated = 1000,
        angles = angles,
        validation_domain_angle = validation_domain_angle
    )
    # make the clients' datasets centralized so that we can build loaders upon them
    centralized_datasets = femnist.centralized(federated_datasets)
    # constructs loaders
    return (
        DataLoader(
            centralized_datasets['training'],
            batch_size = batch_size
        ),
        DataLoader(
            centralized_datasets['validation'],
            batch_size = batch_size
        ),
        DataLoader(
            centralized_datasets['testing'],
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
        lr = args.lr, 
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

def run(model: torch.nn.Module, training_loader: DataLoader, validation_loader: DataLoader, testing_loader: DataLoader, epochs: int) -> tuple[float, float]:
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
        # actually 'testing' is 'validation' but the key is mantained for legacy with
        # respect to previous experiment of centralized baseline
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
    # progress bar for testing
    testing_progress = tqdm.tqdm(total = len(testing_loader), desc = '  [-] testing')
    testing_loss, testing_score = 0, 0
    # actually testing clients are not used in this context but just for final evaluation
    # enable validation mode
    model.eval()
    with torch.no_grad():
        for X, y in testing_loader:
            X, y = X.to(device), y.to(device)
            _, testing_batch_predicted, testing_batch_loss, _ = model.evaluate(X, y)
            testing_batch_score = (testing_batch_predicted == y).sum().item()
            testing_loss, testing_score = testing_loss + testing_batch_loss, testing_score + testing_batch_score
            testing_progress.update(1)
    
    testing_progress.close()
    # adjust metrics
    testing_loss, testing_score = testing_loss / len(testing_loader), testing_batch_score / len(testing_loader.dataset)
    print(f'  [-] loss: {testing_loss:.3f}, score: {testing_score:.3f} (testing)')
    # yields score
    return validation_loss, validation_score

if __name__ == '__main__':
    # command line arguments
    args = get_arguments()
    angles = [ 0, 15, 30, 45, 60, 75 ]
    # set seed
    set_seed(args.seed)
    # simulation identifier
    identifier = f"dg_emnist{'' if args.validation_domain_angle is None else f'_va{args.validation_domain_angle}'}_s{args.seed}_e{args.epochs}_lr{args.lr}_lrs{':'.join(args.scheduler)}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}"
    # log
    wandb.init(
        mode = 'online' if args.log else 'disabled',
        project = 'centralized',
        name = identifier,
        config = {
            'seed': args.seed,
            'dataset': 'emnist',
            'validation_domain_angle': args.validation_domain_angle,
            'model': 'cnn',
            'epochs': args.epochs,
            'lr': args.lr,
            'scheduler': ':'.join(args.scheduler),
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,

        }
    )
    # initial log
    print('[+] running with configuration')
    print(f'  [-] id: {identifier}')
    print(f'  [-] seed: {args.seed}')
    print(f"  [-] angles: {' '.join([ str(a) for a in angles])}")
    print(f"  [-] validation domain angle: {args.validation_domain_angle if args.validation_domain_angle else 'none'}")
    print(f'  [-] batch size: {args.batch_size}')
    print(f'  [-] learning rate: {args.lr}')
    print(f'  [-] momentum: {args.momentum}')
    print(f'  [-] weight decay L2: {args.weight_decay}')
    print(f'  [-] epochs: {args.epochs}')
    print(f"  [-] learning rate scheduling: {' ' .join(args.scheduler)}")
    print(f'  [-] remote log enabled: {args.log}')
    # data loaders
    training_loader, validation_loader, testing_loader = load_rotated_emnist(
        batch_size = args.batch_size,
        angles = angles,
        validation_domain_angle = args.validation_domain_angle
    )
    # load model
    model = load_model(args, n_batches = len(training_loader))
    # train and validate model as expeced
    run(model, training_loader, validation_loader, testing_loader, epochs = args.epochs)
    # terminate weights & biases session by sincynchronizing
    wandb.finish()
