import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Network(nn.Module):
    '''
    This class represents a convolutional neural network employed for
    classification of images taken from EMNIST or FENMIST dataset, which
    are expected to be gray scaled images (not RGB) of dimension 28x28.
    By default, the network expects the dataset to be splitted with 
    'byclass' configuration, thus counting up to 62 different classes.

    Notes
    -----
    The network is built as a convolutional neural network with two convolutional
    layers (32 and 64 filters respectively of dimension 5x5) having a max pooling
    and ReLu activation. These are followed by two fully connected layers, where
    the former has 2048 neurons and the latter has a number of neurons corresponding
    to the amount of classes. Finally the network is trainable using stochastic gradient
    descent and uses the cross entropy loss as cost function.
    '''

    def __init__(
        self,
        n_classes: int,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4
    ):
        '''
        Constructs a convolutional neural network for classifying gray scale
        images of dimension 28x28.

        Parameters
        ----------
        n_classes: int
            Number of classes (64 by defaylt)
        learning_rate: float
            Learning rate for stochastic gradient descent (1e-3 by default)
        momentum: float
            Learning momentum for stochastic gradient descent (0.9 by default)
        weight_decay: float = 1e-4
            L2 weight penalty (1e-4 by default)
        '''

        super(Network, self).__init__()
        # number of classes
        self.n_classes = n_classes
        # convolutional neural network architecture
        self.architecture = nn.Sequential(
            # first convolutional block of 32 channels, max pooling and relu
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # second convolutional block of 64 channels, max pooling and relu
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # transform the two dimensional activation maps into one flatten array
            nn.Flatten(),
            # first fully connected layer of 2048 outputs
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU(),
            # second and last fully connected layer
            nn.Linear(in_features=2048, out_features=n_classes)
        )
        # optimizer configuration
        self.optimizer = optim.SGD(
            self.parameters(),
            momentum=momentum,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # loss criterion configuration
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        '''
        Computes the network output logits.

        Returns
        -------
        torch.Tensor
            Network logits corresponding to x
        '''

        return self.architecture(x)


class Trainer(object):
    '''
    This class establishes a façade pattern for training and validating
    a covolutional neural network.

    Notes
    -----
    For optimization reasons, the GPU is exploited whenever available.

    See also
    --------
    Network
        Trainable convolutional neural network
    '''
    
    def __init__(
        self,
        epochs: int,
        training_loader: DataLoader,
        testing_loader: DataLoader,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose: bool = True
    ):
        '''
        Constructs a trainer for a convolutional neural network.

        Parameters
        ----------
        epochs: int
            Number of epochs for training and validation
        training_loader: DataLoader
            Data loader for training samples
        testing_loader: DataLoader
            Data loader for validation samples
        device: torch.device
            Device employed for running operations (uses GPU is available)
        verbose: bool
            Verbosity of training and validation (True by default)
        '''
        
        self.epochs = epochs
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.device = device
        self.verbose = verbose

    def training_step(self, network: nn.Module) -> tuple[float, float]:
        '''
        Training step within an epoch, that is a single pass over the entire
        training set.

        Parameters
        ----------
        network: nn.Module
            Trainable network

        Returns
        -------
        tuple[float, float]
            Average training loss and accuracy
        '''
        
        loss = 0
        score = 0
        n_samples = len(self.training_loader.dataset)

        # enable training mode
        network.train()

        for X, y in self.training_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            # compute predictions
            predicted = network(X)
            # compute loss
            batch_loss = network.criterion(predicted, y)
            # update weights after gradient computation
            network.optimizer.zero_grad()
            batch_loss.backward()
            network.optimizer.step()
            # update loss and score
            loss += batch_loss.item()
            score += (predicted.argmax(1) ==
                      y).to(dtype=torch.float).sum().item()

        return loss / n_samples, score / n_samples

    def testing_step(self, network: nn.Module):
        '''
        Validation step within an epoch, that is a single pass over the entire
        validation set.

        Parameters
        ----------
        network: nn.Module
            Trainable network

        Returns
        -------
        tuple[float, float]
            Average validation loss and accuracy
        '''
        
        loss = 0
        score = 0
        n_samples = len(self.testing_loader.dataset)

        # enable validation mode
        network.eval()

        with torch.no_grad():
            for X, y in self.testing_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                # compute predictions
                predicted = network(X)
                # compute loss
                batch_loss = network.criterion(predicted, y)
                # update loss and score
                loss += batch_loss.item()
                score += (predicted.argmax(1) ==
                          y).to(dtype=torch.float).sum().item()

        return loss / n_samples, score / n_samples

    def train(self, network: nn.Module):
        '''
        Runs the specified number of epochs for training and validating
        the network module over the given dataset.

        Parameters
        ----------
        network: nn.Module
            Trainable network
        '''
        
        network = network.to(self.device)

        for epoch in range(1, 1 + self.epochs):
            start = time.time()
            training_loss, training_score = self.training_step(network)
            testing_loss, testing_score = self.testing_step(network)
            elapsed = time.time() - start

            if self.verbose:
                self.log(
                    epoch=epoch,
                    time_elapsed=elapsed,
                    training_loss=training_loss,
                    training_score=training_score,
                    testing_loss=testing_loss,
                    testing_score=testing_score
                )

    def log(self, **params):
        '''
        Prints training and validation results over a single epoch run.

        Parameters
        ----------
        params
            Dictionary of parameters
        '''
        
        print('[{}/{}] time elapsed: {:.2f} seconds\n  ├── loss: {:.3f}, score: {:.3f} (training)\n  └── loss: {:.3f}, score: {:.3f} (testing)'.format(
            params['epoch'],
            self.epochs,
            params['time_elapsed'],
            params['training_loss'],
            params['training_score'],
            params['testing_loss'],
            params['testing_score']
        ))
