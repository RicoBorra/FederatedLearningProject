import os
import sys
import numpy
from typing import Union
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
import tqdm

# relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FederatedMetrics(object):
    '''
    Computes metrics on federated emnist, namely FEMNIST dataset.

    Notes
    -----
    Computed metrics are addressed thorugh following keys.
     - `metrics.results['accuracy']` to get overall accuracy, that is `n_correct / n_total_samples`
     - `metrics.results['weighted_accuracy']` to get weighted accuracy according to all classes, that is `mean(n_class_correct / n_class_samples)`
     - `metrics.results['class_accuracy']` to get a vector of classes accuracies, namely `n_class_correct / n_class_samples`

    Examples
    -----
    >>> metrics = FederatedMetrics(n_classes = 62)
    >>> ... 
    >>> for x, y in dataloader:
    >>>     outputs = model(x)
    >>>     metrics.update(outputs, y)
    >>> ...
    >>> metrics.compute() # compute all metrics
    >>> print(metrics.results) # display dictionary of results
    >>> metrics.reset() # reset computation before next epoch
    '''

    def __init__(
        self, 
        n_classes: int = 62, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        '''
        Constructs the set of metrics for a set of clients to be evaluated.

        Parameters
        ----------
        n_classes: int
            Number of classes within the data (62 by default)
        device: torch.device
            Device on which computation is performed (cuda if available)
        '''
        
        self.n_classes = n_classes
        self.device = device
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes = n_classes).to(device)
        self.results = {
            'accuracy': None,
            'weighted_accuracy': None,
            'class_accuracy': None,
        }

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        '''
        Constructs the set of metrics for a set of clients to be evaluated.

        Parameters
        ----------
        predicted: torch.Tensor
            Tensor [BATCH_SIZE, N_CLASSES] of predicted logits 
            or [BATCH_SIZE] for predicted classes
        device: torch.device
            Tensor [BATCH_SIZE] of target labels
        '''

        self.confusion_matrix.update(predicted, target)

    def reset(self):
        '''
        Resets all metrics' internal states.
        '''
        
        self.confusion_matrix.reset()

        for metric in self.results.keys():
            self.results[metric] = None

    def compute(self):
        '''
        Computes metrics' results using accumulated states through updates.
        '''

        # safe guard used to avoid infinity explosion of empty classes
        epsilon = 1e-6
        # confusion matrix where vertical axis is target, horizontal is predictions
        confusion_matrix = self.confusion_matrix.compute()
        # diagonal are true positives of each class
        n_correctly_predicted = confusion_matrix.diagonal(offset = 0)
        # summing along each row gives the true number of samples of each class
        n_class_samples = confusion_matrix.sum(1) + epsilon
        # total number of samples
        n_samples = confusion_matrix.sum()
        # mask is employed to filter only non empty classes for metrics
        mask = n_class_samples > epsilon
        # class accuracy is a vector of accuracies for each class
        self.results['class_accuracy'] = n_correctly_predicted / n_class_samples
        # this is a unique measure given by the average of the single classes accuracies
        self.results['weighted_accuracy'] = self.results['class_accuracy'][mask].mean()
        # this is a unique measure given by the overall accuracy, not indicative for unbalanced data
        self.results['accuracy'] = n_correctly_predicted.sum() / n_samples
        # loads results for retrieval
        for metric in self.results.keys():
            self.results[metric] = self.results[metric].cpu().detach().numpy()

    def __getitem__(self, metric: str) -> Union[float, numpy.ndarray]:
        '''
        Short hand methods for getting a computed metrics after inkoving compute().

        Parameters
        ----------
        metric: str
            Name of metric, should be `accuracy`, `weighted_accuracy` or `class_accuracy`
        
        Returns
        -------
        Union[float, np.ndarray]
            Metric result
        '''

        return self.results[metric]

class ModelEvaluator(object):
    '''
    Evaluates the model on the entire dataset.
    '''
    
    def __init__(
        self,
        dataloader: DataLoader,
        n_classes: int = 62,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        '''
        Constructs an evaluator using FemnistMetrics metrics.

        Parameters
        ----------
        dataloader: DataLoader
            Data loader
        n_classes: int
            Number of classes used to compute metrics (62 by default)
        device: torch.device
            Computation engine (cuda if available)
        '''
        
        self.metrics = FederatedMetrics(n_classes, device = device)
        self.dataloader = dataloader

    def evaluate(self, model: torch.nn.Module, description: str):
        '''
        Evaluate a model by traversing the whole dataset in batches.

        Parameters
        ----------
        model: torch.nn.Module
            Model to evaluate
        description: str
            Description text displayed on progress bar

        Notes
        -----
        At the end it is possible to retrieve results through `metrics` attribute.
        Besides, state is reinitialized whenever evaluate(...) is invoked.
        '''

        progress = tqdm.tqdm(total = len(self.dataloader))
        progress.set_description(description)
        # reset state
        self.metrics.reset()
        # enable validation
        model.eval()
        # run validation over each image
        with torch.no_grad():
            for step, (x, y) in enumerate(self.dataloader):
                # load data into appropriate device
                x, labels = x.to(self.metrics.device), y.to(self.metrics.device)
                # evaluate on test images
                outputs = model(x)
                # update score metrics
                self.metrics.update(outputs, labels)
                # update progress
                progress.update(1)
        # compute results
        self.metrics.compute()
        # end progress
        progress.close()