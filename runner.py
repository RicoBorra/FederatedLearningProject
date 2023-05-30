import itertools
import os
import subprocess
from typing import Any, Dict, Sequence

class ExperimentRunner(object):
    '''
    An experiment runner acts as a scheduler for multiple runs of the same
    experiment, or many experiments, through a parameters grid.
    '''

    def __init__(self):
        '''
        Constructs an experiments runner.
        '''
        
        # command of experiments being executed after scheduling
        self.commands = []

    def schedule(self, script: str, grid: Dict[str, Sequence[Any]]) -> Any:
        '''
        Schedules a set of multiple runs (using the `grid`) of the same experiment.
        
        Parameters
        ----------
        script: str
            Path of python script to be invoked, i.e. `experiments/script.py`
        grid: Dict[str, Sequence[Any]]
            Grid of multiple command line arguments passed to the script for
            multiple executions of the same experiment

        Returns
        -------
        Any
            `ExperimentRunner` instance
        '''
        
        # constructs a command for each script execution with a certain combination of command line parameters
        self.commands.extend([ ExperimentRunner.command(script, params) for params in ExperimentRunner.expand(grid) ])
        return self

    def run(self):
        '''
        Runs all the experiments (over multiple grid explorations) scheduled up until now.
        '''

        for i, experiment in enumerate(self.commands):
            print(f'[+] running experiment {i + 1}/{len(self.commands)}')
            # invokes a subprocess whose stdout is connected to current shell
            # and waits for termination
            process = subprocess.Popen(experiment, shell = True)
            process.wait()
            process.kill()


    @staticmethod
    def command(script: str, params: Dict[str, Any]) -> str:
        '''
        Constructs a shell command for executing python script within the environment.

        Parameters
        ----------
        script: str
            Path of python script to be invoked
        params: Dict[str, Any]
            Combination of parameters

        Returns
        -------
        str
            Constructed command

        Examples
        --------
        >>> ExperimentRunner.command(script = 'script.py', params = { 'lr': 0.9, 'niid': True, 'algorithm': ('fedsr', 128, 0.1, 0.01) })
        >>> './env/bin/python3 script.py --lr 0.9 --niid --algorithm fedsr 128 0.1 0.01'
        '''
        
        result = f'./env/bin/python3 {script}'
        # attach command line parameters according to their type
        for name, value in params.items():
            if value is None:
                continue
            elif type(value) is bool:
                if value:
                    result += f' --{name}'
            elif type(value) in [tuple, list]:
                result += f" --{name} {' '.join([ str(x) for x in value ])}"
            else:
                result += f' --{name} {value}'
        # yields constructed shell command
        return result

    @staticmethod
    def expand(grid: Dict[str, Sequence[Any]]) -> Sequence[Dict[str, Any]]:
        '''
        Flattens a parameters grid by constructing each single parameters combination.

        Parameters
        ----------
        grid: Dict[str, Sequence[Any]]
            Parameters grid

        Returns
        -------
        Sequence[Dict[str, Any]]
            List of each combination of parameters

        Examples
        --------
        >>> grid = { 'lr': [ 1e-3, 1e-2 ], 'tau': [ 0.9, 0.99 ], 'seed': [ 42 ] }
        >>> ExperimentRunner.expand(grid)
        >>> [
        >>>     { 'lr': 1e-3, 'tau': 0.9, 'seed': 42 }, 
        >>>     { 'lr': 1e-3, 'tau': 0.99, 'seed': 42 }, 
        >>>     { 'lr': 1e-2, 'tau': 0.9, 'seed': 42 }, 
        >>>     { 'lr': 1e-2, 'tau': 0.99, 'seed': 42 } 
        >>> ]
        '''
        
        return list(dict(zip(grid, x)) for x in itertools.product(*grid.values()))

if __name__ == '__main__':
    # checkpoints directory for state dictionaries
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    # FIXME enable log to True
    runner = ExperimentRunner()
    # federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [ 1000 ],
            'epochs': [ 1, 5, 10, 20 ],
            'selected': [ 5, 10 ], # with 20 clients crashes on my laptop
            'learning_rate': [ 0.05 ],
            'scheduler': [ ('step', 0.5, 50) ], # halves the learning rate every 50 central rounds
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ False ]
        }
    )
    # smart client selection on federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [ 1000 ],
            'epochs': [ 5 ],
            'selected': [ 10 ],
            'learning_rate': [ 0.05 ],
            'scheduler': [ ('step', 0.5, 50) ], # halves the learning rate every 50 central rounds
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'selection': [ ('hybrid', 0.5, 0.10), ('hybrid', 0.0001, 0.30) ],
            'log': [ False ]
        }
    )
    # smart client selection with power of choice on federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [ 1000 ],
            'epochs': [ 5 ],
            'selected': [ 2, 5, 8 ],
            'learning_rate': [ 0.05 ],
            'scheduler': [ ('step', 0.5, 50) ], # halves the learning rate every 50 central rounds
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'selection': [ ('poc', 10), ('poc', 25) ],
            'log': [ False ]
        }
    )
    # domain generalization in federated setting
    runner.schedule(
        script = 'experiments/federated_generalization.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ], # or only niid ?
            'model': [ 'cnn' ],
            'rounds': [ 1000 ],
            'epochs': [ 5 ],
            'selected': [ 10 ],
            'learning_rate': [ 0.05 ],
            'scheduler': [ ('step', 0.5, 50) ], # halves the learning rate every 50 central rounds
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg', ('fedsr', 128, 1e-1, 1e-1) ],
            'validation_domain_angle': [ None, 0, 15, 30, 45, 60, 75 ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ False ]
        }
    )
    # state of the art algorithms in federated setting
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True ],
            'model': [ 'cnn' ],
            'rounds': [ 1000 ],
            'epochs': [ 5, 10 ],
            'selected': [ 5, 10 ],
            'learning_rate': [ 0.05 ],
            'scheduler': [ ('step', 0.5, 50) ], # halves the learning rate every 50 central rounds
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ ('fedprox', 1e-1), ('fedyogi', 0.9, 0.99, 1e-4, 10 ** (-2.5)) ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ False ]
        }
    )
    # run all experiments
    runner.run()