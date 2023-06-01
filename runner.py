import argparse
import itertools
import os
import subprocess
from typing import Any, Dict, Sequence

class ExperimentRunner(object):
    '''
    An experiment runner acts as a scheduler for multiple runs of the same
    experiment, or many experiments, through a parameters grid.
    '''

    def __init__(self, interpreter: str):
        '''
        Constructs an experiments runner.

        Parameters
        ----------
        interpreter: str
            Command line python interpreter, i.e. '/bin/python'
        '''
        
        self.interpreter = interpreter
        # command of experiments being executed after scheduling
        self.commands = []

    def schedule(self, script: str, grid: Dict[str, Sequence[Any]], enable: bool = True) -> Any:
        '''
        Schedules a set of multiple runs (using the `grid`) of the same experiment.
        
        Parameters
        ----------
        script: str
            Path of python script to be invoked, i.e. `experiments/script.py`
        grid: Dict[str, Sequence[Any]]
            Grid of multiple command line arguments passed to the script for
            multiple executions of the same experiment
        enable: bool
            If `True` (by default) then runs the experiments

        Returns
        -------
        Any
            `ExperimentRunner` instance
        '''
        
        if enable:
            # constructs a command for each script execution with a certain combination of command line parameters
            self.commands.extend([ ExperimentRunner.command(self.interpreter, script, params) for params in ExperimentRunner.expand(grid) ])
        # yields instance
        return self

    def run(self):
        '''
        Runs all the experiments (over multiple grid explorations) scheduled up until now.
        '''

        for i, experiment in enumerate(self.commands):
            print(f'[+] running experiment {i + 1}/{len(self.commands)}')
            print(f'[+] {experiment}')
            # invokes a subprocess whose stdout is connected to current shell
            # and waits for termination
            process = subprocess.Popen(experiment, shell = True)
            process.wait()
            process.kill()


    @staticmethod
    def command(interpreter: str, script: str, params: Dict[str, Any]) -> str:
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
        >>> ExperimentRunner.command('./env/bin/python3', script = 'script.py', params = { 'lr': 0.9, 'niid': True, 'algorithm': ('fedsr', 128, 0.1, 0.01) })
        >>> './env/bin/python3 script.py --lr 0.9 --niid --algorithm fedsr 128 0.1 0.01'
        '''
        
        result = f'{interpreter} {script}'
        # attach command line parameters according to their type
        for name, value in params.items():
            if value is None:
                continue
            elif type(value) is bool:
                if value:
                    result += f' --{name}'
            elif type(value) is tuple:
                result += f" --{name} {' '.join([ str(x) for x in value ])}"
            # NOTE repetition
            elif type(value) is list:
                # repeats argument multiple times
                for subarg in value:
                    if subarg is None:
                        continue
                    elif type(subarg) is bool:
                        if subarg:
                            result += f' --{name}'
                    elif type(subarg) is tuple:
                        result += f" --{name} {' '.join([ str(x) for x in subarg ])}"
                    else:
                        result += f' --{name} {subarg}'
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
    parser = argparse.ArgumentParser(usage = 'run all experiments')
    parser.add_argument('--interpreter', type = str, default = './env/bin/python3', help = 'python interpreter invoked for scripts')
    parser.add_argument('--log', action = 'store_true', default = False, help = 'whether or not to log to weights & biases')
    parser.add_argument('--centralized_base', action = 'store_true', default = False, help = 'run centralized emnist experiments for baseline')
    parser.add_argument('--centralized_dg', action = 'store_true', default = False, help = 'run centralized emnist experiments in domain generalization setting')
    parser.add_argument('--federated_base', action = 'store_true', default = False, help = 'run federated femnist experiments for baseline')
    parser.add_argument('--federated_smart', action = 'store_true', default = False, help = 'run federated femnist experiments using smart client selection')
    parser.add_argument('--federated_opt', action = 'store_true', default = False, help = 'run federated femnist experiments using state of the art algorithms')
    parser.add_argument('--federated_dg', action = 'store_true', default = False, help = 'run federated femnist experiments in domain generalization setting')
    arguments = parser.parse_args()
    # checkpoints directory for state dictionaries
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    # FIXME enable log to True
    runner = ExperimentRunner(interpreter = arguments.interpreter)
    # centralized baseline
    runner.schedule(
        script = 'experiments/centralized_baseline.py',
        grid = {
            'seed': [ 0, 42 ],
            'epochs': [ 10 ],
            'lr': [ 5e-3, 1e-2 ],
            'scheduler': [ ('step', 0.5, 1) ], # decays the learning rate every epoch
            'batch_size': [ 256 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'log': [ arguments.log ]
        },
        enable = arguments.centralized_base
    )
    # centralized domain generalization baseline (rotated domains)
    runner.schedule(
        script = 'experiments/centralized_generalization.py',
        grid = {
            'seed': [ 0 ],
            'validation_domain_angle': [ None, 0, 15, 30, 45, 60, 75 ],
            'epochs': [ 10 ],
            'lr': [ 1e-2 ],
            'scheduler': [ ('step', 0.5, 1) ], # decays the learning rate every epoch
            'batch_size': [ 256 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'log': [ arguments.log ]
        },
        enable = arguments.centralized_dg
    )
    # federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0, 42 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [ 500 ],
            'epochs': [ 1, 5, 10 ],
            'selected': [ 5, 10, 20 ], # with 20 clients crashes on my laptop
            'lr': [ 0.05 ],
            'decay': [ 
                ('lr', 'step', 0.75, 50) # only learning rate parameter is decayed over time (every 50 rounds)
            ],
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ arguments.log ]
        },
        enable = arguments.federated_base
    )
    # smart client selection on federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [5], #[ 500 ],
            'epochs': [ 5 ],
            'selected': [ 10 ],
            'lr': [ 0.05 ],
            'decay': [ 
                ('lr', 'step', 0.75, 50) # only learning rate parameter is decayed over time (every 50 rounds)
            ],
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'selection': [ ('hybrid', 0.5, 0.10), ('hybrid', 0.0001, 0.30) ],
            'log': [ arguments.log ]
        },
        enable = arguments.federated_smart
    )
    # smart client selection with power of choice on federated baseline
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ],
            'model': [ 'cnn' ],
            'rounds': [ 500 ],
            'epochs': [ 5 ],
            'selected': [ 2, 5, 8 ],
            'lr': [ 0.05 ],
            'decay': [ 
                ('lr', 'step', 0.75, 1) # only learning rate parameter is decayed over time (every 50 rounds)
            ],
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 'fedavg' ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'selection': [ ('poc', 10), ('poc', 25) ],
            'log': [ arguments.log ]
        },
        enable = arguments.federated_smart
    )
    # domain generalization in federated setting
    runner.schedule(
        script = 'experiments/federated_generalization.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True, False ], # or only niid ?
            'model': [ 'cnn' ],
            'rounds': [ 500 ],
            'epochs': [ 5 ],
            'selected': [ 10 ],
            'lr': [ 0.05 ],
            'decay': [ 
                [
                    ('lr', 'step', 0.75, 50), # learning rate parameter is decayed over time (every 50 rounds)
                    ('beta_l2', 'step', 2.5, 250), # beta_l2 of fedsr multiplied by 2.5 every 250 rounds
                    ('beta_kl', 'step', 2.5, 250), # beta_kl of fedsr multiplied by 2.5 every 250 rounds
                ]
            ],
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 
                'fedavg', 
                ('fedsr', 128, 1e-1, 1e-1) 
            ],
            'validation_domain_angle': [ None, 0, 15, 30, 45, 60, 75 ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ arguments.log ]
        },
        enable = arguments.federated_dg
    )
    # state of the art algorithms in federated setting
    runner.schedule(
        script = 'experiments/federated_baseline.py',
        grid = {
            'seed': [ 0 ],
            'dataset': [ 'femnist' ],
            'niid': [ True ],
            'model': [ 'cnn' ],
            'rounds': [ 500 ],
            'epochs': [ 5 ],
            'selected': [ 10 ],
            'lr': [ 0.05 ],
            'decay': [ 
                [
                    ('lr', 'step', 0.75, 50), # learning rate decay
                    ('mu', 'step', 2.5, 100), # mu parameter of fedprox increased every 100 rounds
                    ('eta', 'step', 1.25, 250), # eta parameter of fedyogi updated every 250 rounds
                ]
            ],
            'batch_size': [ 64 ],
            'weight_decay': [ 1e-5 ],
            'momentum': [ 0.9 ],
            'algorithm': [ 
                ('fedprox', 1e-3),
                ('fedprox', 1e-2),
                ('fedprox', 1e-1),
                ('fedyogi', 0.9, 0.99, 1e-4, 1e-3), 
                ('fedyogi', 0.9, 0.99, 1e-4, 1e-2), 
                ('fedyogi', 0.9, 0.99, 1e-4, 1e-1) 
            ],
            'evaluation': [ 50 ],
            'evaluators': [ 250 ],
            'log': [ arguments.log ]
        },
        enable = arguments.federated_opt
    )
    # run all experiments
    runner.run()