from abc import ABC, abstractmethod
from typing import Any        

class BaseConfiguration(dict):
    '''
    This basic configuration class acts as a dot access dictionary
    for more confortable notation and usage.

    Examples
    --------
    >>> conf = BaseConfiguration(a = 10)
    >>> # dot access
    >>> conf.a = 1
    >>> # dot access and dictionary access
    >>> conf.b = conf['a'] + 2
    '''

    def __init__(self, **parameters):
        super().__init__(**parameters)
        # defines parameters within the object
        if parameters:
            for k, v in parameters.items():
                self[k] = v

    def __getattr__(self, parameter):
        return self.get(parameter)

    def __setattr__(self, parameter, value):
        self.__setitem__(parameter, value)

    def __setitem__(self, parameter, value):
        super().__setitem__(parameter, value)
        self.__dict__.update({ parameter: value })

    def __delattr__(self, parameter):
        self.__delitem__(parameter)

    def __delitem__(self, parameter):
        super().__delitem__(parameter)
        del self.__dict__[parameter]

    def parameters(self):
        return self.__dict__.keys()


class ParameterDecay(ABC):
    '''
    This class defines a decaying property of some algorithm parameter.
    '''

    def __init__(self, initial_value: float):
        '''
        Initializes the decay of the parameter.

        Parameters
        ----------
        initial_value: float
            Initial value
        '''
        
        super().__init__()
        # initial value of the parameter, remains unchanged
        self.initial_value = initial_value

    @abstractmethod
    def update(self, current_round: int) -> float:
        '''
        Computes updated parameter for `current_round`.

        Parameters
        ----------
        current_round: int
            Current round index

        Returns
        -------
        float
            Updated parameter according to decay rule

        Notes
        -----
        Round index is expected to start from zero.
        '''
        
        raise NotImplementedError
    

class StepDecay(ParameterDecay):
    '''
    This class defines a staircase step decay applied to a parameter
    every `period` of rounds by multiplying the parameter by `factor`.
    '''

    def __init__(self, initial_value: float, factor: float, period: int):
        '''
        Initializes the decay of the parameter as staircase step function.

        Parameters
        ----------
        initial_value: float
            Initial value
        factor: float
            Multiplicative factor every `period` of rounds
        period: int
            Number of rounds among each step update
        '''

        super().__init__(initial_value)
        # step decay properties
        self.factor = factor
        self.period = period

    def update(self, current_round: int) -> float:
        '''
        Computes updated parameter for `current_round`.

        Parameters
        ----------
        current_round: int
            Current round index

        Returns
        -------
        float
            Updated parameter according to decay rule

        Notes
        -----
        Round index is expected to start from zero.
        '''

        # staircase decays every 'period' of rounds
        return self.initial_value * self.factor ** (current_round // self.period)
    
class ExponentialDecay(ParameterDecay):
    '''
    Equivalent to `StepDecay` but with `period` equal to a single round.
    '''

    def __init__(self, initial_value: float, factor: float):
        '''
        Initializes the decay of the parameter as exponential function.

        Parameters
        ----------
        initial_value: float
            Initial value
        factor: float
            Multiplicative factor every single round
        '''

        super().__init__(initial_value)
        # decay property
        self.factor = factor

    def update(self, current_round: int) -> float:
        '''
        Computes updated parameter for `current_round`.

        Parameters
        ----------
        current_round: int
            Current round index

        Returns
        -------
        float
            Updated parameter according to decay rule

        Notes
        -----
        Round index is expected to start from zero.
        '''

        # decays of same factor exponentially each round
        return self.initial_value * self.factor ** current_round

class LinearDecay(ParameterDecay):
    '''
    This decay follows a linear function going from `initial_value` to `final_value`
    of the parameter in exactly `period` of rounds. After such `period` the final value
    is returned (like a ramp).
    '''

    def __init__(self, initial_value: float, final_value: float, period: int):
        '''
        Initializes the decay of the parameter as staircase step function.

        Parameters
        ----------
        initial_value: float
            Initial value
        final_value: float
            Final value of parameter, reached after `period` of rounds
        period: int
            Number of rounds for which linear interpolation is applied before
            asyntotic `final_value`
        '''

        super().__init__(initial_value)
        # decay properties
        self.final_value = final_value
        self.period = period

    def update(self, current_round: int) -> float:
        '''
        Computes updated parameter for `current_round`.

        Parameters
        ----------
        current_round: int
            Current round index

        Returns
        -------
        float
            Updated parameter according to decay rule

        Notes
        -----
        Round index is expected to start from zero.
        '''

        # when current round has overcome the decaying period, then the asyntotic final value is yield
        if current_round > self.period:
            return self.final_value
        # otherwise linear interpolation with respect to current round
        return self.initial_value + current_round * (self.final_value - self.initial_value) / self.period

class AlgorithmConfiguration(BaseConfiguration):
    '''
    This class holds the configuration of (hyper)parameters of a federated learning algorithm.

    Notes
    -----
    It allows dynamic decay of parameters, for instance learning rate.

    Examples
    --------
    In this example we create a dynamic (with decay) configuration for some algorithm.

    >>> config = AlgorithmConfiguration(
    >>>     name = 'dynamic-algorithm-config', # name of algorithm 
    >>>     lmbda = 1.0 # some constant (i.e. not decaying) parameter
    >>> )
    >>> # halves learning rate every 10 rounds
    >>> config.decay('lr', StepDecay(initial_value = 0.01, factor = 0.5, period = 10))
    >>> # linear interpolation of mu value increasing over time
    >>> config.decay('mu', LinearDecay(initial_value = 0.01, final_value = 1.0, period = 50))
    >>> # constant parameter
    >>> config.define('v', 0.045)
    >>> # initialize algorithm with configuration
    >>> algorithm = FancyFedAlgorithm(state, config)
    '''

    def __init__(self, **parameters):
        '''
        Constructs a configuration of (hyper)parameters for some
        federated scenario algorithm.

        Parameters
        ----------
        parameters
            Parameter of the configuration
        '''
        
        super().__init__(**parameters)
        # current round initialized from zero
        self.current_round = 0
        # keeps track of those parameters decaying over multiple rounds of central execution
        self.parameter_decay: dict[str, ParameterDecay] = {}

    def define(self, parameter: str, value: Any) -> Any:
        '''
        Defines a constant parameter of the configuration.

        Parameters
        ----------
        parameter: str
            Name of parameter
        value: Any
            Initial value

        Returns
        -------
        Any
           `AlgorithmConfiguration` instance 
        '''
        
        # forbids redefinition
        if parameter in self.parameters() and self[parameter] != value:
            raise RuntimeError(f'[*] reinitialization of parameter \'{parameter}\' with different value')
        # forbids redefinition when already defined as decaying parameter
        if parameter in self.parameter_decay and self.parameter_decay[parameter].initial_value != value:
            raise RuntimeError(f'[*] inconsistency in initialization of decaying parameter \'{parameter}\', conflicting values \'{value}\' and \'{self.parameter_decay[parameter].initial_value}\'')
        # defines new parameter with initial value
        self[parameter] = value
        
        return self

    def decay(self, parameter: str, decay: ParameterDecay) -> Any:
        '''
        Defines a decaying parameter of the configuration.

        Parameters
        ----------
        parameter: str
            Name of parameter
        decay: ParameterDecay
            Decay of the parameter

        Returns
        -------
        Any
           `AlgorithmConfiguration` instance 
        '''

        # forbids redefinition when already defined as decaying parameter
        if parameter in self.parameters() and decay.initial_value != self[parameter]:
            raise RuntimeError(f'[*] inconsistency in initialization of decaying parameter \'{parameter}\', conflicting values \'{self[parameter]}\' and \'{decay.initial_value}\'')
        # defines new parameter with initial value and relative decaying property
        self[parameter] = decay.initial_value
        self.parameter_decay[parameter] = decay

        return self

    def update(self):
        '''
        Updates internal state, i.e. advances to next round (and decays parameters).
        '''

        # advances from current round
        self.current_round = self.current_round + 1
        # decays all parameters subject to some kind of decaying
        for parameter, decay in self.parameter_decay.items():
            self[parameter] = decay.update(current_round = self.current_round)