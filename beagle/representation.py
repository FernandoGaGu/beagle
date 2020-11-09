from .base import Representation as base_Representation
from .exceptions import IncompleteArguments
# External dependencies
import numpy as np
from copy import deepcopy

AVAILABLE_REPR = ['Binary', 'Integer', 'Permutation', 'Real']


def Representation(genotype: str, **kwargs):
    """
    Function that returns a Representation object based on the indicated genotype.

    Call to the function:

        Representation(genotype: str, **kwargs) -> base.Representation


    Available genotypes:

    Parameters (genotype = 'binary')
    ----------
    length: int
        Number of bits.

    For more info help(beagle.Binary)

    Parameters (genotype = 'integer')
    ----------
    value_coding: dict
        Dictionary with the characteristic as a key and the coding value (integer) as a value.

    replacement: bool
        Value indicating if there can be repeated characteristics. By default False.

    For more info help(beagle.Integer)


    Parameters (genotype = 'permutation')
    ----------
        events: list
            List with the different events on which you want to find the optimal order.

        restrictions: list
            List of functions that must take a list of values and return True if the order falls within the range
            of possibilities or False if there is some restriction for that combination.

    For more info help(beagle.Permutation)

    Parameters (genotype = 'real')
    ----------
        bounds: list
            List of tuples indicating the lower and upper limit for each possible value.

    For more info help(beagle.RealValue)


    Returns
    --------
    :return base.Representation
        Subclass of base.representation that will depend on the indicated genotype.
    """
    if genotype.lower() == 'binary':
        if kwargs.get('length', -1) == -1 or not isinstance(kwargs['length'], int):
            raise IncompleteArguments(Binary.args(), str(Binary(-1)))

        return Binary(kwargs.get('length', -1))

    elif genotype.lower() == 'integer':
        if kwargs.get('value_coding', -1) == -1 or not isinstance(kwargs['value_coding'], dict):
            raise IncompleteArguments(Integer.args(), str(Integer({'None': None})))

        return Integer(kwargs.get('value_coding', {'None': None}))

    elif genotype.lower() == 'permutation':
        if kwargs.get('events', -1) == -1 or not isinstance(kwargs['events'], list):
            raise IncompleteArguments(Permutation.args(), str(Permutation([None])))

        return Permutation(kwargs.get('events', [None]), kwargs.get('restrictions', None))

    elif genotype.lower() == 'real':
        if kwargs.get('bounds', -1) == -1 or not isinstance(kwargs['bounds'], list):
            raise IncompleteArguments(RealValue.args(), str(RealValue([(None, None)])))
        if isinstance(kwargs['bounds'], list):
            for e in kwargs['bounds']:
                if not isinstance(e, tuple):  # limits not indicated as tuple
                    raise IncompleteArguments(RealValue.args(), str(RealValue([(None, None)])))
                if len(e) != 2:               # tuple length smaller than 2
                    raise IncompleteArguments(RealValue.args(), str(RealValue([(None, None)])))
                if e[0] >= e[1]:              # lower limit smaller than upper limit.
                    raise IncompleteArguments(RealValue.args(), str(RealValue([(None, None)])))

        return RealValue(kwargs.get('bounds', [(None, None)]))

    else:
        raise TypeError('Genotype representation %s not implemented. Available representations: %s'
                        % (genotype, ', '.join(AVAILABLE_REPR)))


class RealValue(base_Representation):
    """
    Class to represent real-value numbers. Indicated as a list of tuples which indicate the range in which
    each value can be found.
    """

    def __init__(self, bounds: list, initialization: bool = True):
        """
        __init__(bounds: list, initialization: bool = True)

        Parameters
        -----------
        :param bounds: list
            List of tuples indicating the lower and upper limit for each value.

        :param initialization: bool
             Indicates whether the individual will call the initialization function to perform a random initialization.
        """
        self._bounds = bounds
        self._values = []
        if initialization:
            self.initialization()

    def __repr__(self):
        return "real"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index: int):
        return self._values[index]

    def __setitem__(self, index: int, value: float):
        self._values[index] = value

    def __len__(self):
        return len(self._bounds)

    @property
    def kwargs(self):
        return {'bounds': self._bounds}

    @property
    def permissible_values(self):
        """
        Returns a list of tuples indicating the range that each possible value can take.

        Returns
        --------
        :return: list
            List of tuples of length 2.
        """
        return self._bounds

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values: dict):
        self._values = values

    @classmethod
    def args(cls):
        """
        Class method that returns the arguments required for the initialization of the class.

        Returns
        -------
        :return list
            List of tuples indicating the name of the parameter and its type or instructions.
        """
        return [('bounds', 'list of tuples indicating lower and upper limits')]

    def initialization(self):
        """
        Initializes an array of real-values.
        """
        self._values = [np.random.uniform(low=lower, high=upper) for lower, upper in self._bounds]


class Permutation(base_Representation):
    """
    Class for permutation representation. This type of representation allows to find the order in which a sequence
    of events should occur. This class represents the permutation as a fixed set of values represented as integers.
    """

    def __init__(self, events: list, restrictions: list = None, initialization: bool = True):
        """
        __init__(events: list, restrictions: list = None, initialization: bool = True)

        Parameters
        -----------
        :param events: list
            List with the different events on which you want to find the optimal order.

        :param restrictions: list
            List of functions that must take a list of values and return True if the order falls within the range
            of possibilities or False if there is some restriction for that combination.

        :param initialization: bool
             Indicates whether the individual will call the initialization function to perform a random initialization.
        """
        self._events = events

        if restrictions is None:
            self._restrictions = []
        else:
            for restriction in restrictions:
                if not callable(restriction):
                    raise TypeError("restrictions argument must be a list of callable terms.")
            self._restrictions = restrictions  # list of functions that return boolean values

        self._values = []

        if initialization:
            self.initialization()

    def __repr__(self):
        return "permutation"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index: int):
        return self._values[index]

    def __setitem__(self, index: int, value: int):
        self._values[index] = value

    def __len__(self):
        return len(self._events)

    @property
    def kwargs(self):
        return {'events': self._events, 'restrictions': self._restrictions}

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def permissible_values(self):
        """
        Returns the list of possible events of the representation.

        Returns
        --------
        :return: list
            List of possible events.
        """
        return self._events

    @property
    def restrictions(self):
        return self._restrictions

    @values.setter
    def values(self, values):
        self._values = values

    @classmethod
    def args(cls):
        """
        Class method that returns the arguments required for the initialization of the class.

        Returns
        -------
        :return list
            List of tuples indicating the name of the parameter and its type or instructions.
        """
        return [('events', 'list')]

    def is_valid(self, values: list):
        """
        It checks if the order of the values passed as parameters is valid based on the restrictions indicated
        by the user in the constructor.

        Parameters
        ----------
        :param values: list
            List of values.

        Returns
        --------
        :return bool
            True if the order is valid, otherwise False.
        """
        if len(self._restrictions) == 0:
            return True

        for restriction in self._restrictions:
            if not restriction(values):
                return False

        return True

    def initialization(self):
        """
        Initializes a new random order based on permissible values and restrictions.
        """
        new_order = deepcopy(self._events)
        np.random.shuffle(new_order)
        while not self.is_valid(new_order):
            np.random.shuffle(new_order)

        self._values = new_order


class Integer(base_Representation):
    """
    Class for integer representations. Adequate representation for a problem in which we want to look for the best
    subset of variables within a wide range of options. In this representation the genotype consist of and array
    of integer values of variable length.
    """

    def __init__(self, value_coding: dict, replacement: bool = False, initialization: bool = True):
        """
        __init__(value_coding: dict, replacement: bool = False, initialization: bool = True)

        Parameters
        ----------
        :param value_coding: dict
            Dictionary with the characteristic as a key and the coding value (integer) as a value.

        :param replacement: bool
            Value indicating if there can be repeated characteristics. By default False.

        :param initialization: bool
             Indicates whether the individual will call the initialization function to perform a random initialization.

        """
        self._value_coding = value_coding
        self._replacement = replacement
        self._values = []
        if initialization:
            self.initialization()

    def __repr__(self):
        return "integer"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value):
        self._values[index] = value

    def __len__(self):
        return len(self._values)

    @property
    def kwargs(self):
        return {'value_coding': self._value_coding, 'replacement': self._replacement}

    @property
    def permissible_values(self):
        """
        It returns a list with the coding values of the variables in the form of integers.

        Returns
        --------
        :return: list
            List of integer values.
        """
        return list(self._value_coding.values())

    @property
    def replacement(self):
        """
        Returns
        --------
        :return: bool
            True if a variable can be repeated, otherwise False.
        """
        return self._replacement

    @property
    def values(self):
        return self._values

    @property
    def value_coding(self):
        """
        Returns
        --------
        :return: dict
            Dictionary with the variable and its codification.
        """
        return self._value_coding

    @values.setter
    def values(self, values):
        self._values = values

    @classmethod
    def args(cls):
        """
        Class method that returns the arguments required for the initialization of the class.

        Returns
        -------
        :return list
            List of tuples indicating the name of the parameter and its type or instructions.
        """
        return [('value_coding', 'dict')]

    def initialization(self):
        """
        Initializes a random array of variable length using the available values from value_coding used for
        codification.
        """
        self._values = np.random.choice(
            list(self._value_coding.values()),  # possible values
            size=np.random.randint(low=2, high=len(self._value_coding)),  # variable size from 2 to all values
            replace=self._replacement  # Either repeated values are allowed or not
        )


class Binary(base_Representation):
    """
    Class for binary representations. In this type of representation the genotype consists of an array of binary values.
    """

    def __init__(self, length: int, initialization: bool = True):
        """
        __init__(length: int, initialization: bool = True)

        Parameters
        ----------
        :param length: int
            Number of bits.

        :param initialization: bool
             Indicates whether the individual will call the initialization function to perform a random initialization.
        """
        self._length = length
        self._values = []
        if initialization:
            self.initialization()

    def __repr__(self):
        return "binary"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value):
        self._values[index] = value

    def __len__(self):
        return self._length

    @property
    def kwargs(self):
        return {'length': self._length}

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def permissible_values(self):
        return None

    @classmethod
    def args(cls):
        """
        Class method that returns the arguments required for the initialization of the class.

        Returns
        -------
        :return list
            List of tuples indicating the name of the parameter and its type or instructions.
        """
        return [('length', 'int')]

    def initialization(self):
        """
        Initializes a random binary array of fixed length.
        """
        self._values = np.random.randint(low=0, high=2, size=self._length)

