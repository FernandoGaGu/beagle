# External dependencies
from abc import ABCMeta, abstractmethod
from collections import defaultdict


class Representation:
    """
    Base class for all Representations of possible solutions. Abstract methods that a representation must implement:

        __getitem__: Returns the value of the representation indicated by the index.

        __setitem__: Change the value indicated by the index.

        __len__: Returns the length of the representation.

        initialization: Randomly initializes the values of a representation.

        values (property): Getter for values.

        permissible_values (property): Return "something" depending of the representation type, that indicates the
            range of permissible values that a representation could take.

        kwargs (property): Returns a dictionary with the parameters used for the construction of the representation.

        args (class method): It returns a list of tuples with the values required for a certain representation and
            its type.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __getitem__(self, index: int): raise NotImplementedError

    @abstractmethod
    def __setitem__(self, index: int, value): raise NotImplementedError

    @abstractmethod
    def __len__(self): raise NotImplementedError

    @abstractmethod
    def initialization(self): raise NotImplementedError

    @property
    @abstractmethod
    def values(self): raise NotImplementedError

    @property
    @abstractmethod
    def permissible_values(self): raise NotImplementedError

    @property
    @abstractmethod
    def kwargs(self): raise NotImplementedError

    @classmethod
    @abstractmethod
    def args(cls): raise NotImplementedError


class ReportBase:
    """
    Class that allows monitoring the evolution of populations over different generations.
    """
    def __init__(self):
        """
        __init__()
        """
        self._report = defaultdict(dict)
        self._current_generation = 1
        self._population_id = 0

    def create_report(self, population, population_name: str = None, increment_generation: bool = False):
        raise NotImplementedError

    def increment_generation(self):
        """
        Method used to increase the current generation of the report.
        """
        self._current_generation += 1
        self._population_id = 0

    @property
    def report(self):
        """
        Return the report.

        Returns
        --------
        :return: dict
            Dictionary with the generations as keys and the populations and their corresponding fitness values as
            values.
        """
        return self._report

    @property
    def current_generation(self):
        """
        Returns the current generation.

        Returns
        :return: int
            Current generation.
        """
        return self._current_generation


class Indicator:
    """
    Base class used for all those indicators that are used to indicate situations such as early stop or that a solution 
    has been found.

    Attributes
    ----------
    - solution  (getter)
    """
    __metaclass__ = ABCMeta

    def __init__(self, solution):
        self._solution = solution

    @property
    def solution(self): 
        return self._solution


class Solution:
    """
    Class representing a solution. Solution subclasses can be used as fitness value.
    """
    __metaclass__ = ABCMeta

    def __init__(self, values):
        self._values = values
        self.rank = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        self._values = new_values

    @abstractmethod
    def restart(self): raise NotImplementedError

    @abstractmethod
    def __eq__(self, solution): raise NotImplementedError

    @abstractmethod
    def __ge__(self, solution): raise NotImplementedError

    @abstractmethod
    def __gt__(self, solution): raise NotImplementedError

    @abstractmethod
    def __le__(self, solution): raise NotImplementedError

    @abstractmethod
    def __lt__(self, solution): raise NotImplementedError

    @abstractmethod
    def __ne__(self, solution): raise NotImplementedError


