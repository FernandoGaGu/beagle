import numpy as np
from copy import deepcopy
from .representation import Representation
from .base import Representation as BaseRepresentation
from .exceptions import InconsistentLengths


class Individual:
    """
    Class representing an individual in the population. This individual contains a representation which corresponds
    to a coding of a potential solution, a set of fitness values and an age.

    Methods
    -------
    - increment_age()
    - evaluated()

    Attributes
    ----------
    - kwargs            (getter)
    - representation    (getter)
    - values            (getter / setter)
    - fitness           (getter / setter)
    - age               (getter)
    """
    def __init__(self, genotype: str, **kwargs):
        """
        __init__(genotype: str, **kwargs)

        Parameters
        -----------
        :param genotype: str
            Genotype representation. Available: 'binary', 'integer', 'real', 'permutation'.
        :param kwargs:
            Necessary arguments for the construction of the representation. For more information use
            help(beagle.Representation).
        """
        if kwargs.get('representation', None) is None:
            representation = Representation(genotype=genotype, **kwargs)
        else:
            representation = kwargs['representation']
            if not isinstance(representation, BaseRepresentation):
                raise TypeError(
                    'representation for individual creation parameter must be a base.Representation instance.'
                )

        self._representation = representation
        self._fitness = []
        self._age = 0

    def __str__(self):
        return 'Individual(representation=%s)' % self._representation

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._representation)

    def __getitem__(self, index):
        return self._representation[index]

    def __setitem__(self, index, value):
        """
        Setter that allows to modify the values of the representation.
        """
        self._representation[index] = value

    def increment_age(self):
        """
        Method to increment the age of the individual.
        """
        self._age += 1

    def evaluated(self):
        """
        Method that indicates whether an individual has been evaluated or not.

        Returns
        -------
        :return bool
            True if the individual has been evaluated, otherwise False.
        """
        return True if len(self._fitness) != 0 else False

    def copy(self):
        """
        Method that returns a deepcopy of the individual.

        Returns
        -------
        :return Individual
            Deepcopy of the individual.
        """
        copy_individual = Individual(genotype=str(self._representation), **self.kwargs)
        copy_individual.values = deepcopy(self.values)
        copy_individual.fitness = self._fitness
        copy_individual._age = self._age

        return copy_individual

    @property
    def kwargs(self):
        """
        Getter of the arguments required for the initialization of the representation

        Returns
        --------
        :return: dict
            Arguments used for the initialization of the genotype representation.
        """
        return self._representation.kwargs

    @property
    def representation(self):
        """
        Getter of the representation

        Returns
        --------
        :return: str
            Representation used to codify the genotype of the individual.
        """
        return self._representation

    @property
    def values(self):
        """
        Getter of the values of the representation.

        Returns
        --------
        :return:
            Individual's genotype.
        """
        return self._representation.values

    @values.setter
    def values(self, values):
        """
        Setter of the values of the representation.
        """
        self._representation.values = values

    @property
    def fitness(self):
        """
        Getter of the individual's fitness values.

        Returns
        ---------
        :return: list
            Individual's fitness values.
        """
        return self._fitness

    @fitness.setter
    def fitness(self, new_fitness):
        """
        Setter of the individual's fitness.

        Parameters
        -----------
        :param new_fitness: list
            New fitness values.
        """
        self._fitness = new_fitness

    @property
    def age(self):
        """
        Getter of the individual's age.

        Returns
        ---------
        :return: int
            Individual's age.
        """
        return self._age


class Population:
    """
    Class representing a population of individuals.

    Available representations:

    Parameters (representation = 'binary')
    ----------
    length: int
        Number of bits.

    For more info help(beagle.Binary)

    Parameters (representation = 'integer')
    ----------
    value_coding: dict
        Dictionary with the characteristic as a key and the coding value (integer) as a value.

    replacement: bool
        Value indicating if there can be repeated characteristics. By default False.

    For more info help(beagle.Integer)


    Parameters (representation = 'permutation')
    ----------
        events: list
            List with the different events on which you want to find the optimal order.

        restrictions: list
            List of functions that must take a list of values and return True if the order falls within the range
            of possibilities or False if there is some restriction for that combination.

    For more info help(beagle.Permutation)

    Parameters (representation = 'real')
    ----------
        bounds: list
            List of tuples indicating the lower and upper limit for each possible value.

    For more info help(beagle.RealValue)

    Methods
    -------
    - increment_age(): Increment the age of the individuals in the population.
    - sort(indices): Sort the individuals in the population based on a list of indices.
    - replace_older_individuals(): Replace older individuals.

    Attributes
    ----------
    - representation (getter)
    - evaluated (getter / setter)
    """
    def __init__(self, size: int, representation: str, **kwargs):
        """
         __init__(size: int, representation: str, **kwargs)

         Parameters
         -----------
        :param size: int
            Population size.
        :param representation: str
            Representation of individuals in population. Available representations: 'binary', 'integer', 'real',
            'permutation'.
        :param individuals: (optional) list of beagle.Individuals
            Array of individuals from which a new population will be created. If individuals are not provided they
            will be randomly initialized.
        :param kwargs:
            Necessary arguments for the construction of the representation. For more information use
            help(beagle.Representation).
        """
        if kwargs.get('individuals', None) is None:
            individuals = [Individual(representation, **kwargs) for n in range(size)]
        else:
            individuals = kwargs['individuals']
            for indv in individuals:
                if not isinstance(indv, Individual):
                    raise TypeError('Individuals must belong to Individual class.')

        self._individuals = individuals
        self._representation = representation.lower()
        self._size = size
        self._evaluated = False
        self.__current = 0  # iterator

    def __str__(self):
        return f"Population(size={self._size}; representation={self._representation})"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index: int):
        return self._individuals[index]

    def __setitem__(self, idx: int, value: Individual):
        self._individuals[idx] = value

    def __len__(self):
        return self._size

    def __iter__(self):
        self.__current = 0
        return self

    def __next__(self) -> Individual:
        self.__current += 1
        if self.__current <= self._size:
            return self._individuals[self.__current-1]
        raise StopIteration

    def increment_age(self):
        """
        Method for increment the age of each individual in population.
        """
        for individual in self._individuals:
            individual.increment_age()

    def sort(self, indices):
        """
        Function that sort the individuals in population based on the indices.

        Parameters
        ----------
        :param indices: list or numpy.array
             Indices indicating the new position in population of each individual.
        """
        if len(indices) != len(self._individuals): raise InconsistentLengths('Impossible to sort the population.')

        new_order = [None] * len(self._individuals)

        i = 0
        for idx in indices:
            new_order[i] = self._individuals[idx]
            i += 1

        self._individuals = new_order

    def replace_older_individuals(self, *, age_threshold: int = 2, new_individuals: list = None):
        """
        Function that replaces individuals older than (>=) the indicated age with random individuals (indicating an age
        threshold) or with a new population. In the latter case the older individuals will be replaced without
        considering the age threshold.

        Parameters
        ----------
        age_threshold: int (default 2)
            Age limit from which individuals will be replaced with random population. Default 2.
        new_individuals: Population (default None)
            Individuals used to replace older individuals in the population. If this argument is provided the age
            limit will not be considered.
        """

        if new_individuals is None:
            for individual in self._individuals:
                if individual.age >= age_threshold:
                    individual.representation.initialization()                      # Restart values
                    individual._age = 0                                             # Restart age
        else:
            individual_ages = [individual.age for individual in self._individuals]
            sorted_ages = np.argsort(individual_ages)[::-1]                         # Ordered from highest to lowest

            for new_idx, older_idx in enumerate(sorted_ages):
                if new_idx == len(new_individuals): break

                self._individuals[older_idx] = deepcopy(new_individuals[new_idx])  # Replace older individual

    @property
    def representation(self):
        """
        Return the representation used to represent the genotype of the individuals in population.

        Returns
        :return: str
            Genotype representation.
        """
        return self._representation

    @property
    def evaluated(self):
        return self._evaluated

    @evaluated.setter
    def evaluated(self, is_evaluated: bool):
        self._evaluated = is_evaluated


def merge_populations(*populations) -> Population:
    """
    Function that receives a variable number of populations and brings them together in a new population.
    This function performs a deepcopy of the populations received as argument.

    Parameters
    -----------
    :param populations: beagle.Population
        Variable number of populations to merge.

    Returns
    --------
    :return: beagle.Population
        New population.
    """
    individuals = [individual.copy() for population in populations for individual in population]

    return Population(size=len(individuals), representation=populations[0].representation, individuals=individuals)
