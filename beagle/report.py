from collections import defaultdict
from .population import Population
from .base import Indicator, ReportBase
from .utils.moea import hypervolume
# External dependencies
import numpy as np
from copy import deepcopy


class Report(ReportBase):
    """
    Class that allows monitoring the evolution of populations over different generations.

    Methods
    -------
    create_report(population: beagle.Population, population_name: str = None, increment_generation: bool = False)
    increment_generation()
    average_fitness(generation: int, population: str, fitness_idx: int)
    total_fitness(generation: int, population: str, fitness_idx: int)
    best_fitness(generation: int, population: str, fitness_idx: int)

    Attributes
    ---------
    report (getter)
    current_generation (getter)
    """
    def create_report(self, population: Population, population_name: str = None, increment_generation: bool = False):
        """
        Method used to create a report on a certain population.

        Parameters
        -----------
        :param population: beagle.Population
            Population on which the report will be created.
        :param population_name: (optional) str
            Name of the population.
        :param increment_generation: (optional) bool
            True to increment the generation, otherwise the generation won't be incremented. The current generation
            can be increment also using the increment_generation() method.
        """

        if population_name is None or not isinstance(population_name, str):
            population_name = f"population_{self._population_id}"

        if population_name in self._report[self._current_generation]:
            raise TypeError('Repeated population_name in the same generation.')

        self._report[self._current_generation][population_name] = deepcopy(population)

        if increment_generation:
            self.increment_generation()
        else:
            self._population_id += 1

    def average_fitness(self, generation: int, population: str, fitness_idx: int) -> float:
        """
        Returns the average fitness in a given generation.

        Parameters
        ----------
        :param generation: int
            Generation.
        :param population: str
            Population name.
        :param fitness_idx: int
            Fitness index.

        Returns
        --------
        :return: float
            Average of the population fitness in the indicated generation.
        """
        total_fitness = self.total_fitness(generation, population, fitness_idx)

        return total_fitness / len(self._report[generation][population])

    def total_fitness(self, generation: int, population: str, fitness_idx: int) -> float:
        """
        Returns the total fitness in a given generation.

        Parameters
        ----------
        :param generation: int
            Generation.
        :param population: str
            Population name.
        :param fitness_idx: int
            Fitness index.

        Returns
        --------
        :return: float
            Total population fitness in the indicated generation.
        """
        selected_pop = self._report[generation][population]

        total_fitness = 0

        for individual in selected_pop:
            total_fitness += individual.fitness[fitness_idx]

        return total_fitness

    def best_fitness(self, generation: int, population: str, fitness_idx: int) -> float:
        """
        Returns the best fitness in a given generation.

        Parameters
        ----------
        :param generation: int
            Generation.
        :param population: str
            Population name.
        :param fitness_idx: int
            Fitness index.

        Returns
        --------
        :return: float
            Best population fitness in the indicated generation.
        """
        all_fitness = [individual.fitness[fitness_idx] for individual in self._report[generation][population]]

        return np.max(all_fitness)

    def std_fitness(self, generation: int, population: str, fitness_idx: int) -> float:
        """
        Returns the std of the fitness in a given generation.

        Parameters
        ----------
        :param generation: int
            Generation.
        :param population: str
            Population name.
        :param fitness_idx: int
            Fitness index.

        Returns
        --------
        :return: float
            Standard deviation of the population fitness in the indicated generation.
        """
        all_fitness = [individual.fitness[fitness_idx] for individual in self._report[generation][population]]

        return float(np.std(all_fitness))


class MOEAReport(ReportBase):
    """DESCRIPTION"""

    def __init__(self, num_objectives: int):
        super(MOEAReport, self).__init__()
        self._num_objectives = num_objectives

    def create_report(self, population: Population, population_name: str = None, increment_generation: bool = False):
        """
        Method used to create a report on a certain population.

        Parameters
        -----------
        :param population: beagle.Population
            Population on which the report will be created.
        :param population_name: (optional) str
            Name of the population.
        :param increment_generation: (optional) bool
            True to increment the generation, otherwise the generation won't be incremented. The current generation
            can be increment also using the increment_generation() method.
        """

        if population_name is None or not isinstance(population_name, str):
            population_name = f"population_{self._population_id}"

        if population_name in self._report[self._current_generation]:
            raise TypeError('Repeated population_name in the same generation.')

        self._report[self._current_generation][population_name] = deepcopy(population)

        if increment_generation:
            self.increment_generation()
        else:
            self._population_id += 1

    def hypervolume(self, generation: int, population: str):
        """DESCRIPTION"""
        return hypervolume(self.pareto_front(generation, population))

    def fitness_values(self, generation: int, population: str):
        """DESCRIPTION"""
        return np.max(self.pareto_front(generation, population), axis=0)

    def pareto_front(self, generation: int, population: str):
        """DESCRIPTION"""
        selected_pop = self._report[generation][population]

        pareto_front_values = np.array([
            individual.fitness[0].values for individual in selected_pop if individual.fitness[0].rank == 0
        ])

        return pareto_front_values

    @property
    def num_objectives(self):
        return self._num_objectives


class EarlyStopping(Indicator):
    """
    Class used to indicate an early stop during the execution of an algorithm.
    """
    def __str__(self):
        return "EarlyStopping"

    def __repr__(self):
        return self.__str__()


class SolutionFound(Indicator):
    """
    Class used to indicate that a solution has been found.
    """
    def __str__(self):
        return "SolutionFound"

    def __repr__(self):
        return self.__str__()


class SaveBestSolution(Indicator):
    """
    Class that allows you to save the best solution based on the fitness value. As individuals can have several fitness value,
    the fitness would be indicated as an index, by default the first value.
    """
    def __init__(self, population: Population, fitness_idx: int = 0):
        solution = SaveBestSolution._get_best(population, fitness_idx)

        super(SaveBestSolution, self).__init__(solution)

    @staticmethod
    def _get_best(population: Population, fitness_idx: int):
        """
        Return the best individual in the population.
        """
        best_fitness = -np.inf
        best_individual = None

        for individual in population:
            try:
                if individual.fitness[fitness_idx] > best_fitness:
                    best_fitness = individual.fitness[fitness_idx]
                    best_individual = individual
            except IndexError:
                raise IndexError('Fitness index provided: %d. Number of fitness values in individual: %d')
            except Exception:
                raise Exception('Exception not identified in the SaveBestSolution indicator.')

        return deepcopy(best_individual)

    def __str__(self):
        return "SaveBestSolution"

    def __repr__(self):
        return self.__str__()
