import numpy as np
import time
from tqdm import tqdm
from copy import deepcopy
from .fitness import Fitness, evaluate
from .population import Population
from .exceptions import IncompleteArguments, StepFunctionError
from .report import EarlyStopping, SolutionFound, SaveBestSolution, Report


class Algorithm:
    """
    A class used for the creation of algorithms that implement elements defined within the
    framework or defined by the user.

    Methods
    -------
    - run(generations: int, random_seed: int = None)

    Attributes
    ----------
    - report (getter)
    - solution (getter)
    - id (getter)
    - population (getter)
    """
    def __init__(self, step, fitness: Fitness, initial_population: Population = None,
                    alg_id: str = str(time.time()), **kwargs):
        """
         __init__(step, fitness: Fitness, initial_population: Population = None, **kwargs)

         Parameters
         ----------
        :param step: callable
             User-defined function indicating the steps carried out by the algorithm in each generation.
             These steps can include elements defined in the framework or directly implemented by the user.
             This function must return a tuple, the second argument can be a subclass of an indicator object 
             or if no None indicators are used. On the other hand, the first argument should be the population 
             of individuals that will make up the next generation. 
        :param fitness: beagle.Fitness
            Instance used to evaluate the fitness of individuals in the population. For more info help(beagle.Fitness)
        :param initial_population: (optional) beagle.Population
            Initial population, if this population is not passed as an argument, a population size can be indicated by
            the population_size (int) argument, also indicating the type of representation that the individuals in the
            population must have by means of the individual_representation (str) parameter and the arguments required
            for the indicated representation.
        :param alg_id: (optional) str
            Algorithm id.
        :param kwargs: dict
            Parameters used by the algorithm in the step function and/or for the initialization of the algorithm itself.

        """
        if not callable(step): raise TypeError('step argument must be callable')
        if not isinstance(fitness, Fitness): raise TypeError('fitness argument must be a fitness object')

        _kwargs_consistency(step, 'step', **kwargs)   # check if arguments in step are repeated in kwargs arguments

        if initial_population is None:
            _check_population_initialization(**kwargs)
            initial_population = Population(
                size=kwargs['population_size'], representation=kwargs['individual_representation'], **kwargs)

        self._evaluate_out_of_step = kwargs.get('evaluate_out_of_step', True)
        self._parents = initial_population
        self._step = step
        self._fitness = fitness
        self._id = alg_id
        self._kwargs = kwargs
        self._solutions = []

    def run(self, generations: int, random_seed: int = None):
        """
        It executes the step function defined by the user and passed by argument during the number of generations
        indicated and optionally establishing a random seed in order to maintain the reproducibility of the algorithm.
        If no seed is provided this will be None.

        Parameters
        ----------
        :param generations: int
            Number of generations (> 1).

        :param random_seed: (optional) int
            Random seed.
        """
        if generations <= 0: raise TypeError('The number of generations must be an integer greater than 1.')
        if random_seed is not None: np.random.seed(random_seed)

        if self._evaluate_out_of_step:
            evaluate(self._parents, self._fitness)  # evaluate initial population

        for n in tqdm(range(generations), desc='(%s) Generations ' % self._id):
            try:
                self._kwargs['current_generation'] = n
                next_gen_population, indicator = self._step(self._parents, self._fitness, **self._kwargs)
            except ValueError:
                raise StepFunctionError(
                    'The step function should return a tuple of two values, the first corresponding to the '
                    'population that will pass to the next generation and the second an object of the class '
                    'Indicator or None if no indicators are being used.')

            if isinstance(indicator, EarlyStopping):    # Early stopping
                print('Early stopping in generation %d of %d' % (n, generations))
                self._solutions.append(indicator.solution)
                break

            elif isinstance(indicator, SolutionFound) or isinstance(indicator, SaveBestSolution):  # Save solution
                self._solutions.append(indicator.solution)
            
            self._parents = next_gen_population

        return self

    @property
    def report(self) -> Report or None:
        """If report argument has been provided in **kwargs during initialization return it, otherwise return None"""
        if 'report' in self._kwargs:
            return self._kwargs['report']

        return None

    @property
    def solution_found(self) -> bool:
        if len(self._solutions) == 0:
            return False

        return True

    def solutions(self, fitness_idx: int = 0, only_best: bool = False) -> list:
        """
        Method that returns all the solutions found (useful in combination with the SaveBestSolution indicator defined
        in the step function) or if the only_best parameter is specified it will return the best solution.

        Parameters
        ----------
        :param fitness_idx: int (default 0)
            Fitness value on the basis of which to choose the best solution, useful for when more than one fitness
            function has been applied.
        :param only_best: bool (default False)
            Returns only the best solution.

        Returns
        -------
        :return list
            Best solution(s).
        """
        # If no solutions have been saved, a list with the best solution from the current population returns.
        if self.solution_found and only_best:
            return [get_best_solution(self, fitness_idx)]

        return self._solutions

    @property
    def id(self) -> str:
        """
        Get algorithm id.

        Parameters
        ----------
        :return str
        """
        return self._id

    @property
    def population(self) -> Population or tuple:
        """
        Return the population used by the algorithm.

        Returns
        -------
        :return beagle.Population
        """
        return self._parents


def get_best_solution(algorithm: Algorithm, fitness_idx: int = 0):
    """
    It returns the best solution found in all the solutions stored in the Algorithm. As one solution can have several
    fitness values, an index can be specified to indicate which fitness value to use for comparing the solutions.
    By default it will be the first fitness value.

    Parameters
    ----------
    :param algorithm: be.Algorithm
        Algorithm.
    :param fitness_idx: int (default 0)
        Index of fitness value.

    Returns
    -------
    :return Individual / None
        Best solution or None if no solutions have been saved.
    """
    best_fitness = -np.inf
    best_solution = None

    for solution in algorithm._solutions:
        try:
            if solution.fitness[fitness_idx] > best_fitness:
                best_fitness = solution.fitness[fitness_idx]
                best_solution = solution
        except IndexError:
            raise IndexError('Fitness index provided: %d. Number of fitness values in individual: %d')
        except Exception:
            raise Exception('Exception not identified in the SaveBestSolution indicator.')

    return deepcopy(best_solution)


def _check_population_initialization(**arguments):
    """
    Check if the arguments provided can be used for the initialization of the population.
    """
    if arguments.get('population_size', None) is None or not isinstance(arguments.get('population_size', None), int):
        raise IncompleteArguments([('population_size: int', 'population size')], 'Algorithm')

    if arguments.get('individual_representation', None) is None or \
            not isinstance(arguments.get('individual_representation', None), str):
        raise IncompleteArguments([('individual_representation: str', 'representation of individuals')], 'Algorithm')


def _kwargs_consistency(function, function_name: str, **kwargs):
    """
    Function that checks if the name of the variables required in the function are repeated in kwargs arguments.
    """
    if not callable(function): raise TypeError('function in _kwargs_consistency must be a callable object.')

    for f_arg in function.__code__.co_varnames:
        if kwargs.get(f_arg, None) is not None:
            raise TypeError('argument %s in %s is repeated in **kwargs' % (f_arg, function_name))
