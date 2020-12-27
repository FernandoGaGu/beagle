import multiprocessing as mp
from .population import Population, Individual
from .exceptions import FitnessFunctionError


CPUS = mp.cpu_count()


class Fitness:
    """
    Class that includes the set of fitness functions used to evaluate the quality of each individual based
    on their phenotype by assigning a value.
    """
    def __init__(self, *functions):
        """
         __init__(*functions)

        :param functions: callable
            Variable number of functions that receive input values (corresponding to the values of each individual)
            and generate a real value (corresponding to the quality of the individual based on his or her phenotype).
        """
        for func in functions:
            if not callable(func):
                raise TypeError('Fitness functions must be callable objects.')

        self._functions = functions

    def __len__(self):
        return len(self._functions)

    def __call__(self, individual: Individual) -> list:
        """
        Evaluates an individual's fitness based on their values.

        Parameters
        ----------
        :param individual: beagle.Individual
            Individual from population.

        Returns
        -------
        :return list
            List of real values corresponding to the fitness values of an beagle.Individual.

        """
        return [f_function(individual.values) for f_function in self._functions]


def evaluate(population: Population, fitness_function: Fitness):  
    """
    Function that takes as parameters a population and a fitness instance and assigns the fitness to each individual
    according to the objective functions defined by the user in the creation of the fitness instance passed as a
    parameter.

    Parameters
    ----------
    :param population: beagle.Population
        Population of possible solutions.
    :param fitness_function: beagle.Fitness
        Fitness instance with the objective functions to be maximized.
    """

    for individual in population:
        try:
            individual.fitness = fitness_function(individual)
        except Exception:
            raise FitnessFunctionError(individual.values)

    population.evaluated = True


def evaluate_parallel(population: Population, fitness_function: Fitness):
    """
    Similar evaluate but implemented in parallel.

    Parameters
    ----------
    :param population: beagle.Population
        Population of possible solutions.
    :param fitness_function: beagle.Fitness
        Fitness instance with the objective functions to be maximized.
    """
    with mp.Pool(CPUS) as p:
        fitness_values = p.map(fitness_function, [individual for individual in population])

    for i, fitness in enumerate(fitness_values):
        population[i].fitness = fitness

    population.evaluated = True
