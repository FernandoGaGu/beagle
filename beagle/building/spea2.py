import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from copy import deepcopy
from ..population import Population, merge_populations
from ..fitness import Fitness, evaluate, evaluate_parallel
from ..base import Solution
from ..utils.moea import fast_non_dominated_sort
from ..selection import uniform_selection, ranking_selection, fitness_proportional_selection, survivor_selection
from ..recombination import recombination
from ..mutation import mutation
from ..report import SaveBestSolution
from ..utils.auxiliary_functions import static_vars


class SPEA2Solution(Solution):
    """
    Class that represents a possible solution with the values of each of the objective functions.
    """

    def __init__(self, values: list):
        """

            __init__(values)

        Notes
        -------
        - values: 1d-array -> List of target function values.
        - dominated_set: list(Solution) -> Solutions dominated by the current solution.
        - np: int -> Number of times this solution is dominated.
        - rank: int -> Indicates which front the current solution is on.
        - raw_fitness -> Number of solutions by which the solution is dominated.
        - distances -> Distances from other solutions.
        """
        super(SPEA2Solution, self).__init__(values)

        self.dominated_set = []
        self.np = 0
        self.fitness = 0
        self.raw_fitness = 0
        self.distances = []

    def __str__(self):
        return f"Solution(rank={self.rank} fitness={self.fitness} raw_fitness={self.raw_fitness} distances={self.distances})"

    def __repr__(self):
        return self.__str__()

    def select_density(self, k):
        """
        Return fitness using density estimation to k-th neighbor. Based on

            "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
             Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
             Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
             and Lothar Thiele"

        """
        self.fitness = self.raw_fitness + (1 / (self.distances[k] + 2))

    def restart(self):
        """
        Reset all values in the solution.
        """
        self.dominated_set = []
        self.np = 0
        self.fitness = 0
        self.rank = None
        self.raw_fitness = 0
        self.distances = []

    def __eq__(self, other_sol):
        """
        Operator ==
        """
        if other_sol == -np.inf:
            return False

        if self.fitness == other_sol.fitness:
            return True

        return False

    def __lt__(self, other_sol):
        """
        Operator <
        """
        if other_sol == -np.inf:
            return False

        if self.fitness < other_sol.fitness:
            return True

        return False

    def __ge__(self, other_sol):
        """
        Operator >=
        """
        if self.fitness >= other_sol.fitness:
            return True

        return False

    def __gt__(self, other_sol):
        """
        Operator >
        """
        if other_sol == -np.inf:
            return True

        if self.fitness > other_sol.fitness:
            return True

        return False

    def __le__(self, other_sol):
        """
        Operator <
        """
        if self.fitness < other_sol.fitness:
            return True

        return False

    def __ne__(self, other_sol):
        """
        Operator not
        """

        return not self.__eq__(other_sol)

    @staticmethod
    def restart_solutions(population: Population):
        """
        Function that resets the values of the solutions by applying the restart() method.

        Parameters
        ----------
        :param population: beagle.Population
        """
        for individual in population:
            individual.fitness[0].restart()

    @staticmethod
    def create_solutions(population: Population):
        """
        Function that transforms fitness values (at least 2 or more values) into NSGA2Solution objects.

        Parameters
        ----------
        :param population: beagle.Population
        """
        for individual in population:
            individual.fitness = [SPEA2Solution(individual.fitness)]


@static_vars(k=None, distance=None, archive_length=None)
def spea2(parents: Population or dict, fitness_function: Fitness, **kwargs):
    """SPEA2 DESCRIPTION"""

    # SPEA2 arguments (Only first iteration)
    if spea2.k is None:
        spea2.k = kwargs.get('spea2_k', int(np.sqrt(len(parents))))
    if spea2.distance is None:
        spea2.distance = kwargs.get('spea2_distance', distance.euclidean)
    if spea2.archive_length is None:
        spea2.archive_length = kwargs.get('spea2_archive', len(parents))

    if isinstance(parents, dict):
        population_length = len(parents['next_population'])
        if kwargs.get('evaluate_in_parallel', True) is True:
            # parents[0] = parents; parents[1] = archive
            evaluate_parallel(parents['next_population'], fitness_function)
        else:
            # First iteration
            evaluate(parents['next_population'], fitness_function)

        # Create SPEA2 solutions
        SPEA2Solution.create_solutions(parents['next_population'])
        # Restart archive solutions
        SPEA2Solution.restart_solutions(parents['archive'])
        # Merge parents and archive populations
        parents = merge_populations(parents['next_population'], parents['archive'])
    else:
        population_length = len(parents)
        if kwargs.get('evaluate_in_parallel', True) is True:
            # parents[0] = parents; parents[1] = archive
            evaluate_parallel(parents, fitness_function)
        else:
            # First iteration
            evaluate(parents, fitness_function)

        # Create SPEA2 solutions
        SPEA2Solution.create_solutions(parents)

    fast_non_dominated_sort(parents)

    fitness_assignment(parents)

    density_estimation(parents, spea2.k, spea2.distance)

    archive = archive_selection(parents, spea2.archive_length)

    # Selection (by default tournament selection with k=2, w=1 and replacement=False)
    mating_pool = _SELECTION_OPERATORS[kwargs.get('selection', 'ranking')](
        population=archive, n=population_length,
        # optional arguments
        **_selection_default_params(**kwargs)
    )

    offspring = recombination(mating_pool, n=population_length,
                              # optional arguments
                              **_recombination_default_params(**kwargs)
                              )

    mutation(offspring,
             # optional arguments
             **_mutation_default_params(**kwargs)
             )

    # Annotate archive values
    kwargs['report'].create_report(archive, population_name='archive', increment_generation=True)

    return {'next_population': offspring, 'archive': archive}, SaveBestSolution(archive)


def fitness_assignment(population: Population):
    """
    SPEA2 fitness assignment based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param population: beagle.Population
        This population must be previously evaluated and the individual's fitness values must consist of
        objects from SPEA2Solution class.
    """
    for individual in population:

        strength_value = len(individual.fitness[0].dominated_set)

        for dom_individual in individual.fitness[0].dominated_set:
            population[dom_individual].fitness[0].raw_fitness += strength_value


def density_estimation(population, k: int, distance: callable):
    """
    SPEA2 density estimation based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param population: beagle.Population
        This population must be previously evaluated and the individual's fitness values must consist of
        objects from SPEA2Solution class.
    :param distance: function
        Metric to evaluate the distance between two vectors.
    :param k: int
        Neighbor from which the fitness density of the solution is calculated.
    """
    population_length = len(population)

    distances = np.zeros(shape=(population_length, population_length))

    for i in range(population_length):
        for j in range(population_length):
            distances[i, j] = distance(population[i].fitness[0].values, population[j].fitness[0].values)

    distances = np.sort(distances, axis=1)

    for i in range(distances.shape[0]):
        population[i].fitness[0].distances = distances[i, 1:]     # The first distance is from a solution to itself
        population[i].fitness[0].select_density(k)


def shorter_distance(sol_1, sol_2, num_neighbors: int, k: int = 0):
    """
        Function that returns true if the first solution has a shorter distance or
        both solutions are the same. Otherwise it will returns false.
        """
    # If they are both the same distance, go to the next neighbor
    if sol_1.distances[k] == sol_2.distances[k]:
        k += 1

        #  To stop recursion when the two solutions are the same
        if k == num_neighbors:
            return False

        return shorter_distance(sol_1, sol_2, num_neighbors, k)

    # Solution 1 has a shorter distance
    elif sol_1.distances[k] < sol_2.distances[k]:
        return True

    else:
        return False


def selection_sort(indices: list, population, nn: int):
    # Traverse through all array elements
    for i in range(len(indices)):

        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, len(indices)):
            if shorter_distance(
                    population[indices[min_idx]].fitness[0], population[indices[j]].fitness[0], nn):
                min_idx = j

        # Swap the found minimum element with the first element
        indices[i], indices[min_idx] = indices[min_idx], indices[i]


def archive_selection(population: Population, archive_length: int):
    """
    SPEA2 archive selection based on:

        "SPEA2: Improving the Strength Pareto Evolutionary Algorithm for Multiobjective Optimization January 2001
         Conference: Evolutionary Methods for Design, Optimization and Control with Applications to Industrial
         Problems. Proceedings of the EUROGEN'2001. Athens. Greece, September 19-21. Eckart Zitzler, Marco Laumanns
         and Lothar Thiele"

    Parameters
    ------------
    :param population: beagle.Population
    :param archive_length: int

    Returns
    ---------
    :return: list
        List of solutions that make up the archive.
    """
    front = defaultdict(list)

    # Get fronts
    for i in range(len(population)):
        front[population[i].fitness[0].rank].append(i)

    front_length = len(front[0])

    if front_length == archive_length:
        #  Archive length and non-dominated front match
        return Population(
            size=archive_length, representation=population.representation,
            individuals=[deepcopy(population[i]) for i in front[0]])

    elif front_length < archive_length:
        #  The non-dominated front is not enough to fill the archive
        num_front = 1
        num_individuals = front_length
        archive_filled = True
        # Get the number of fronts needed to cover the length of the archive
        while num_individuals < archive_length:
            num_individuals += len(front[num_front])
            num_front += 1
            # When all solutions are not capable of filling the archive it will be the same as the population
            if len(front[num_front]) == 0:
                archive_filled = False
                break

        # Get the indices of the individuals that will form the archive
        individual_indices = [idx for n in range(num_front) for idx in front[n]]

        return Population(
            size=archive_length if archive_filled else len(individual_indices), representation=population.representation,
            individuals=[deepcopy(population[i]) for i in individual_indices[:archive_length]])

    else:
        #  The non-dominated front exceeds the length of the archive, apply truncation operator
        #  The individual with the sorthest distance will be removed at each iteration until fill the archive
        num_neighbors = len(population[0].fitness[0].distances)

        # Sort front
        selection_sort(front[0], population, num_neighbors)

        return Population(
            size=archive_length, representation=population.representation,
            individuals=[deepcopy(population[i]) for i in front[0][:archive_length]])


# Operators default arguments
def _selection_default_params(**kwargs):
    params = {
        'schema': kwargs.get('selection_schema', 'tournament'),
        'idx_fitness': kwargs.get('selection_idx_fitness', '_d'),
        'replacement': kwargs.get('selection_replacement', '_d'),
        'k': kwargs.get('selection_k', '_d'),
        'w': kwargs.get('selection_w', '_d'),
        'rank_schema': kwargs.get('selection_rank_schema', '_d'),
        'fitness_idx': kwargs.get('selection_fitness_idx', '_d')
    }

    return params


def _recombination_default_params(**kwargs):
    params = {
        'schema': kwargs.get('recombination', None),
        'cut_points': kwargs.get('recombination_cut_points', '_d'),
        'probability': kwargs.get('recombination_probability_uc', '_d'),
        'ari_type': kwargs.get('recombination_ari_type', '_d'),
        'alpha': kwargs.get('recombination_alpha', '_d')
    }

    return params


def _mutation_default_params(**kwargs):
    params = {
        'schema': kwargs.get('mutation', None),
        'probability': kwargs.get('mutation_probability', '_d'),
        'max_mutation_events': kwargs.get('mutation_max_mutation_events', '_d'),
        'distribution': kwargs.get('mutation_distribution', '_d'),
        'std': kwargs.get('mutation_std', '_d'),
        'std_idx': kwargs.get('mutation_std_idx', '_d'),
        'sigma_threshold': kwargs.get('mutation_sigma_threshold', '_d'),
        'possible_events': kwargs.get('mutation_possible_events', '_d'),
        'max_attempts': kwargs.get('mutation_max_attempts', '_d'),
        'tau': kwargs.get('mutation_tau', '_d')
    }

    return params


_SELECTION_OPERATORS = {
    'uniform': uniform_selection,
    'ranking': ranking_selection,
    'proportional': fitness_proportional_selection
}
