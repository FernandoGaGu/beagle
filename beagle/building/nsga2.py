import numpy as np
from ..population import Population, merge_populations
from ..fitness import Fitness, evaluate, evaluate_parallel
from ..base import Solution
from ..utils.moea import fast_non_dominated_sort
from ..selection import uniform_selection, ranking_selection, fitness_proportional_selection, survivor_selection
from ..recombination import recombination
from ..mutation import mutation
from ..report import SaveBestSolution


class NSGA2Solution(Solution):
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
        - crowding_distance: float -> Crowding distance.
        """
        super(NSGA2Solution, self).__init__(values)

        self.dominated_set = []
        self.np = 0
        self.crowding_distance = 0

    def __str__(self):
        return f"NSGA2Solution(values={self.values} rank={self.rank} crowding={self.crowding_distance})"

    def __repr__(self):
        return self.__str__()

    def restart(self):
        """
        Reset all values in the solution.
        """
        self.dominated_set = []
        self.np = 0
        self.rank = None
        self.crowding_distance = 0

    def _crowded_comparision(self, other_sol):
        """
        Comparison operator between two solutions. Based on:

            K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm:
            NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002.

        Crowded comparision operator:
            If we have two solutions with a different Pareto ranking, we choose the one with the lowest value.
            If they have the same ranking we take the one with the highest crowding (the least covered solution).
        """
        # Current solution dominates
        if (self.rank < other_sol.rank) or ((self.rank == other_sol.rank) and
                                            (self.crowding_distance > other_sol.crowding_distance)):
            return 1
        # Both solutions are equal
        elif (self.rank == other_sol.rank) and (self.crowding_distance == other_sol.crowding_distance):
            return 0

        return -1

    def __eq__(self, other_sol):
        """
        Operator ==
        """
        if other_sol == -np.inf:
            return False

        if self._crowded_comparision(other_sol) == 0:
            return True

        return False

    def __lt__(self, other_sol):
        """
        Operator <
        """
        if other_sol == -np.inf:
            return False

        if self._crowded_comparision(other_sol) == -1:
            return True

        return False

    def __ge__(self, other_sol):
        """
        Operator >=
        """
        comparision = self._crowded_comparision(other_sol)

        if comparision == 0 or comparision == 1:
            return True

        return False

    def __gt__(self, other_sol):
        """
        Operator >
        """
        if other_sol == -np.inf:
            return True

        if self._crowded_comparision(other_sol) == 1:
            return True

        return False

    def __le__(self, other_sol):
        """
        Operator <=
        """
        comparision = self._crowded_comparision(other_sol)

        if comparision == 0 or comparision == -1:
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
            individual.fitness = [NSGA2Solution(individual.fitness)]


def nsga2(parents: Population, fitness_function: Fitness, **kwargs):
    """NSGA2 DESCRIPTION"""

    # If the population has not been evaluated for the first time, carry out the evaluation
    if not parents.evaluated:
        if kwargs.get('evaluate_in_parallel', True) is True:
            evaluate_parallel(parents, fitness_function)
        else:
            evaluate(parents, fitness_function)

        # Create NSGA2 solutions
        NSGA2Solution.create_solutions(parents)

        # Fast Non Dominated Sort and calculate crowding
        fast_non_dominated_sort(parents)
        calculate_crowding(parents)

    # Selection (by default tournament selection with k=2, w=1 and replacement=False)
    mating_pool = _SELECTION_OPERATORS[kwargs.get('selection', 'ranking')](
        population=parents, n=len(parents),
        # optional arguments
        **_selection_default_params(**kwargs)
        )

    offspring = recombination(mating_pool, n=len(mating_pool),
                              # optional arguments
                              **_recombination_default_params(**kwargs)
                              )

    mutation(offspring,
             # optional arguments
             **_mutation_default_params(**kwargs)
             )

    # Evaluate offspring
    if kwargs.get('evaluate_in_parallel', True) is True:
        evaluate_parallel(offspring, fitness_function)
    else:
        evaluate(offspring, fitness_function)

    # Create NSGA2 solutions
    NSGA2Solution.create_solutions(offspring)

    # Restart parents solutions
    NSGA2Solution.restart_solutions(mating_pool)

    # Merge populations
    parents_and_offspring = merge_populations(mating_pool, offspring)

    # Fast Non Dominated Sort and calculate crowding
    fast_non_dominated_sort(parents_and_offspring)
    calculate_crowding(parents_and_offspring)

    # Select best solutions (using elitism and selecting the 50% to math original population size)
    next_generation = survivor_selection(parents_and_offspring, schema='elitism', select=0.5, fitness_idx=0)
    next_generation.evaluated = True    # Individuals already have an assigned fitness value

    kwargs['report'].create_report(next_generation, population_name='population', increment_generation=True)

    return next_generation, SaveBestSolution(next_generation)


# Crowding distance for NSGA2
def calculate_crowding(population: Population):
    """
    Method to calculate crowding for all solutions using _crowding_distance() method for each solution.

    Parameters
    -----------
    :param population: beagle.Population
    """

    def crowding_distance(population, indices: list, current: int, objective: int, diff: float):
        """
        Function that calculates the crowding distance (cuboid) for a certain solution.

        Parameters
        ------------
        :param population: beagle.Population
        :param indices: list
            List of the indices associated with the position of individuals in population sorted by objective value.
        :param current: int
            Index of the objective.
        :param objective: int
            Index indicating the objective fitness value.
        :param diff: float
            Difference between max and min value for a given fitness value.
        """
        # If all the solutions are the same crowding is 0
        if diff == 0.0:
            return 0

        # Calculate crowding distance
        distance = (population[indices[current + 1]].fitness[0].values[objective] -
                    population[indices[current - 1]].fitness[0].values[objective]) / diff

        return distance

    # Get the number of objectives
    num_objectives = len(population[0].fitness[0].values)

    for objective in range(num_objectives):

        # Get all fitness values for a given objective
        sorted_indices = list(
            np.argsort([individual.fitness[0].values[objective] for individual in population]))

        # Select limits to infinite
        population[sorted_indices[0]].fitness[0].crowding_distance = -np.inf
        population[sorted_indices[-1]].fitness[0].crowding_distance = np.inf

        # Get max and min values and calculate the difference
        min_obj_value = population[sorted_indices[0]].fitness[0].values[objective]
        max_obj_value = population[sorted_indices[-1]].fitness[0].values[objective]
        diff = max_obj_value - min_obj_value

        # Calculate crowding distance for target function
        for i in range(1, len(sorted_indices) - 1):
            population[sorted_indices[i]].fitness[0].crowding_distance += crowding_distance(population=population,
                                                                                            indices=sorted_indices,
                                                                                            current=i,
                                                                                            objective=objective,
                                                                                            diff=diff)


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
