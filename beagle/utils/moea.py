import numpy as np
from collections import defaultdict
from functools import reduce
from ..base import Solution
from ..population import Population


# Fast Non Dominated Sort for MOEAs
def dominance(solution_1: Solution, solution_2: Solution) -> int:
    """
    Function that analyze solutions dominance.

    Parameters
    -----------
    :param solution_1: Solution
    :param solution_2: Solution

    Returns
    ---------
    :return int
        If solution_1 dominates solution_2 -> return 1
    :return -1
        If solution_2 dominates solution_1 -> return -1
    :return 0
        If neither solution dominates the other -> return 0
    """
    dominance_1, dominance_2 = False, False

    for i, value in enumerate(solution_1.values):

        if value > solution_2.values[i]:
            #  Solution 1 at least greater in one value
            dominance_1 = True

        elif value < solution_2.values[i]:
            # Solution 2 at least greater in one value
            dominance_2 = True

    # Solution 1 dominates solution 2
    if dominance_1 and not dominance_2:
        return 1

    # Solution 2 dominates solution 1
    if not dominance_1 and dominance_2:
        return -1

    return 0


def fast_non_dominated_sort(population: Population):
    """
    Apply fast non dominated sort. This function modifies the individuals in the population.

    Parameters
    -----------
    :param population: beagle.Population
    """
    # Initialize an empty front (stores the indices of the individuals in population)
    front = defaultdict(list)

    population_size = len(population)

    for i in range(population_size):
        for j in range(population_size):
            # if both solutions are the same pass to next one
            if i == j: continue

            # Analyze dominance between solutions
            dominates = dominance(population[i].fitness[0], population[j].fitness[0])

            if dominates > 0:  # First solution dominates the other solution
                population[i].fitness[0].dominated_set.append(j)

            elif dominates < 0:  # The other solution dominates the first solution
                population[i].fitness[0].np += 1

        if population[i].fitness[0].np == 0:  # Save only first front
            front[population[i].fitness[0].np].append(i)

    # Get other fronts
    n = 0
    while len(front[n]) != 0:
        for individual_idx in front[n]:
            for dominated_idx in population[individual_idx].fitness[0].dominated_set:

                # Update front
                population[dominated_idx].fitness[0].np -= 1

                # Check if solution is in the next front
                if population[dominated_idx].fitness[0].np == 0:
                    # Add to next front
                    front[n + 1].append(dominated_idx)
        n += 1

        # Select rank attribute of each solution
    for rank, indices in front.items():
        for idx in indices:
            population[idx].fitness[0].rank = rank


# Hypervolume calculation
def hypervolume(pareto_front: np.ndarray):
    """
    Function that calculate hypervolume based on inclusion-exclusion algorithm.

    Parameters
    -----------
    :param pareto_front: list
        List of solutions in Pareto front

    Returns
    ---------
    :return float
        Hypervolume covered by Pareto front
    """

    def vol(solution):
        """
        Function that calculates the volume covered by a solution.

        Parameters
        -----------
        :param solution: Solution

        Returns
        ---------
        :return float
        """
        return reduce(lambda coor1, coor2: coor1 * coor2, solution)

    def vol_intersec(*solutions):
        """
        Function that calculates the volume covered by the intersection between two solutions.

        Parameters
        ------------
        :param solutions: Solution
            One or more solutions.

        Returns
        ---------
        :return float
        """
        return reduce(lambda coor1, coor2: coor1 * coor2, np.min(solutions, axis=0))

    # Sort solution based on one target function
    front = np.sort(pareto_front, axis=0)

    # Calculate intersections between each pair of adjacent solutions
    intersec = [vol_intersec(front[n], front[n + 1]) for n in range(len(front) - 1)]

    # Calculate the volume of each solution and subtract the volumes of the intersections
    return sum(list(map(vol, front))) - sum(intersec)


def pareto_front(moea):
    """
    Method that allows to extract the values of the individuals from a multi-objective genetic algorithm
    of the last generation.

    Parameters
    ----------
    :param moea: beagle.Algorithm
        Multi-objective genetic algorithm.

    Returns
    -------
    :return tuple
        (Indices of individuals in the population, Values of individuals on the non-dominated front)
    """

    last_generation = max(list(moea.report._report.keys()))
    populations = list(moea.report._report[last_generation].keys())

    front = {population: [] for population in populations}
    indices = {population: [] for population in populations}

    for population in populations:
        for i, individual in enumerate(moea.report._report[last_generation][population]):
            if individual.fitness[0].rank == 0 and not list(individual.values) in front[population]:
                front[population].append(list(individual.values))
                indices[population].append(i)

    return indices, front
