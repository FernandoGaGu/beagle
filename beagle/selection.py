from .utils.auxiliary_functions import static_vars
from .exceptions import UnrecognisedParameter
from .population import Population, Individual
from .fitness import Fitness
# External dependencies
import numpy as np
from functools import reduce
from copy import deepcopy


# -- SELECTION OPERATORS -- #
def uniform_selection(population: Population, n: int, schema: str = None, **kwargs) -> Population:
    """
    Selection mechanism in which all individuals have the same chance to be selected. It is not a highly recommended
    operator, only used in very specific situations.

    Parameters
    ----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param schema: str
        Implemented for compatibility.
    :param kwargs:
        Implemented for compatibility.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """
    individuals = [None] * n
    current_member = 0

    while current_member < n:
        idx_order = np.random.choice(len(population), replace=False, size=len(population))
        for idx in idx_order:
            individuals[current_member] = population[idx]
            current_member += 1

            if current_member == n:
                break

    individuals = [deepcopy(individual) for individual in individuals]  # Make a deepcopy of each selected individual

    return Population(size=n, representation=population.representation, individuals=individuals)


def ranking_selection(population: Population, n: int, schema: str = None, **kwargs) -> Population:
    """
    Rank-based selection. This schema sort the population on the basis of their fitness values selecting the individuals
    according to the individual's rank. This operator preserves a constant selection pressure. In this way individuals
    with greater fitness values are more likely to be selected.
    The mapping from rank number to selection probability can be done in several ways, for example linearly or
    exponentially decreasing:

        Lineal:   P = (2 - s) / mu + 2i * (s -1) / mu * (mu - 1)

            where s must be in the interval (1, 2], by default 2.

        Exponential: P = (1 - e^-i) / c

    This probability is used when using selection operators:

        Stochastic Universal Sampling       help(beagle.SELECTION_SCHEMAS['stochastic_universal_sampling])
        Roulette wheel                      help(beagle.SELECTION_SCHEMAS['roulette_wheel])

    If tournament selection is used [help(beagle.SELECTION_SCHEMAS['tournament_selection])] these probabilities
    are not used.

    Parameters
    ----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param schema: (optional, default 'sigma_scaling') str
         Selection schema.
    :param kwargs:
        Additional arguments depending on the selection schema followed.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """
    if schema is None: schema = 'tournament'

    fitness_idx = kwargs.get('fitness_idx', 0)

    # Default arguments
    fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx

    if schema == 'tournament':
        k = kwargs.get('k', 2)
        w = kwargs.get('w', 1)
        replacement = kwargs.get('replacement', False)

        # Default arguments
        k = _k if k == '_d' else k
        w = _w if w == '_d' else w
        replacement = _replacement if replacement == '_d' else replacement

        individuals = _tournament_selection(population, n, fitness_idx, k, w, replacement)

    elif schema == 'sus':
        rank_schema = kwargs.get('rank_schema', 'lineal')

        # Default argument
        rank_schema = _rank_schema if rank_schema == '_d' else rank_schema

        prob_distribution = _cumulative_probs(population, rank_schema, **kwargs)
        individuals = _stochastic_universal_sampling(population, prob_distribution, n)

    elif schema == 'roulette':
        rank_schema = kwargs.get('rank_schema', 'lineal')

        # Default argument
        rank_schema = _rank_schema if rank_schema == '_d' else rank_schema

        prob_distribution = _cumulative_probs(population, rank_schema, **kwargs)
        individuals = _roulette_wheel(population, prob_distribution, n)

    else:
        raise UnrecognisedParameter(schema, 'schema')

    return Population(size=n, representation=population.representation, individuals=individuals)


def fitness_proportional_selection(population: Population, n: int, schema: str = None, **kwargs) -> Population:
    """
    In this operator the probability that an individual i was selected depends of its absolute fitness value. This
    operator has the following drawbacks:
        - Premature convergence
        - No selection pressure then all fitness values are very close.
        - Different behaviour if the fitness function is transposed.

    Therefore the following procedures have been implemented in addition to the typical selection scheme.
        - fitness proportional selection.       For more info use: help(beagle.SELECTION_SCHEMAS['fps'])
        - win-dowing.                           For more info use: help(beagle.SELECTION_SCHEMAS['win_dowing'])
        - sigma scaling (default operator).     For more info use: help(beagle.SELECTION_SCHEMAS['sigma_scaling'])

    Parameters
    ----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param schema: (optional, default 'sigma_scaling') str
         Selection schema.
    :param kwargs:
        Additional arguments depending on the selection schema followed.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """
    if schema is None: schema = 'sigma_scaling'

    if schema == 'fps':
        individuals = _fps(population, n, **kwargs)
    elif schema == 'win_dowing':
        individuals = _win_dowing(population, n, **kwargs)
    elif schema == 'sigma_scaling':
        individuals = _sigma_scaling(population, n, **kwargs)
    else:
        raise UnrecognisedParameter(schema, 'schema')

    return Population(size=n, representation=population.representation, individuals=individuals)


def survivor_selection(population: Population, schema: str = None, **kwargs) -> Population:
    """
    Survivor selection operator. These operators include strategies of selection and annihilation of the population
    based on their fitness values. Unlike the other operators, they receive two populations, the parents and the
    offspring, and apply the selection considering the two populations together, or optionally using only one of
    the two populations (specified by apply_to param).

    Available schemes:

        Annihilation: Replace the worst individuals in population. For more info:

                help(beagle.SELECTION_SCHEMES['annihilation')

        Elitism: Select the best individuals in population: For more info:

                help(beagle.SELECTION_SCHEMES['elitism')

        Round Robin Tournament: Select he best individuals in population based on a schema analogous to tournament
        selection.

                help(beagle.SELECTION_SCHEMES['round_robin_tournament'])


    Parameters
    ----------
    :param population: beagle.Population
        Target population
    :param schema: (optional, default 'round_robin') str
         Selection schema. Available: 'annihilation', 'elitism' and 'round_robin'.
    :param kwargs:
        Additional arguments depending on the selection schema followed.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """

    if schema is None: schema = 'round_robin'

    if schema == 'annihilation':
        
        if kwargs.get('annihilate', None) is None:
            raise TypeError('Survivor selection with annihilation schema needs a "annihilate" argument indicating the '
                            'percentage of individuals in target population to be removed.')

        return _annihilation(population, kwargs['annihilate'], kwargs.get('fitness_idx', 0))

    elif schema == 'elitism':
        if kwargs.get('select', None) is None:
            raise TypeError('Survivor selection with elitism schema needs a "select" argument indicating the '
                            'percentage of individuals in target population to be selected.')

        return _elitism(population, kwargs['select'], kwargs.get('fitness_idx', 0))

    elif schema == 'round_robin':
        size = kwargs.get('size', len(population))
        q = kwargs.get('q', 10)
        fitness_idx = kwargs.get('fitness_idx', 0)

        # Default argument
        size = len(population) if size == '_d' else size
        q = _q if q == '_d' else q
        fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx

        return _round_robin_tournament(population, size, q, fitness_idx)

    else:
        raise UnrecognisedParameter(schema, 'schema')


# -- Functions used in survivor_selection -- #
def _elitism(population: Population, n: float, fitness_idx: int):
    """
    Operator who selects the best individuals in a deterministic way from their fitness values returning a new
    population.

    Parameters
    ----------
    :param population: beagle.Population
        Target population.
    :param n: int
        Elite size.
    :param fitness_idx: int
        Fitness value to be considered.

    Returns
    -------
    :return: beagle.Population
        Elite population.
    """
    elite_length = int(len(population) * n)

    elite = _select_elite(population, elite_length, fitness_idx)

    elite = [deepcopy(individual) for individual in elite]  # Make a deepcopy of each selected individual

    return Population(size=elite_length, representation=population.representation, individuals=elite)


def _annihilation(population: Population, n: float, fitness_idx: int):
    """
    Annihilation operator. This operator can lead to rapid convergence issues, for this reason it is usually combined
    with large populations and/or no duplicates policy. In this case the offspring population is modified.

    Parameters
    ----------
    :param population: beagle.Population
        Target population.
    :param n: float
        Percentage of individuals to be removed.
    :param fitness_idx: int
        Fitness value to be considered.

    Returns
    -------
    :return: beagle.Population
        Offspring population in which the worst individuals have been replaced based on how the user has specified
        using the best_perc and random_perc parameters.
    """

    # Default arguments
    fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx

    num_individuals_to_replace = int(len(population) * n)

    new_individuals = np.argsort(np.array([i.fitness[fitness_idx] for i in population]))[num_individuals_to_replace:]
    
    new_individuals = [deepcopy(population[idx]) for idx in new_individuals]

    return Population(
        size=len(population) - num_individuals_to_replace, representation=population.representation, individuals=new_individuals)


def _round_robin_tournament(population: Population, n: int, q: int, fitness_idx: int) -> Population:
    """
    Each individual is evaluated against q others randomly chosen from the population. The n individuals with more wins
    are returned.

    Parameters
    ----------
    :param population: beagle.Population
        Target population.
    :param n: int
        New population size.
    :param q: int
        Number of individuals sampled in each tournament, analogous to k value in tournament selection.
    :param fitness_idx: int
        Fitness value to be considered.

    Returns
    -------
    :return: beagle.Population
        Elite population.
    """
    def win(individual_1: Individual, individual_2: Individual, fitness_idx_: int):
        """Return True if individual_1 is more fitted than individual_2"""
        if individual_1.fitness[fitness_idx_] > individual_2.fitness[fitness_idx_]:
            return True

        return False

    length_population = len(population)
    individual_wins = {}

    for i, individual in enumerate(population):
        random_idx = np.random.choice(length_population, replace=False, size=q)

        individual_wins[i] = 0

        for idx in random_idx:
            if win(individual, population[idx], fitness_idx):
                individual_wins[i] += 1

    winners_idx = np.argsort(list(individual_wins.values()))[::-1][:n]

    return Population(size=n, representation=population.representation,
                      individuals=[deepcopy(population[i]) for i in winners_idx])


# -- Functions used in ranking_selection -- #
def _roulette_wheel(parents: Population, prob_distribution: list, n: int):
    """
    With this operator individuals with a higher fitness value will be more likely to be selected compared to
    those with a lower fitness value.

    Parameters
    ----------
    :param parents: beagle.Population
        Population from which n individuals are going to be selected.
    :param prob_distribution: list
        Cumulative probability distribution.
    :param n: int
        Length of the selected population.

    Returns
    -------
    :return: list of beagle.Individual
        Selected individuals.
    """
    current_member = 0
    mating_pool = [None] * n

    while current_member < n:
        random_num = np.random.uniform()

        i = 0
        while prob_distribution[i] < random_num:
            i += 1

        mating_pool[current_member] = parents[i]
        current_member += 1

    mating_pool = [deepcopy(individual) for individual in mating_pool]  # Make a deepcopy of each selected individual

    return mating_pool


def _tournament_selection(parents: Population, n: int, fitness_idx: int = 0, k: int = 2, w: int = 1,
                          replacement: bool = False):
    """
    Tournament selection is an operator that doesn't require any global knowledge of the population, nor a quantifiable
    measure of quality. In this case the larger the tournament size the greater the chance it will contain members of
    above-average fitness.

    Parameters
    ----------
    :param parents: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Length of the selected population.
    :param fitness_idx: (optional, default 0)
        Fitness value to be considered. By default 0.
    :param k: (optional, default 2)
        Individuals to be selected in each tournament.
    :param w: (optional, default 1)
        Number of winners in each tournament.
    :param replacement: (optional, default False) bool
        Indicates if the individuals selected for each tournament are sampled with or without replacement.

    Returns
    -------
    :return: list of beagle.Individual
        Selected individuals.
    """

    def select_best(individuals: list, i: int):
        """
        Select the best individual assuming a maximization problem.
        """
        max_fitness = -np.inf
        best_idx = -1

        for idx, individual in enumerate(individuals):
            if individual.fitness[i] > max_fitness:
                best_idx = idx
                max_fitness = individual.fitness[i]

        best_individual = individuals[best_idx]
        del individuals[best_idx]

        return best_individual

    mating_pool = [None] * n
    length_population = len(parents)

    current_member = 0
    while current_member < n:

        k_individuals = [parents[i] for i in np.random.choice(length_population, size=k, replace=replacement)]

        i = 0
        while i < w:
            mating_pool[current_member] = select_best(k_individuals, fitness_idx)
            current_member += 1

            if current_member == n:
                break

            i += 1
    mating_pool = [deepcopy(individual) for individual in mating_pool]  # Make a deepcopy of each selected individual

    return mating_pool


def _stochastic_universal_sampling(parents: Population, prob_distribution: list, n: int):
    """
    Stochastic universal sampling (SUS) algorithm. Whenever more than one sample is to be drawn from the distribution
    the use of the stochastic universal sampling algorithm is preferred compared to roulette wheel algorithm.

    Parameters
    ----------
    :param parents: beagle.Population
        Population from which n individuals are going to be selected.
    :param prob_distribution: list
        Cumulative probability distribution.
    :param n: int
        Length of the selected population.

    Returns
    -------
    :return: list of beagle.Individual
        Selected individuals.

    Exceptions
    -----------
    :raise Exception
        If the algorithm enters an infinite loop because random_num is greater than 1 an exception will occur.
    """
    current_member, i = 0, 0
    mating_pool = [None] * n

    random_num = np.random.uniform(low=0, high=(1/n))

    while current_member < n:

        while random_num <= prob_distribution[i]:

            mating_pool[current_member] = parents[i]
            random_num += 1 / n
            current_member += 1

            if random_num > 1:
                raise Exception(
                    'The SUS algorithm has entered an infinite loop. Verify that the selected population '
                    'sizes are suitable for this type of operator.')

        i += 1

    mating_pool = [deepcopy(individual) for individual in mating_pool]  # Make a deepcopy of each selected individual

    return mating_pool


# -- Functions used in fitness_proportional_selection -- #
@static_vars(least_fitted_vals=[])
def _win_dowing(population: Population, n: int, **kwargs):
    """
    This operator tries to mitigate two of the main problems of fitness proportional selection, that the operator
    behaves differently if the fitness function is transposed and to avoid the loss of selective pressure. Under this
    scheme the fitness differentials are maintained by subtracted from the raw fitness of each individual a value beta
    which depends on the recent search history. The simplest approach consist on subtracting the value of the
    least-fitted individual or the average over the last few generations (specified by num_generations param).

    Parameters
    -----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param num_generations: (optional, default 1) int
        Number of generations over which to consider averaging the worst fitness value. By default the last generation.
    :param fitness_idx: (optional, default 0) int
        Fitness value to consider. Value that only matters when each individual has more than one fitness value..
        By default 0.
    :param reset: (optional, default False) bool
        It resets the stored fitness values of previous generations.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """

    num_generations = kwargs.get('num_generations', 1)
    fitness_idx = kwargs.get('fitness_idx', 0)

    # Default arguments
    num_generations = _num_generations if num_generations == '_d' else num_generations
    fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx

    if kwargs.get('reset', False):
        _win_dowing.least_fitted_vals = []

    fitness_values = [indv.fitness[fitness_idx] for indv in population]

    # calculate average over the last num_generations of the least-fit member
    _win_dowing.least_fitted_vals.append(np.min(fitness_values))
    beta = np.mean(_win_dowing.least_fitted_vals[::-1][:num_generations])

    total_fitness = 0
    for val in fitness_values:
        total_fitness += (val - beta)

    individuals = [None] * n
    idx = 0

    while idx < n:
        for indv in population:
            if (indv.fitness[fitness_idx] - beta) / total_fitness >= np.random.uniform():  # add individual
                individuals[idx] = indv
                idx += 1
                if idx == n: break

    individuals = [deepcopy(individual) for individual in individuals]  # Make a deepcopy of each selected individual

    return individuals


def _fps(population: Population, n: int, **kwargs):
    """
    Basic scheme of fitness proportional selection . In this case, individuals with greater fitness are more likely
    to be selected.

    Parameters
    -----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param fitness_idx: (optional, default 0) int
        Fitness value to consider. Value that only matters when each individual has more than one fitness value..
        By default 0.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """
    fitness_idx = kwargs.get('fitness_idx', 0)

    # Default arguments
    fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx

    total_fitness = reduce(lambda a, b: a + b, [indv.fitness[fitness_idx] for indv in population])
    individuals = [None] * n
    idx = 0

    while idx < n:
        for indv in population:
            if indv.fitness[fitness_idx] / total_fitness >= np.random.uniform():  # add individual
                individuals[idx] = indv
                idx += 1
                if idx == n: break

    individuals = [deepcopy(individual) for individual in individuals]  # Make a deepcopy of each selected individual

    return individuals


def _sigma_scaling(population: Population, n: int, **kwargs):
    """
    This schema of fitness proportional selection includes information about the mean and the standard deviation of
    the population fitness. In this way, the fitness value of each individual is calculated using the following
    expression:

                    f'(Xi) = max(f(Xi) - ( mean(population_fitness) - c * std(population_fitness) ), 0)

    where Xi represents an individual and the c constant is usually set to 2.


    Parameters
    -----------
    :param population: beagle.Population
        Population from which n individuals are going to be selected.
    :param n: int
        Number of individuals to be selected.
    :param fitness_idx: (optional, default 0) int
        Fitness value to consider. Value that only matters when each individual has more than one fitness value..
        By default 0.
    :param c: (optional, default 2) float
        Constant value. By default 2.

    Returns
    -------
    :return beagle.Population
        Population of size n.
    """
    c = kwargs.get('c', 2.0)
    fitness_idx = kwargs.get('fitness_idx', 0)

    # Default arguments
    fitness_idx = _fitness_idx if fitness_idx == '_d' else fitness_idx
    c = _c if c == '_d' else c

    fitness_values = [indv.fitness[fitness_idx] for indv in population]
    sigma = np.std(fitness_values)
    mean = np.mean(fitness_values)
    individuals = [None] * n
    idx = 0

    while idx < n:
        for indv in population:
            if np.max([indv.fitness[fitness_idx] - (mean - c * sigma), 0]) >= np.random.uniform():  # add individual
                individuals[idx] = indv
                idx += 1
                if idx == n: break

    individuals = [deepcopy(individual) for individual in individuals]  # Make a deepcopy of each selected individual

    return individuals


# -- HELPING FUNCTIONS -- #
def _cumulative_probs(pop: Population, rank: str, fitness_idx: int = 0, **kwargs):
    """
    Function that calculates the cumulative probabilities of each individual according to their fitness value.

    Parameters
    -----------
    :param pop: beagle.Population
        Population.
    :param rank: str
        Available:

        - 'lineal':  P = (2 - s) / mu + 2i * (s -1) / mu * (mu - 1)

        - 'exponential':  P = (1 - e^-i) / c

    :param fitness_idx: (optional, default 0)
    :param s: (optional, default 2) float
        Parameter used in lineal ranking. This parameter must be grater than 1 and less or equal to 2. By default 2.

    Returns
    --------
    :return: list
        Array of cumulative probabilities.
    """
    def lineal_cumulative_probs(mu: int, s: float):

        return np.array([(2 - s) / mu + 2 * i * (s - 1) / (mu * (mu - 1)) for i in range(mu)])

    def exponential_cumulative_probs(mu: int):
        probs_ = [1 - np.exp(-i) for i in range(mu)]

        return np.array(probs_) / np.sum(probs_)  # scaling to sum up 1

    def calculate_cumulative(probability_dist: list):
        accumulated = 0
        for i in range(len(probability_dist)):
            probability_dist[i] += accumulated
            accumulated = probability_dist[i]

        return probability_dist

    rank_indices = np.argsort([i.fitness[fitness_idx] for i in pop])

    pop.sort(rank_indices)   # Sort population according to the individual's fitness value

    if rank == 'lineal':
        s = kwargs.get('s', 2)

        # Default argument
        s = _s if s == '_d' else s

        if not 1 < s <= 2:
            raise TypeError('Parameter s for ranking selection must be: 1 < s <= 2.')

        probs = lineal_cumulative_probs(len(pop), s)

    elif rank == 'exponential':
        probs = exponential_cumulative_probs(len(pop))

    else:
        raise UnrecognisedParameter(rank, 'rank_schema')

    return calculate_cumulative(probs[rank_indices].tolist())


def _select_elite(population: Population, n: int, fitness_idx: int = 0):
    """
    Return a list with the n best individuals in population according their fitness value.

    Parameters
    ----------
    :param population: beagle.Population
        Target population.
    :param n: int
        Elite size.
    :param fitness_idx: int
        Fitness value to be considered.

    Returns
    -------
    :return: list of beagle.Individual
        Best individuals.
    """
    if n == 0:
        return []

    # get the indices of the individuals with the higher fitness value
    elite_indices = np.argsort(np.array([i.fitness[fitness_idx] for i in population]))[::-1]

    return [population[idx] for idx in elite_indices[:n]]


def _generate_random_individuals(n: int, representation: str, fitness: Fitness, kwargs: dict):
    """
    Return a list with n random individuals.

    Parameters
    ----------
    :param n: int
        Elite size.
    :param representation: str
        Representation used to code the information of possible solutions in each individual.

    Returns
    -------
    :return: list of beagle.Individual
        Best individuals.
    """
    if n == 0:
        return []

    individuals = [Individual(genotype=representation, **kwargs) for i in range(n)]

    for individual in individuals:
        fitness.evaluate(individual)

    return individuals


# --- DEFAULT ARGUMENTS --- #
_fitness_idx = 0            # ranking_selection, _elitism, _round_robin_tournament, _win_dowing, _fps, _sigma_scaling,
# _tournament_selection, _annihilation, _cumulative_probs, _select_elite
_k = 2                      # ranking_selection, _tournament_selection
_w = 1                      # ranking_selection, _tournament_selection
_replacement = False        # ranking_selection, _tournament_selection
_rank_schema = 'lineal'     # ranking_selection, _stochastic_universal_sampling, _roulette_wheel
_size = None                # survivor_selection, _round_robin_tournament
_q = 10                     # survivor_selection, _round_robin_tournament
_num_generations = 1        # _win_dowing
_reset = False              # _win_dowing
_c = 2.0                    # _sigma_scaling
_s = 2                      # _cumulative_probs


SELECTION_SCHEMAS = {
    'uniform_selection': uniform_selection,
    'ranking_selection': ranking_selection,
    'fitness_proportional_selection': fitness_proportional_selection,
    'survivor_selection': survivor_selection,
    'elitism': _elitism,
    'annihilation': _annihilation,
    'round_robin_tournament': _round_robin_tournament,
    'roulette_wheel': _roulette_wheel,
    'tournament_selection': _tournament_selection,
    'stochastic_universal_sampling': _stochastic_universal_sampling,
    'win_dowing': _win_dowing,
    'fps': _fps,
    'sigma_scaling': _sigma_scaling
}
