from ..fitness import Fitness, evaluate, evaluate_parallel
from ..population import Population, merge_populations
from ..mutation import mutation
from ..recombination import recombination
from ..selection import uniform_selection, ranking_selection, fitness_proportional_selection, survivor_selection
from ..report import SaveBestSolution


# -- BASIC SCHEMAS -- #
def basic_ga_1(parents: Population, fitness_function: Fitness, **kwargs):
    """
    Scheme of the step function of a basic genetic algorithm The steps followed by this scheme are:

    1. recombination(parents) -> Offspring (length == parents)
    2. mutation(offspring)
    3. evaluation(offspring)
    4. parents + offspring -> population
    5. selection(population) -> parents (and create a report)

    Parameters
    -----------
    :param parents: beagle.Population
    :param fitness_function: beagle.Fitness
    :param kwargs: dict
        Additional arguments for recombination, mutation and selection operators.

    Returns
    -------
    :return: beagle.Population
        Population for the next generation.
    """

    offspring = recombination(parents, n=len(parents),
                              # optional arguments
                              **_recombination_default_params(**kwargs)
                              )
    mutation(offspring,
             # optional arguments
             **_mutation_default_params(**kwargs)
             )

    if kwargs.get('evaluate_in_parallel', True):
        evaluate_parallel(offspring, fitness_function)
    else:
        evaluate(offspring, fitness_function)

    parents_offspring = merge_populations(parents, offspring)

    next_generation = _SELECTION_OPERATORS[kwargs.get('selection', 'ranking')](
        population=parents_offspring, n=len(parents),
        # optional arguments
        **_selection_default_params(**kwargs)
        )

    kwargs['report'].create_report(next_generation, population_name='population', increment_generation=True)

    return next_generation, SaveBestSolution(next_generation)


def basic_ga_2(parents: Population, fitness_function: Fitness, **kwargs):
    """
    Scheme of the step function of a basic genetic algorithm The steps followed by this scheme are:

        1. Select elite -> Elite 
        2. Annihilate worst individuals from population
        3. Apply tournament selection -> Best Individuals
        4. Apply cross-over to Best Individuals -> Offspring
        5. Apply mutation to Offspring
        6. Offspring + Elite -> Next generation

    Some of the parameters that can be specified for this step function are:

        - Percentage of individuals in elite. (By default 0.1)
            elitism_select: float

        - Percentage of individuals to be annihilated. (By default 0.1)
            annihilation_annihilate: float

        - Number of individuals selected for tournament selection. (By default 2)
            tournament_k: int

        - Number of winners in tournament selection. (By default 1)
            tournament_w: int

        - Tournament selection with or without replacement. (By default False)
            tournament_replacement: bool
    """
    population_length = len(parents)

    elite = survivor_selection(population=parents, schema='elitism', select=kwargs.get('elitism_select', 0.1))

    parents = survivor_selection(
        population=parents, schema='annihilation', annihilate=kwargs.get('annihilation_annihilate', 0.1))

    mating_pool = ranking_selection(
        population=parents, n=population_length-len(elite), schema='tournament', w=kwargs.get('tournament_w', 1),
        k=kwargs.get('tournament_k', 2), replacement=kwargs.get('tournament_replacement'))

    offspring = recombination(
        population=mating_pool, n=len(mating_pool),
        # optional arguments
        **_recombination_default_params(**kwargs)
    )

    mutation(offspring,
             # optional arguments
             **_mutation_default_params(**kwargs)
             )

    if kwargs.get('evaluate_in_parallel', True):
        evaluate_parallel(offspring, fitness_function)
    else:
        evaluate(offspring, fitness_function)

    next_generation = merge_populations(offspring, elite)

    kwargs['report'].create_report(next_generation, population_name='population', increment_generation=True)

    return next_generation, SaveBestSolution(next_generation)


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


def _recombination_default_params(**kwargs):
    params = {
        'schema': kwargs.get('recombination', None),
        'cut_points': kwargs.get('recombination_cut_points', '_d'),
        'probability': kwargs.get('recombination_probability_uc', '_d'),
        'ari_type': kwargs.get('recombination_ari_type', '_d'),
        'alpha': kwargs.get('recombination_alpha', '_d')
    }

    return params


def _selection_default_params(**kwargs):
    params = {
        'schema': kwargs.get('selection_schema', None),
        'annihilate': kwargs.get('annihilation_annihilate', None),
        'select': kwargs.get('elitism_select', None),
        'idx_fitness': kwargs.get('selection_idx_fitness', '_d'),
        'replacement': kwargs.get('selection_replacement', '_d'),
        'k': kwargs.get('selection_k', '_d'),
        'w': kwargs.get('selection_w', '_d'),
        'rank_schema': kwargs.get('selection_rank_schema', '_d'),
        'fitness_idx': kwargs.get('selection_fitness_idx', '_d')
    }

    return params


_SELECTION_OPERATORS = {
    'uniform': uniform_selection,
    'ranking': ranking_selection,
    'proportional': fitness_proportional_selection
}

