from .exceptions import UnrecognisedParameter
from .utils.auxiliary_functions import adjust_limits
from .population import Population
# External dependencies
import numpy as np
from copy import deepcopy

# --- DEFAULT ARGUMENTS --- #
_probability = 0.1                # _bit_flip, _random_resetting, _creep_mutation, _uniform_mutation,
# _non_uniform_mutation, _uncorrelated_mutation_one_step, _uncorrelated_mutation_n_step, _inversion_mutation,
# _swap_mutation, _insert_mutation, _scramble_mutation
_max_mutation_events = np.inf     # _bit_flip, _random_resetting, _creep_mutation, _uniform_mutation,
# _non_uniform_mutation, _uncorrelated_mutation_one_step, _uncorrelated_mutation_n_step
_distribution = np.random.normal  # _creep_mutation, _non_uniform_mutation
_std = 1                          # _creep_mutation, _non_uniform_mutation
_std_idx = False                  # _non_uniform_mutation
_sigma_threshold = 0.1            # _uncorrelated_mutation_one_step, _uncorrelated_mutation_n_step
_possible_events = 1              # _inversion_mutation, _swap_mutation, _insert_mutation, _scramble_mutation
_max_attempts = np.inf            # _inversion_mutation, _swap_mutation, _insert_mutation, _scramble_mutation
_tau = None                       # _uncorrelated_mutation_one_step (IMP, REAL VALUE IS np.sqrt(len(population[0])))
_sigma_idx = None                 # _uncorrelated_mutation_one_step (IMP, REAL VALUE IS len(population[0]) - 1)


# --- MUTATION OPERATORS --- #
# Binary representation
def _bit_flip(population: Population, **kwargs):
    """
    Bit-flip mutation operator for binary representations. This gene considers each gene independently by allowing each
    bit to flip (e.g. from 0 to 1 or from 1 to 0) based on a user-specified probability (default 0.1) from a random
    normal distribution.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    """
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)

    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events

    for indv in population:
        num_mutations = 0
        for idx, random_value in enumerate(np.random.rand(len(indv))):

            if random_value < probability:
                indv[idx] += 1 if indv[idx] == 0 else - 1
                num_mutations += 1
                if num_mutations > max_mutation_events:
                    break


# Integer representation
def _random_resetting(population: Population, **kwargs):
    """
    Random resetting mutation operator for integer representations (cardinal attributes). Extension of the bit-flip
    operator used for binary representations In this case for each position independently a new value can be chosen
    within the range of values allowed by the representation.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    """
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)

    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events

    for indv in population:
        num_mutations = 0
        for idx, random_value in enumerate(np.random.rand(len(indv))):
            if random_value < probability:

                indv[idx] = np.random.choice(list(set(indv.representation.permissible_values) - set(indv.values)))
                num_mutations += 1
                if num_mutations > max_mutation_events:
                    break


# Integer representation
def _creep_mutation(population: Population, **kwargs):
    """
    Creep mutation operator for integer representations (ordinal attributes). This scheme adds a small value
    (positive or negative) to each gene from a probability following a certain probability distribution.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    :param distribution: (optional, default np.random.normal) callable with loc and scale arguments in parameters.
        Probability distribution from which the change values are generated. If the user specifies a probability
        distribution this should receive as parameters loc (mean) and scale (standard deviation). By default
         np.random.normal with mean 0.
    :param std: (optional, default 1)
        Standard deviation of the probability distribution. By default std = 1.
    """
    distribution = kwargs.get('distribution', np.random.normal)  # normal distribution
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)
    std = kwargs.get('std', 1)

    distribution = _distribution if distribution == '_d' else distribution
    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events
    std = _std if std == '_d' else _std

    max_bound = max(population[0].representation.permissible_values)
    min_bound = min(population[0].representation.permissible_values)

    for indv in population:
        num_mutations = 0
        for idx in range(len(indv)):
            if np.random.rand() < probability:
                continue

            new_value = int(indv[idx] + distribution(loc=0, scale=std))  #  distribution with mean 0 and std

            if not indv.representation.replacement and new_value in indv.values:
                continue  #  if there cannot be repeated values and the generated value is repeated pass
            else:
                indv[idx] += new_value
                adjust_limits(indv, idx, min_bound, max_bound)

            num_mutations += 1
            if num_mutations > max_mutation_events:
                break


# Real-value representation
def _uniform_mutation(population: Population, **kwargs):
    """
    Uniform random mutation operator for real-value representations. Operator analogous to bit-flip mutation for binary
    representations and random resetting for integer representations. This operator changes a given gene with a
    probability to a new value sampled uniformly from a distribution and within the limits allowed for that value.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    """
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)

    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events

    max_bounds = np.max(population[0].representation.permissible_values, axis=1)
    min_bounds = np.min(population[0].representation.permissible_values, axis=1)

    for indv in population:
        num_mutations = 0
        for idx, random_value in enumerate(np.random.rand(len(indv))):
            if random_value < probability:
                indv[idx] = np.random.uniform(max_bounds[idx], min_bounds[idx])
                num_mutations += 1
                if num_mutations > max_mutation_events:
                    break


# Real-value representation
def _non_uniform_mutation(population: Population, **kwargs):
    """
    Non uniform mutation mutation operator for real-value representations. Operator analogous to creep mutation
    for integer representations.
    Usually, the objective of this operator is to introduce a small change in each value when it is selected to
    be mutated. To achieve this, the operator adds to the selected value a randomly sampled amount of change from
    an normal distribution with zero mean and a given standard deviation and adjusts the value to the allowed
    range of values.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    :param distribution: (optional, default np.random.normal) callable with loc and scale arguments in parameters.
        Distribution from which the amount of change introduced is generated. If the user specifies a probability
        distribution this should receive as parameters loc (mean) and scale (standard deviation). By default
         np.random.normal with mean 0.
    :param std: (optional, default 1)
        Standard deviation of the probability distribution. By default std = 1.
    :param std_idx: (optional, default False) int
        Usually the distribution deviation can be included as a value within the genotype itself to be optimized.
        In this case the position within the genotype in which the value associated to the std parameter is found must
        be specified. If this parameter is used it is recommended to put it at the beginning or at the end of the
        representation and be careful not to treat it as another value when evaluating the fitness of the individual.
        If this parameter is specified the mutation operator acquires an auto-adaptive behavior. By default uninhabited,
        it is recommended to use it only at more advanced levels of use.
    """
    distribution = kwargs.get('distribution', np.random.normal)  # normal distribution
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)
    std = kwargs.get('std', 1)
    std_idx = kwargs.get('std_idx', False)

    distribution = _distribution if distribution == '_d' else distribution
    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events
    std = _std if std == '_d' else std
    std_idx = _std_idx if std_idx == '_d' else std_idx

    max_bounds = np.max(population[0].representation.permissible_values, axis=1)
    min_bounds = np.min(population[0].representation.permissible_values, axis=1)

    for indv in population:
        num_mutations = 0
        for idx in range(len(indv)):
            if np.random.rand() < probability:
                continue

            if std_idx:
                indv[idx] += distribution(loc=0, scale=indv[std_idx])  # distribution with mean 0 and std auto-adaptive
            else:
                indv[idx] += distribution(loc=0, scale=std)  # distribution with mean 0 and std

            adjust_limits(indv, idx, min_bounds[idx], max_bounds[idx])

            num_mutations += 1
            if num_mutations > max_mutation_events:
                break


# Real-value representation
def _uncorrelated_mutation_one_step(population: Population, **kwargs):
    """
    Uncorrelated mutation operator with one step size for real-value representations. This is a self-adapting operator
    that varies the standard deviation of the probability distribution used to increase the value of the positions
    selected for mutation. In this case it is imperative to specify which position the standard deviation is
    represented (self-adaptive parameter) within the genotype of the individual (default to the last position).
    The standard deviation is subjected to mutation on the basis of the following formulas:

        std' = std * e^(tau * N(0, 1))

        Xi' = Xi + std' * Ni(0, 1)          being Xi the select i position of the individual.

    Tau parameter can be specified by the user and can be interpreted as a learning rate, by default:

        tau = 1 / sqrt(len(population))

    Since standard deviations very close to zero can have negative effects the following boundary rule is used:

        std' < sigma_threshold -> std' = sigma_threshold    being sigma_threshold the lower limit

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    :param tau: (optional, default 1 / sqrt(len(population))) float
        Parameter used to model the amount of change. This parameter can be interpreted as learning rate.
        By default 1 / sqrt(len(population)).
    :param std_idx: (optional, default last position in the individual's genotype) int
        Position that the self-adaptive parameter (the standard deviation) occupies in the genotype of individuals.
        By default the last position in the individual's genotype.
    :param sigma_threshold: (optional, default 0.1) float
        Minimum value that the standard deviation can take. By default 0.1.
    """
    probability = kwargs.get('probability', 0.1)
    tau = kwargs.get('tau', 1 / np.sqrt(len(population[0])))
    sigma_idx = kwargs.get('std_idx', len(population[0]) - 1)  # by default in the last position of the individual
    sigma_threshold = kwargs.get('sigma_threshold', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)

    probability = _probability if probability == '_d' else probability
    tau = _tau if tau == '_d' else np.sqrt(len(population[0]))
    sigma_idx = _sigma_idx if sigma_idx == '_d' else len(population[0]) - 1
    sigma_threshold = _sigma_threshold if sigma_threshold == '_d' else sigma_threshold
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events

    max_bounds = np.max(population[0].representation.permissible_values, axis=1)
    min_bounds = np.min(population[0].representation.permissible_values, axis=1)

    for indv in population:
        num_mutations = 0
        for idx, random_value in enumerate(np.random.rand(len(indv))):
            if random_value < probability:
                new_sigma = indv[sigma_idx] * np.exp(tau * np.random.normal(loc=0, scale=1))
                new_sigma = sigma_threshold if new_sigma < sigma_threshold else new_sigma
                indv[sigma_idx] = new_sigma
                indv[idx] += new_sigma * np.random.normal(loc=0, scale=1)

                adjust_limits(indv, idx, min_bounds[idx], max_bounds[idx])

                num_mutations += 1
                if num_mutations > max_mutation_events:
                    break


# Real-value representation
def _uncorrelated_mutation_n_step(population: Population, **kwargs):
    """
    Uncorrelated mutation operator with n step size for real-value representations. Extension of the
    uncorrelated_one_step operator for more info use help(beagle._uncorrelated_mutation_one_step).

    The aim of this operator is to use different step sizes for each of the individual positions. Therefore each value
    in the genotype has an associated standard deviation. In this case the last half of the individual's genotype must
    include  the sigma values. Caution should be taken when evaluating the fitness of individuals.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param probability: (optional, default 0.1) float
         Probability of a change in a certain gene. By default 0.1
    :param max_mutation_events: (optional, default inf) int
        Maximum number of mutations an individual can experience. By default unlimited.
    :param sigma_threshold: (optional, default 0.1) float
        Minimum value that the standard deviation can take. By default 0.1.
    """
    probability = kwargs.get('probability', 0.1)
    max_mutation_events = kwargs.get('max_mutation_events', np.inf)
    sigma_threshold = kwargs.get('sigma_threshold', 0.1)

    probability = _probability if probability == '_d' else probability
    max_mutation_events = _max_mutation_events if max_mutation_events == '_d' else max_mutation_events
    sigma_threshold = _sigma_threshold if sigma_threshold == '_d' else sigma_threshold

    tau = np.sqrt(len(population[0]))
    tau_ = np.sqrt(2 * np.sqrt(len(population[0])))
    max_bounds = np.max(population[0].representation.permissible_values, axis=1)
    min_bounds = np.min(population[0].representation.permissible_values, axis=1)

    for indv in population:
        num_mutations = 0
        length_indv = len(indv)
        for idx, random_value in enumerate(np.random.rand(int(length_indv / 2))):
            if random_value < probability:
                new_sigma = indv[int(length_indv / 2) + idx] * np.exp(tau * np.random.normal(loc=0, scale=1)
                                                                      + tau_ * np.random.normal(loc=0, scale=1))

                new_sigma = sigma_threshold if new_sigma < sigma_threshold else new_sigma
                indv[int(length_indv / 2) + idx] = new_sigma
                indv[idx] += new_sigma * np.random.normal(loc=0, scale=1)

                adjust_limits(indv, idx, min_bounds[idx], max_bounds[idx])

                num_mutations += 1
                if num_mutations > max_mutation_events:
                    break


# Permutation representation
def _inversion_mutation(population: Population, **kwargs):
    """
    Inversion mutation operator for permutation representation. This operator selects two values at random and reverses
    the order of the values included within the region bounded by the two selected values.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param possible_events: (optional, default 1) int
        Number of mutation events that can occur in an individual.
    :param max_attempts: (optional, default inf) int
        that can occur when the resulting genotypes are not valid according to representation restrictions.
        In this case, if the maximum number of attempts is exceeded, a new random genotype will be initialised.
    """
    probability = kwargs.get('probability', 0.1)
    possible_events = kwargs.get('possible_events', 1)
    max_attempts = kwargs.get('max_attempts', np.inf)  # Maximum number of attempts for invalid orders

    probability = _probability if probability == '_d' else probability
    possible_events = _possible_events if possible_events == '_d' else possible_events
    max_attempts = _max_attempts if max_attempts == '_d' else max_attempts

    for indv in population:
        for n in range(possible_events):
            if np.random.uniform() < probability:
                attempts = 0

                while True:  # do-while loop
                    cut_point_1, cut_point_2 = np.sort(np.random.choice(len(indv), size=2, replace=False))

                    new_order = indv.values[:cut_point_1] + indv.values[cut_point_1:cut_point_2][::-1] + indv.values[
                                                                                                         cut_point_2:]

                    if indv.representation.is_valid(new_order) or attempts == max_attempts:
                        indv.values = new_order
                        break

                    attempts += 1


# Permutation representation
def _swap_mutation(population: Population, **kwargs):
    """
    Swap mutation operator for permutation representations. In this operator two positions in the individual's genotype
    are selected at random and their values are swapped.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param possible_events: (optional, default 1) int
        Number of mutation events that can occur in an individual.
    :param max_attempts: (optional, default inf) int
        that can occur when the resulting genotypes are not valid according to representation restrictions.
        In this case, if the maximum number of attempts is exceeded, a new random genotype will be initialised.
    """
    probability = kwargs.get('probability', 0.1)
    possible_events = kwargs.get('possible_events', 1)
    max_attempts = kwargs.get('max_attempts', np.inf)  # Maximum number of attempts for invalid orders

    probability = _probability if probability == '_d' else probability
    possible_events = _possible_events if possible_events == '_d' else possible_events
    max_attempts = _max_attempts if max_attempts == '_d' else max_attempts

    for indv in population:
        for n in range(possible_events):
            if np.random.uniform() < probability:
                attempts = 0
                new_order = deepcopy(indv.values)

                while True:
                    position_1, position_2 = np.random.choice(len(indv), size=2, replace=False)
                    new_order[position_1], new_order[position_2] = new_order[position_2], new_order[position_1]

                    if indv.representation.is_valid(new_order) or attempts == max_attempts:
                        indv.values = new_order
                        break
                    # else revert change
                    new_order[position_1], new_order[position_2] = new_order[position_2], new_order[position_1]

                    attempts += 1


# Permutation representation
def _insert_mutation(population: Population, **kwargs):
    """
    Insert mutation operator for permutation representations. In this operator two positions are selected at random and
    the second value is moved after the first value.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param possible_events: (optional, default 1) int
        Number of mutation events that can occur in an individual.
    :param max_attempts: (optional, default inf) int
        that can occur when the resulting genotypes are not valid according to representation restrictions.
        In this case, if the maximum number of attempts is exceeded, a new random genotype will be initialised.
    """
    probability = kwargs.get('probability', 0.1)
    possible_events = kwargs.get('possible_events', 1)
    max_attempts = kwargs.get('max_attempts', np.inf)  # Maximum number of attempts for invalid orders

    probability = _probability if probability == '_d' else probability
    possible_events = _possible_events if possible_events == '_d' else possible_events
    max_attempts = _max_attempts if max_attempts == '_d' else max_attempts

    for indv in population:
        for n in range(possible_events):
            if np.random.uniform() < probability:

                attempts = 0
                new_order = deepcopy(indv.values)
                while True:
                    position_1, position_2 = np.random.choice(len(indv), size=2, replace=False)
                    old_order = deepcopy(new_order)
                    value_to_move = new_order[position_2]
                    del new_order[position_2]
                    new_order.insert(position_1 + 1, value_to_move)

                    if indv.representation.is_valid(new_order) or attempts == max_attempts:
                        indv.values = new_order
                        break
                    # revert order
                    new_order = old_order
                    attempts += 1


# Permutation representation
def _scramble_mutation(population: Population, **kwargs):
    """
    Insert mutation operator for permutation representations. In this operator the whole chromosome or a region is
    chosen and its positions are scrambled.

    Parameters
    ----------
    :param population: beagle.Population
        Population to be mutated.
    :param possible_events: (optional, default 1) int
        Number of mutation events that can occur in an individual.
    :param max_attempts: (optional, default inf) int
        that can occur when the resulting genotypes are not valid according to representation restrictions.
        In this case, if the maximum number of attempts is exceeded, a new random genotype will be initialised.
    """
    probability = kwargs.get('probability', 0.1)
    possible_events = kwargs.get('possible_events', 1)
    max_attempts = kwargs.get('max_attempts', np.inf)  # Maximum number of attempts for invalid orders

    probability = _probability if probability == '_d' else probability
    possible_events = _possible_events if possible_events == '_d' else possible_events
    max_attempts = _max_attempts if max_attempts == '_d' else max_attempts

    for indv in population:
        for n in range(possible_events):
            if np.random.uniform() < probability:
                attempts = 0
                new_order = np.array(deepcopy(indv.values), dtype=int)
                while True:
                    position_idx = np.random.choice(len(indv), size=np.random.randint(len(indv)), replace=False)
                    mixed_idx = deepcopy(position_idx)
                    np.random.shuffle(mixed_idx)
                    new_order[position_idx] = new_order[mixed_idx]

                    if indv.representation.is_valid(new_order) or attempts == max_attempts:
                        new_order = new_order.tolist()
                        indv.values = new_order
                        break
                    # revert order
                    new_order[mixed_idx] = new_order[mixed_idx]

                    attempts += 1


MUTATION_SCHEMAS = {
    'bit_flip': _bit_flip,                                      # Binary
    'random_resetting': _random_resetting,                      # Integer
    'creep': _creep_mutation,                                   # Integer
    'uniform': _uniform_mutation,                               # Real-value
    'non_uniform': _non_uniform_mutation,                       # Real-value
    'uncorrelated_one_step': _uncorrelated_mutation_one_step,   # Real-value
    'uncorrelated_n_step': _uncorrelated_mutation_n_step,       # Real-value
    'inversion': _inversion_mutation,                           # Permutation
    'swap': _swap_mutation,                                     # Permutation
    'insert': _insert_mutation,                                 # Permutation
    'scramble': _scramble_mutation                              # Permutation
}

DEFAULT_SCHEMAS = {
    'binary': _bit_flip,
    'integer': _random_resetting,
    'real': _uniform_mutation,
    'permutation': _inversion_mutation
}


def mutation(population: Population, schema: str = None, **kwargs):
    """
    Variation operator that subjects a population to mutation events. For more information on available operators use:

        For bit-flip operator (binary representation):
            help(beagle.MUTATION_SCHEMAS['bit_flip])

        For random resetting operator (integer representation)
            help(beagle.MUTATION_SCHEMAS['random_resetting])

        For creep mutation operator (integer representation)
            help(beagle.MUTATION_SCHEMAS['creep'])

        For uniform mutation operator (real-value representation)
            help(beagle.MUTATION_SCHEMAS['uniform'])

        For non-uniform mutation operator (real-value representation)
            help(beagle.MUTATION_SCHEMAS['non_uniform'])

        For uncorrelated one step mutation operator (real-value representation)
            help(beagle.MUTATION_SCHEMAS['uncorrelated_one_step'])

        For uncorrelated N step mutation operator (real-value representation)
            help(beagle.MUTATION_SCHEMAS['uncorrelated_n_step'])

        For inversion mutation operator (permutation representation)
            help(beagle.MUTATION_SCHEMAS['inversion'])

        For swap mutation operator (permutation representation)
            help(beagle.MUTATION_SCHEMAS['swap'])

        For insert mutation operator (permutation representation)
            help(beagle.MUTATION_SCHEMAS['insert'])

        For scramble mutation operator (permutation representation)
            help(beagle.MUTATION_SCHEMAS['scramble'])

    Parameters
    ---------
    :param population: beagle.Population
        Population to be mutated.

    :param schema: (optional) str
        Scheme to be followed. Possible schemes:

            bit_flip                    (binary representation)
            random_resetting            (integer representation)
            creep                       (integer representation)
            uniform                     (real-value representation)
            non_uniform                 (real-value representation)
            uncorrelated_one_step       (real-value representation)
            uncorrelated_n_step         (real-value representation)
            inversion                   (permutation representation)
            swap                        (permutation representation)
            insert                      (permutation representation)
            scramble                    (permutation representation)

        Default schemes:

            bit_flip                    (binary representation)
            random_resetting            (integer representation)
            uniform                     (real-value representation)
            inversion                   (permutation representation)
    :param kwargs:
        Additional arguments depending on the scheme to be used.
    """
    if schema is None:  # default parameter for representation
        DEFAULT_SCHEMAS[population.representation](population, **kwargs)
    else:
        if schema in MUTATION_SCHEMAS.keys():
            MUTATION_SCHEMAS[schema](population, **kwargs)
        else:
            raise UnrecognisedParameter(schema, 'mutation')
