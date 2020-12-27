import numpy as np
from copy import deepcopy
from .exceptions import UnrecognisedParameter
from .population import Population, Individual
from .representation import Integer, Binary, Permutation, RealValue
from .utils.auxiliary_functions import adjust_limits


# --- DEFAULT ARGUMENTS --- #
_cut_points = 1  # _n_point_crossover_b, _n_point_crossover_i, _n_point_crossover_r
_probability_uc = 0.5  # _uniform_crossover_b, _uniform_crossover_i, _uniform_crossover_r
_ari_type = 'whole'  # _arithmetic_recombination
_alpha = 0.5  # _arithmetic_recombination, _blend_crossover


# --- HELPING FUNCTIONS --- #
def _get_parents(population: Population, length: int):
    """
    Function that returns two individuals randomly selected from the population received as argument.

    Parameters
    -----------
    :param population: beagle.Population
        Population used to sample individuals.
    :param length: int
        Population size.

    Returns
    --------
    :return: tuple of beagle.Individuals
        Two random individuals from population.
    """
    r_int_1, r_int_2 = np.random.randint(0, length, size=2)
    parent_1, parent_2 = population[r_int_1], population[r_int_2]

    return parent_1, parent_2


def _create_real_individuals(n: int, bounds: list):
    """
    Function that creates n individuals with a non-initialized genotype represented as real values.

    Parameters
    ----------
    :param n: int
        Number of individuals to create.
    :param bounds: list of tuples of size 2
        Limits of the values that the representation can take.

    Returns
    --------
    :return: list of beagle.Individuals
    """
    return [Individual(genotype='real', representation=RealValue(bounds=bounds, initialization=False)) for i in
            range(n)]


def _create_integer_individuals(n: int, value_coding: dict, replacement: bool):
    """
    Function that creates n individuals with a non-initialized genotype represented as integer values.

    Parameters
    ----------
    :param n: int
        Number of individuals to create.
    :param value_coding: dict
        Dictionary with the codification of the attributes in integer values.
    :param replacement: bool
        Variable that indicates if there can be repeated values within the genotype.

    Returns
    --------
    :return: list of beagle.Individuals
    """

    return [Individual(genotype='integer', representation=Integer(value_coding, replacement, initialization=False))
            for i in range(n)]


def _create_binary_individuals(n: int, length: int):
    """
    Function that creates n individuals with a non-initialized genotype represented as binary values.

    Parameters
    ----------
    :param n: int
        Number of individuals to create.
    :param length: int
        Length of the individuals genotype.

    Returns
    --------
    :return: list of beagle.Individuals
    """
    return [Individual(genotype='binary', representation=Binary(length, initialization=False)) for i in range(n)]


def _create_permutation_individuals(n: int, events: list, restrictions: list):
    """
    Function that creates n individuals with a non-initialized genotype represented as permutation values.

    Parameters
    ----------
    :param n: int
        Number of individuals to create.
    :param events: list
        List of possible events.
    :param restrictions: bool
        List of restrictions. For more info use help(beagle.Permutation)

    Returns
    --------
    :return: list of beagle.Individuals
    """
    return [
        Individual(genotype='permutation', representation=Permutation(events=events, restrictions=restrictions,
                                                                      initialization=False))
        for i in range(n)]


def _replace_repeated_values(values, permissible_values):
    """
    Function that takes a list of values and possible values that can be included in an individual's genotype and
    replaces the repeated values with new ones.

    Parameters
    ----------
    :param values: list
        Values in the genotype of the individual.
    :param permissible_values: list
        List of permissible values to include in the individual genotype.

    Returns
    --------
    :return: list
        Values in the genotype of the individual without repeated variables.
    """
    unique_values = np.unique(values)
    num_repeated = len(values) - len(unique_values)

    if num_repeated != 0:
        try:
            values = np.append(unique_values, np.random.choice(
                np.setdiff1d(unique_values, permissible_values), size=num_repeated, replace=False)
                               )
        except ValueError:  # exhausted options to replace repeated values after the crossover

            values = np.unique(values)

            if len(values) < 2:  # unique values are less than 2
                values = np.random.choice(permissible_values, size=2, replace=False)

    np.random.shuffle(values)

    return values


def _one_point_crossover(parent_1: Individual, parent_2: Individual, length: int):
    """
    Generalization of one point cross-over procedure.

    Parameters
    ----------
    :param parent_1: beagle.Individual
    :param parent_2: beagle.Individual
    :param length: int

    Returns
    -------
    :return tuple of size 2 of arrays
    """
    cut_p = np.random.randint(low=1, high=length)

    return np.append(parent_1[0:cut_p], parent_2[cut_p:]), np.append(parent_2[0:cut_p], parent_1[cut_p:])


def _n_point_crossover(parent_1: Individual, parent_2: Individual, length: int, n_breaks: int):
    """
    Generalization of n cross-over procedure.

    Parameters
    ----------
    :param parent_1: beagle.Individual
    :param parent_2: beagle.Individual
    :param length: int
    :param n_breaks: int
        Number of break points.

    Returns
    -------
    :return tuple of size 2 of arrays
    """
    def check_crossover_requirements(individual_: Individual, n_breaks_: int):
        """
        If n_breaks is greater than 1 individuals with the same length than n_breaks will raise errors therefore it is
        necessary to add a random value to reach the necessary length and perform the n-point crossover.
        """
        if n_breaks_ > 1:
            if len(individual_) <= n_breaks_:
                while not len(individual_) > n_breaks_:
                    individual_.values = np.append(
                        individual_.values, np.random.choice(individual_.representation.permissible_values))
                    _replace_repeated_values(individual_.values, individual_.representation.permissible_values)

    check_crossover_requirements(parent_1, n_breaks)
    check_crossover_requirements(parent_2, n_breaks)

    cut_points = np.sort(np.random.choice(length - 1, size=n_breaks, replace=False) + 1)  # generate random cut points
    cut_points = np.insert(cut_points, [0, len(cut_points)], [0, length])  # insert beginning and end

    repr_1, repr_2 = [], []
    for n in range(len(cut_points) - 1):
        if n % 2 == 0:
            repr_1 += parent_1[cut_points[n]:cut_points[n + 1]].tolist()
            repr_2 += parent_2[cut_points[n]:cut_points[n + 1]].tolist()
        else:
            repr_1 += parent_2[cut_points[n]:cut_points[n + 1]].tolist()
            repr_2 += parent_1[cut_points[n]:cut_points[n + 1]].tolist()

    return np.array(repr_1), np.array(repr_2)


def _uniform_crossover(parent_1: Individual, parent_2: Individual, probability: float, length: int):
    """
    Generalization of uniform cross-over operator.

    Parameters
    -----------
    :param parent_1: beagle.Individual
    :param parent_2: beagle.Individual
    :param probability: float
        Probability of sampling from parent_1 (probability) and parent_2 (1 - probability)
    :param length: int

    Returns
    -------
    :return tuple of size 2 of arrays
    """
    child_values = []

    for i in range(length):

        if probability <= np.random.uniform():
            if i < len(parent_1):
                child_values.append(parent_1[i])
        else:
            if i < len(parent_2):
                child_values.append(parent_2[i])

    return child_values


def _adjust_real_value_bounds(individual: Individual):
    """
    Function that adjusts an individual's values to the limits specified in his or her representation.

    Parameters
    ----------
    :param individual: beagle.Individual
    """
    for i in range(len(individual)):
        adjust_limits(individual, i, *individual.representation.permissible_values[i])


def _single_arithmetic(parent_1: Individual, parent_2: Individual, alpha: float):
    k = np.random.randint(1, len(parent_1))

    repr_1 = np.concatenate(
        (parent_1.values[:k], [alpha * parent_1.values[k] + (1 - alpha) * parent_2.values[k]], parent_2[k + 1:])
    )
    repr_2 = np.concatenate(
        (parent_2.values[:k], [alpha * parent_2.values[k] + (1 - alpha) * parent_1.values[k]], parent_1[k + 1:])
    )

    return repr_1, repr_2


def _simple_arithmetic(parent_1: Individual, parent_2: Individual, alpha: float):
    k = np.random.randint(1, len(parent_1))

    repr_1 = np.concatenate(
        (parent_1.values[:k],
         [alpha * parent_1.values[i] + (1 - alpha) * parent_2.values[i] for i in range(k, len(parent_1))])
    )
    repr_2 = np.concatenate(
        (parent_2.values[:k],
         [alpha * parent_2.values[i] + (1 - alpha) * parent_1.values[i] for i in range(k, len(parent_1))])
    )

    return repr_1, repr_2


def _whole_arithmetic(parent_1: Individual, parent_2: Individual, alpha: float):
    repr_1 = np.array([alpha * parent_1.values[i] + (1 - alpha) * parent_2.values[i] for i in range(len(parent_1))])

    if alpha == 0.5:
        repr_2 = deepcopy(repr_1)
    else:
        repr_2 = np.array([alpha * parent_2.values[i] + (1 - alpha) * parent_1.values[i] for i in range(len(parent_1))])

    return repr_1, repr_2


def _pmx(parent_1: Individual, parent_2: Individual, cut_points):
    offspring = [None] * len(parent_1)
    offspring[cut_points[0]:cut_points[1]] = parent_1[cut_points[0]:cut_points[1]]

    # element from parent 2 that aren't in offspring and their indices
    not_copied_elements, not_copied_indices = [], []

    for idx in range(cut_points[0], cut_points[1]):
        if parent_2[idx] not in offspring:
            not_copied_elements.append(parent_2[idx])
            not_copied_indices.append(idx)

    for e, idx in zip(not_copied_elements, not_copied_indices):
        vacant = False

        while not vacant:
            # get the position occupied by the element not present from parent 2 in the offspring in parent 1
            idx_child = np.where(np.array(parent_2.values) == parent_1[idx])[0][0]

            if parent_1[idx_child] not in offspring[cut_points[0]:cut_points[1]]:
                vacant = True
            else:
                idx = idx_child

        # put the element from parent 2 indicated by the calculated index
        offspring[idx_child] = e

    # copy elements from parent 2 offspring
    for idx in range(len(offspring)):
        if offspring[idx] is None:
            offspring[idx] = parent_2[idx]

    return offspring


def _adjacency_table(parent_1: Individual, parent_2: Individual):
    adjacency_table = {}
    length_representation = len(parent_1.values) - 1

    for idx, value in enumerate(parent_1.values):
        if idx == length_representation:
            adjacency_table[value] = [parent_1[idx - 1], parent_1[0]]
        else:
            adjacency_table[value] = [parent_1[idx - 1], parent_1[idx + 1]]

    for idx, value in enumerate(parent_2.values):
        if idx == length_representation:
            adjacency_table[value] += [parent_2[idx - 1], parent_2[0]]
        else:
            adjacency_table[value] += [parent_2[idx - 1], parent_2[idx + 1]]

    return adjacency_table


def _edge_crossover_construction(adjacency_table: dict):
    def common_edge(l: list):
        """
        Check if there is a common edge and return the value.
        """
        for e in l:
            if l.count(e) > 1:
                return e
        return None

    def shortest_list(element: int, table: dict):
        """
        Return the element with the shorted number of elements in the adjacency table.
        If all elements has the same length return None.
        """
        freq = [len(np.unique(table[e])) for e in table[element]]
        min_value = min(freq)

        if min_value == max(freq):
            return None

        for idx, val in enumerate(freq):
            if val == min_value:
                return table[element][idx]

    def is_repeated(element: int, l: list):
        if element is None or element in l:
            return None

        return element

    selected_element = np.random.choice(list(adjacency_table.keys()))  # random selection
    repr_values = [selected_element]

    while len(adjacency_table) != len(repr_values):
        # delete added elements

        new_element = is_repeated(common_edge(adjacency_table[selected_element]), repr_values)

        if new_element is None:
            new_element = is_repeated(shortest_list(selected_element, adjacency_table), repr_values)

            if new_element is None:
                possible_selections = [e for e in adjacency_table[selected_element] if e not in repr_values]

                if len(possible_selections) == 0:
                    possible_selections = [e for e in list(adjacency_table.keys()) if e not in repr_values]

                if len(possible_selections) == 1:  # last element
                    new_element = possible_selections[0]
                else:  # all possible elections already included
                    new_element = np.random.choice(possible_selections)

        repr_values.append(new_element)

        selected_element, new_element = new_element, None

    return repr_values


def _order_crossover_copy(parent_1: Individual, parent_2: Individual, cut_points):
    length = len(parent_1)
    repr_values = [None] * length
    repr_values[cut_points[0]:cut_points[1]] = parent_1[cut_points[0]:cut_points[1]]
    idx = cut_points[1]

    for parent_val in np.concatenate((parent_2[idx:], parent_2[:cut_points[1]])):

        if parent_val not in parent_1[cut_points[0]:cut_points[1]]:
            repr_values[idx] = parent_val

            idx += 1

            if idx == length:
                idx = 0

    return repr_values


def _identify_cycles(parent_1: Individual, parent_2: Individual):
    def get_cycle(initial_index: int, p1: Individual, p2: Individual, index: int = None, cycle: list = None):
        if cycle is None:
            cycle = [initial_index]
        if index is None:
            index = initial_index

        index = np.argwhere(np.array(p2.values) == p1[index])[0][0]

        if index in cycle:
            return cycle

        cycle.append(index)

        return get_cycle(initial_index, p1, p2, index, cycle)

    cycles = []
    remaining = np.arange(len(parent_1))  # index of the array
    next_index = 0

    while len(remaining) > 0:
        if parent_1[next_index] == parent_2[next_index]:
            cycles.append([next_index])
            remaining = np.delete(remaining, np.argwhere(remaining == next_index))
            if len(remaining) > 0:
                next_index = remaining[0]
        else:
            new_cycle = get_cycle(next_index, parent_1, parent_2)
            cycles.append(new_cycle)
            remaining = np.setdiff1d(remaining, np.sort(new_cycle), assume_unique=True)

            if len(remaining) > 0:
                next_index = remaining[0]
                remaining = np.delete(remaining, 0)
                if len(remaining) == 0:
                    cycles.append([next_index])

    return cycles


# --- RECOMBINATION OPERATORS --- #
# Binary representation
def _n_point_crossover_b(population: Population, n: int, **kwargs):
    """
    N-point cross-over operator adapted for binary representations.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param cut_points: (optional, default 1) int
        Number of cut-points, if this value is equal to 1 one-point cross-over will be performed, otherwise n-point
        cross-over will take place. By default one-point cross-over will be carried out. By default 1.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    cut_points = kwargs.get('cut_points', 1)
    cut_points = _cut_points if cut_points == '_d' else cut_points

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        if cut_points == 1:
            repr_1, repr_2 = _one_point_crossover(parent_1, parent_2, len(parent_1))
        else:
            repr_1, repr_2 = _n_point_crossover(parent_1, parent_2, len(parent_1), cut_points)

        # create new individuals
        child_1, child_2 = _create_binary_individuals(n=2, length=len(parent_1))

        # assign new representations
        child_1.values, child_2.values = repr_1, repr_2

        individuals[idx] = child_1
        idx += 1

        if idx == n: break

        individuals[idx] = child_2
        idx += 1

    return Population(size=n, representation='binary', individuals=individuals)


# Binary representation
def _uniform_crossover_b(population: Population, n: int, **kwargs):
    """
    Random uniform point cross-over operator adapted for binary representations. This operator worked by dividing the
    parents into a number of sections of contiguous genes and reassembling them to produce the offspring.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param probability: (optional, default 0.5) float between 0 and 1
        Probability that a certain gene of the genotype is chosen from one parent, the probability that the gene that
        occupies the same position in the other parent is chosen will be 1 - probability. By default 0.5.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    probability = kwargs.get('probability', 0.5)
    probability = _probability_uc if probability == '_d' else probability

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        # create new individuals
        child = Individual(genotype='binary', representation=Binary(length=len(parent_1), initialization=False))

        # assign new representations
        child.values = _uniform_crossover(parent_1, parent_2, probability, len(parent_1))

        individuals[idx] = child

        idx += 1

    return Population(size=n, representation='binary', individuals=individuals)


# Integer representation
def _n_point_crossover_i(population: Population, n: int, **kwargs):
    """
    N-point cross-over operator adapted for integer representations.


    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param cut_points: (optional, default 1) int
        Number of cut-points, if this value is equal to 1 one-point cross-over will be performed, otherwise n-point
        cross-over will take place. By default one-point cross-over will be carried out. By default 1.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    cut_points = kwargs.get('cut_points', 1)
    cut_points = _cut_points if cut_points == '_d' else cut_points

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        if cut_points == 1:
            repr_1, repr_2 = _one_point_crossover(
                parent_1, parent_2, len(parent_1) if len(parent_1) > len(parent_2) else len(parent_2))
        else:
            repr_1, repr_2 = _n_point_crossover(
                parent_1, parent_2, len(parent_1) if len(parent_1) > len(parent_2) else len(parent_2), cut_points)

        # create new individuals
        child_1, child_2 = _create_integer_individuals(n=2, value_coding=parent_1.representation.value_coding,
                                                       replacement=parent_1.representation.replacement)

        # check if both individuals are valid (for repeated values when replacement is false)
        if not parent_1.representation.replacement:
            repr_1 = _replace_repeated_values(repr_1, child_1.representation.permissible_values)
            repr_2 = _replace_repeated_values(repr_2, child_1.representation.permissible_values)

        # assign new representations
        child_1.values, child_2.values = repr_1, repr_2
        individuals[idx] = child_1
        idx += 1

        if idx == n: break

        individuals[idx] = child_2
        idx += 1

    return Population(size=n, representation='integer', individuals=individuals)


# Integer representation
def _uniform_crossover_i(population: Population, n: int, **kwargs):
    """
    Random uniform point cross-over operator adapted for integer representations. This operator worked by dividing the
    parents into a number of sections of contiguous genes and reassembling them to produce the offspring.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param probability: (optional, default 0.5) float between 0 and 1
        Probability that a certain gene of the genotype is chosen from one parent, the probability that the gene that
        occupies the same position in the other parent is chosen will be 1 - probability. By default 0.5.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    probability = kwargs.get('probability', 0.5)
    probability = _cut_points if _probability_uc == '_d' else probability

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        # create new individual
        child = Individual(
            genotype='integer', representation=Integer(parent_1.representation.value_coding,
                                                       parent_1.representation.replacement, initialization=False))

        repr_values = _uniform_crossover(
            parent_1, parent_2, probability, len(parent_1) if len(parent_1) > len(parent_2) else len(parent_2))

        # check if both individuals are valid (for repeated values when replacement is false)
        if not parent_1.representation.replacement:
            repr_values = _replace_repeated_values(repr_values, child.representation.permissible_values)

        # assign new representations values
        child.values = repr_values

        individuals[idx] = child

        idx += 1

    return Population(size=n, representation='integer', individuals=individuals)


# Real-value representation
def _arithmetic_recombination(population: Population, n: int, **kwargs):
    """
    Arithmetic recombination operator for real-valued representations. This operator for each gene position creates a
    new allele value in the offspring that lies between those of the parents. This operator is able to create new gene
    material.
    In this case the selected value Xi will consist of:

        Xi = alpha * parent_1i + (1 - alpha) * parent_2i

    for some alpha between 0 and 1.
    Three types of arithmetic recombination operators have been implemented:

        - 'simple': Simple arithmetic recombination operator.
        - 'single': Single arithmetic recombination operator.
        - 'whole': Whole arithmetic recombination operator (default).

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param ari_type: (optional, default whole)str
        It specifies the type of arithmetic recombination carried out. Available: 'simple', 'single', 'whole'. By
        default 0.5.
    :param alpha (optional, default 0.5): float
        It specifies the contribution of each parent to the genotype of the offspring. By default 0.5.

    """
    ari_type = kwargs.get('ari_type', 'whole')
    alpha = kwargs.get('alpha', 0.5)

    ari_type = _ari_type if ari_type == '_d' else _ari_type
    alpha = _alpha if alpha == '_d' else alpha

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:

        parent_1, parent_2 = _get_parents(population, length_population)

        if ari_type == 'whole':
            repr_1, repr_2 = _whole_arithmetic(parent_1, parent_2, alpha)
        elif ari_type == 'single':
            repr_1, repr_2 = _single_arithmetic(parent_1, parent_2, alpha)
        elif ari_type == 'simple':
            repr_1, repr_2 = _simple_arithmetic(parent_1, parent_2, alpha)
        else:
            raise UnrecognisedParameter(ari_type, 'ari_type')

        # create new individuals
        child_1, child_2 = _create_real_individuals(n=2, bounds=parent_1.representation.permissible_values)

        # assign new representations
        child_1.values, child_2.values = repr_1, repr_2

        _adjust_real_value_bounds(child_1)
        _adjust_real_value_bounds(child_2)

        individuals[idx] = child_1
        idx += 1

        if idx == n: break

        individuals[idx] = child_2
        idx += 1

    return Population(size=n, representation='real', individuals=individuals)


# Real-value representation
def _n_point_crossover_r(population: Population, n: int, **kwargs):
    """
    N-point cross-over operator adapted for real-value representations, also called discrete recombination. This
    operator isn't able to create new genetic material.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param cut_points: (optional, default 1) int
        Number of cut-points, if this value is equal to 1 one-point cross-over will be performed, otherwise n-point
        cross-over will take place. By default one-point cross-over will be carried out. By default 1.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.

    """
    cut_points = kwargs.get('cut_points', 1)
    cut_points = _cut_points if cut_points == '_d' else cut_points

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        if cut_points == 1:
            repr_1, repr_2 = _one_point_crossover(parent_1, parent_2, len(parent_1))
        else:
            repr_1, repr_2 = _n_point_crossover(parent_1, parent_2, len(parent_1), cut_points)

        # create new individuals
        child_1, child_2 = _create_real_individuals(n=2, bounds=parent_1.representation.permissible_values)

        # assign new representations
        child_1.values, child_2.values = repr_1, repr_2

        _adjust_real_value_bounds(child_1)
        _adjust_real_value_bounds(child_2)

        individuals[idx] = child_1
        idx += 1

        if idx == n: break

        individuals[idx] = child_2
        idx += 1

    return Population(size=n, representation='real', individuals=individuals)


# Real-value representation
def _uniform_crossover_r(population: Population, n: int, **kwargs):
    """
    Random uniform point cross-over operator adapted for integer representations. This operator worked by dividing the
    parents into a number of sections of contiguous genes and reassembling them to produce the offspring. This
    operator isn't able to create new genetic material.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param probability: (optional, default 0.5) float between 0 and 1
        Probability that a certain gene of the genotype is chosen from one parent, the probability that the gene that
        occupies the same position in the other parent is chosen will be 1 - probability. By default 0.5.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    probability = kwargs.get('probability', 0.5)
    probability = _probability_uc if probability == '_d' else probability

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        # create new individual
        child = Individual(genotype='real', representation=RealValue(bounds=parent_1.representation.permissible_values,
                                                                     initialization=False))

        repr_values = _uniform_crossover(parent_1, parent_2, probability, len(parent_1))

        # assign new representations values
        child.values = repr_values

        _adjust_real_value_bounds(child)

        individuals[idx] = child

        idx += 1

    return Population(size=n, representation='real', individuals=individuals)


# Real-value representation
def _blend_crossover(population: Population, n: int, **kwargs):
    """
    Blend crossover operator for real-value representations. This operator allows to create offspring in a region that
    is bigger than the n-dimensional rectangle spanned by the parents. The extra space is proportional to the distance
    between the parents and it varies per coordinate.
    Considering two parents, X and Y, if the value in the ith position Xi < Yi then the difference Di = Yi - Xi and
    the range for the ith value in the child Z will be [Xi - alpha * Di, Xi + alpha *Di]. Therefore to create a child
    we can sample a random number mu uniformly from [0, 1], calculate gamma as (1 - 2 * alpha) * mu - alpha and set Zi
    to (1 - gamma) * Xi + gamma * Yi.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.
    :param alpha: (optional, default 0.5 [recommended]) float between 0 and 1
        The alpha parameter indicates the balance between the contribution of each parent to the offspring.
        By default 0.5.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    alpha = kwargs.get('alpha', 0.5)

    alpha = _alpha if alpha == '_d' else alpha

    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        mu = np.random.uniform(low=0, high=1, size=len(parent_1))
        gamma = np.array([(1 - 2 * alpha) * mu_val - alpha for mu_val in mu])

        repr_1 = (1 - gamma) * parent_1.values + gamma * parent_2.values
        repr_2 = (1 - gamma) * parent_2.values + gamma * parent_1.values

        # create new individuals
        child_1, child_2 = _create_real_individuals(n=2, bounds=parent_1.representation.permissible_values)

        # assign new representations
        child_1.values, child_2.values = repr_1, repr_2

        _adjust_real_value_bounds(child_1)
        _adjust_real_value_bounds(child_2)

        individuals[idx] = child_1
        idx += 1

        if idx == n: break

        individuals[idx] = child_2
        idx += 1

    return Population(size=n, representation='real', individuals=individuals)


# Permutation representation
def _pmx_crossover(population: Population, n: int, **kwargs):
    """
    Partially Mapped Crossover for permutation representations. Particularly useful for adjacency-type problems.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)
        cut_points = np.sort(np.random.choice(len(parent_1) - 1, size=2, replace=False) + 1)

        child_1, child_2 = _create_permutation_individuals(
            2, parent_1.representation.permissible_values, parent_1.representation.restrictions)

        repr_1 = _pmx(parent_1, parent_2, cut_points)
        repr_2 = _pmx(parent_2, parent_1, cut_points)

        if parent_1.representation.is_valid(repr_1):
            child_1.values = repr_1
            individuals[idx] = child_1
            idx += 1
            if idx == n: break

        if parent_1.representation.is_valid(repr_2):
            child_2.values = repr_2
            individuals[idx] = child_2
            idx += 1

    return Population(size=n, representation='permutation', individuals=individuals)


# Permutation representation
def _edge_3_crossover(population: Population, n: int, **kwargs):
    """
    Edge-3-crossover operator for permutation representations. Implementation based on Whitley article. This operator
    ensures that common edges are preserved.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        adjacency_table = _adjacency_table(parent_1, parent_2)

        repr_values = _edge_crossover_construction(adjacency_table)

        child = Individual(genotype='permutation',
                           representation=Permutation(events=parent_1.representation.permissible_values,
                                                      restrictions=parent_1.representation.restrictions,
                                                      initialization=False)
                           )
        if parent_1.representation.is_valid(repr_values):
            child.values = repr_values
            individuals[idx] = child
            idx += 1

    return Population(size=n, representation='permutation', individuals=individuals)


# Permutation representation
def _order_crossover(population: Population, n: int, **kwargs):
    """
    Order crossover for permutation representations. This operator transmit information about the relative order from
    the second parent.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)
        cut_points = np.sort(np.random.choice(len(parent_1) - 1, size=2, replace=False) + 1)

        child_1, child_2 = _create_permutation_individuals(
            2, parent_1.representation.permissible_values, parent_1.representation.restrictions)

        repr_1 = _order_crossover_copy(parent_1, parent_2, cut_points)
        repr_2 = _order_crossover_copy(parent_2, parent_1, cut_points)

        if parent_1.representation.is_valid(repr_1):
            child_1.values = repr_1
            individuals[idx] = child_1
            idx += 1
            if idx == n: break

        if parent_1.representation.is_valid(repr_2):
            child_2.values = repr_2
            individuals[idx] = child_2
            idx += 1

    return Population(size=n, representation='permutation', individuals=individuals)


# Permutation representation
def _cycle_crossover(population: Population, n: int, **kwargs):
    """
    Cycle crossover operator for permutation representations. This operator aims to preserve as much as possible about
    the absolute position in which the elements in the parents occur.

    Parameters
    ----------
    :param population: beagle.Population
        Mating pool.
    :param n: int
        Size of the offspring.

    Returns
    -------
    :return beagle.Population
        Offspring generated from mating pool.
    """
    length_population = len(population)
    individuals = [None] * n
    idx = 0

    while idx < n:
        parent_1, parent_2 = _get_parents(population, length_population)

        cycles = _identify_cycles(parent_1, parent_2)

        repr_1, repr_2 = [], []
        for j, indices in enumerate(cycles):  # combine cycles
            if j % 2:
                repr_1.append(np.array(parent_1.values)[indices].tolist())
                repr_2.append(np.array(parent_2.values)[indices].tolist())
            else:
                repr_1.append(np.array(parent_2.values)[indices].tolist())
                repr_2.append(np.array(parent_1.values)[indices].tolist())

        # transform 2d array to 1d
        repr_1 = [val for arr in repr_1 for val in arr]
        repr_2 = [val for arr in repr_2 for val in arr]

        child_1, child_2 = _create_permutation_individuals(
            2, parent_1.representation.permissible_values, parent_1.representation.restrictions)

        if parent_1.representation.is_valid(repr_1):
            child_1.values = repr_1
            individuals[idx] = child_1
            idx += 1
            if idx == n: break

        if parent_1.representation.is_valid(repr_2):
            child_2.values = repr_2
            individuals[idx] = child_2
            idx += 1

    return Population(size=n, representation='permutation', individuals=individuals)


RECOMBINATION_SCHEMAS = {
    'one_point_b': _n_point_crossover_b,        # binary
    'n_point_b': _n_point_crossover_b,          # binary
    'uniform_b': _uniform_crossover_b,          # binary
    'one_point_i': _n_point_crossover_i,        # integer
    'n_point_i': _n_point_crossover_i,          # integer
    'uniform_i': _uniform_crossover_i,          # integer
    'arithmetic': _arithmetic_recombination,    # real-value
    'one_point_r': _n_point_crossover_r,        # real-value
    'n_point_r': _n_point_crossover_r,          # real-value
    'uniform_r': _uniform_crossover_r,          # real-value
    'blend': _blend_crossover,                  # real-value
    'pmx': _pmx_crossover,                      # permutation
    'edge-3': _edge_3_crossover,                # permutation
    'order': _order_crossover,                  # permutation
    'cycle': _cycle_crossover                   # permutation
}

DEFAULT_SCHEMAS = {
    'binary': _n_point_crossover_b,
    'integer': _n_point_crossover_i,
    'real': _blend_crossover,
    'permutation': _pmx_crossover
}


def recombination(population: Population, n: int, schema: str = None, **kwargs):
    """
    A variation operator that generates a new population from another by means of recombination events. For more
    information on available operators use:

        For one point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['one_point_b'])

        For N point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['n_point_b'])

        For uniform crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['uniform_b'])

        For one point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['one_point_i'])

        For N point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['n_point_i'])

        For uniform crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['uniform_i'])

        For arithmetic crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['arithmetic'])

        For one point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['one_point_r'])

        For N point crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['n_point_r'])

        For uniform crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['uniform_r'])

        For blend crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['blend'])

        For pmx crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['pmx'])

        For edge-3 crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['edge-3'])

        For order crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['order'])

        For cycle crossover operator
            help(beagle.RECOMBINATION_SCHEMAS['cycle'])

    Parameters
    ---------
    :param population: beagle.Population
        Population to be mutated.
    :param n: int
        Offspring length.
    :param schema: (optional) str
        Scheme to be followed. Possible schemes:
            one_point_b         (binary representation)
            n_point_b           (binary representation)
            uniform_b           (binary representation)
            one_point_i         (integer representation)
            n_point_i           (integer representation)
            uniform_i           (integer representation)
            arithmetic          (real-value representation)
            one_point_r         (real-value representation)
            n_point_r           (real-value representation)
            uniform_r           (real-value representation)
            blend               (real-value representation)
            pmx                 (permutation representation)
            edge-3              (permutation representation)
            order               (permutation representation)
            cycle               (permutation representation)

        Default schemes:
            one_point_b         (binary representation)
            one_point_i         (integer representation)
            blend               (real-value representation)
            pmx                 (permutation representation)

    :param kwargs:
        Additional arguments depending on the scheme to be used.

    Returns
    --------
    :return: beagle.Population
        Offspring population.
    """
    if schema is None:  # default parameter for representation
        return DEFAULT_SCHEMAS[population.representation](population, n, **kwargs)
    else:
        if schema in RECOMBINATION_SCHEMAS.keys():
            return RECOMBINATION_SCHEMAS[schema](population, n, **kwargs)
        else:
            raise UnrecognisedParameter(schema, 'mutation')
