from ..fitness import Fitness
from ..population import Population
from ..report import Report, MOEAReport
from ..algorithm import Algorithm
from ..exceptions import UnrecognisedParameter
from .basic import basic_ga_1, basic_ga_2
from .nsga2 import nsga2
from .spea2 import spea2

NSGA2_ID = 'NSGA2'
SPEA2_ID = 'SPEA2'

AVAILABLE_BUILDINGS = {
    'GA1': basic_ga_1,
    'GA2': basic_ga_2,
    NSGA2_ID: nsga2,
    SPEA2_ID: spea2
}

_MOEAs = [NSGA2_ID, SPEA2_ID]


def use_algorithm(name: str, fitness: Fitness, **kwargs):
    """
    Function that returns a Algorithm based on a pre-defined schema.
    Currently available:

        'GA1'       for more info use:   help(beagle.AVAILABLE_BUILDINGS['GA1')
        'GA2'       for more info use:   help(beagle.AVAILABLE_BUILDINGS['GA2')

    To specify parameters for each operator it is required to prefix the process name (e.g. mutation, recombination
    or selection) with a _ and the parameter name. For example to specify the mutation probability it is necessary to
    use the argument: 'mutation_probability'.

    Parameters
    ----------
    :param name: str
        Pre-defined schema name. Available: 'GA1'
    :param fitness: beagle.Fitness
        Fitness object used to evaluate the individuals in population.
    :param initial_population: beagle.Population
        Initial population.
    :param evaluate_out_of_step: bool (optional, by default False)
        Indicates whether to make an initial evaluation of the population before calling the step() function at the
        first interaction. In most algorithms, except for multi-objective ones such as NSGA2 or SPEA2 the parameter
        will be True.
    :param kwargs:
        Parameters of the mutation, recombination and selection operators.

    Returns
    -------
    :return beagle.Algorithm
        Pre-build algorithm.
    """
    if name not in AVAILABLE_BUILDINGS: raise UnrecognisedParameter(name, 'name: str in use_algorithm()')

    initial_population = None

    if name in _MOEAs:
        kwargs['report'] = MOEAReport(num_objectives=len(fitness))   # create report for multi-objective algorithms
        kwargs['evaluate_out_of_step'] = False
    else:
        kwargs['report'] = Report()       # create report for basic algorithms

    if 'initial_population' in kwargs:
        initial_population = kwargs['initial_population']

        if not isinstance(initial_population, Population): raise TypeError('initial_population must be a Population.')

        del kwargs['initial_population']  # eliminate initial_population from kwargs

    return Algorithm(step=AVAILABLE_BUILDINGS[name], fitness=fitness, initial_population=initial_population, **kwargs)
