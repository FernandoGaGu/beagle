from .fitness import Fitness, evaluate, evaluate_parallel
from .population import Population, merge_populations
from .mutation import mutation, MUTATION_SCHEMAS
from .recombination import recombination, RECOMBINATION_SCHEMAS
from .selection import uniform_selection, ranking_selection, fitness_proportional_selection, survivor_selection, SELECTION_SCHEMAS
from .algorithm import Algorithm, get_best_solution
from .report import Report, MOEAReport, EarlyStopping, SolutionFound, SaveBestSolution
from .visualization import display, display_w
from .building.loader import use_algorithm, AVAILABLE_BUILDINGS
from .representation import Representation, Binary, Integer, RealValue, Permutation
from .wrapper import parallel
from .utils.loader import save_population, load_population
from .utils.moea import pareto_front
from .building.nsga2 import nsga2
from .building.spea2 import spea2


__all__ = ['building']
