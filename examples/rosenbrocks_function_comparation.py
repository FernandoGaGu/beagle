"""
ROSENBROCK'S BANANA FUNCTION OPTIMIZATION (II))
----------------------------------------------
  This example shows how to run several genetic algorithms in parallel. More specifically, it compares the execution of
  two of the algorithms implemented in buildings.The problem of minimization of the ROSENBROCK'S function has been
  taken as a model. For more details on this problem and other examples, please consult:
        'rosenbrocks_function_problem.py' and 'rosenbrocks_function_problem(II).py'
"""
import sys
sys.path.append('..')
import beagle as be
import numpy as np


def fitness_function(values) -> float:
    """Fitness function (As this is a problem of minimization, therefore we multiply the function by -1)"""
    return -1*((1 - values[x_idx])**2 + 100*(values[y_idx] - values[x_idx]**2)**2)


# Define random seed
np.random.seed(1997)

# Index of the x and y values
x_idx = 0
y_idx = 1

# Algorithm parameters
generations = 100
population_size = 200
representation = 'real'
x_parameter_range = (-100.0, 100.0)
y_parameter_range = (-100.0, 100.0)

# Create fitness function
fitness = be.Fitness(fitness_function)


ga_01 = be.use_algorithm(
    'GA1', fitness=fitness, population_size=population_size, individual_representation=representation,
    bounds=[x_parameter_range, y_parameter_range], alg_id='GA1', elitism_select=0.2, evaluate_in_parallel=False
)

ga_02 = be.use_algorithm(
    'GA2', fitness=fitness, population_size=population_size, individual_representation=representation,
    bounds=[x_parameter_range, y_parameter_range], alg_id='GA2', elitism_select=0.2, evaluate_in_parallel=False
)

wrapper = be.parallel(ga_01, ga_02, generations=generations)

# To show the convergence of both algorithms the display_w function can be useful
be.display_w(wrapper, path='./convergence/Rosenbrocks_function_comparation_Wrapper.pdf')

