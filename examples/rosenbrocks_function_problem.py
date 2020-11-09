"""
ROSENBROCK'S BANANA FUNCTION OPTIMIZATION (I)
---------------------------------------------
  Idea and description problem from 'https://www.johndcook.com/blog/2010/07/28/rosenbrocks-banana-function/'.

  Rosenbrock’s banana function is a famous test case for optimization software. It’s called the banana function
  because of its curved contours. The definition of the function is:

                f(x, y) = (1 - x)^2 + 100(y - x^2)^2

  This function has a global minimum at {1, 1}. Therefore the representation to be used is of real values. To make the
  example more challenging, very wide intervals will be set for the x and y values.

  This example uses pre-building 'GA1'
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
generations = 60
population_size = 1000
representation = 'real'
x_parameter_range = (-100.0, 100.0)
y_parameter_range = (-100.0, 100.0)

# Create fitness function
fitness = be.Fitness(fitness_function)

# Use a pre-defined algorithm
basic_ga_alg = be.use_algorithm(
    'GA1', fitness=fitness, population_size=population_size, individual_representation=representation,
    bounds=[x_parameter_range, y_parameter_range], alg_id='GA1'
)

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)


# To show the convergence of the algorithm the display function can be useful
be.display(basic_ga_alg.report, path='./convergence/Rosenbrocks_function(GA1).pdf')


solution = be.get_best_solution(basic_ga_alg)

print('Fitness value: ', solution.fitness)
print('X = %.5f\nY = %.5f' % (solution.values[x_idx], solution.values[y_idx]))
