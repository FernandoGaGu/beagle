"""
KNAPSACK PROBLEM (using a predefined algorithm)
-----------------------------------------------
This problem represents a generalization that can be applied to a wide variety of real-life problems.

Considering a set of 'n' elements, each one has associated a value 'v' and a cost 'c'. The objective of the problem
is to select the subset of elements that maximizes the sum of the values while keeping the cost within a certain
limit denoted as 'C'. In essence, it is about maximizing utility within a set of constraints.

- Representation

  For this problem, the candidate solutions can be represented as an array of binaries of length n, where 1 indicates
  that a certain item is included and 0 that this item is omitted.

  There are different alternatives for mapping from genotype to phenotype. In this case, the approach in which we
  reduce the maximum computational cost would be the following:

    An individual's genotype consists of an array of binaries, in order to perform genotype-to-genotype mapping we
    start reading the genotype from left to right. As we read the genotype we compute the sum of the values (utility)
    associated with the item indicated by each position and the cost. In this way, if we exceed the limit imposed by
    the maximum cost value C, we stop going through the genotype and keep the items seen up to that point (the total
    utility of the genotype). This rule could be identified as: Go through the items included in the individual until
    the cost is exceeded.

    Consequently, the fitness value of an individual will correspond to the sum of the values of the items included.
    This form of mapping ensures that there will be no invalid individuals in the population.

  First we have to define our items, values and cost. As I don't feel like writing it manually I will do it randomly
  creating a vector of length N associated to the items, and two other vectors of the same length with the utility
  value and the cost respectively. For this purpose I will use the numpy library.
  After that, defining the fitness function is something trivial.
"""
import sys
sys.path.append('..')
import beagle as be
import numpy as np


# Definition of fitness function
def fitness_function(values) -> float:
    """The fitness function does what is indicated in the description, it goes through the genotype of the individuals
    calculating the utility as long as the value of the cost is not exceeded."""
    utility, cost = 0.0, 0.0

    for i, position in enumerate(values):
        if position == 1:
            if cost < maximum_cost:
                utility += item_utility[i]
                cost += item_costs[i]
            else:
                return utility

    return utility


np.random.seed(1997)   # Random seed

# Define problem
num_items = 100
items = np.arange(num_items)
item_utility = np.random.uniform(low=1, high=10, size=num_items)   # Utility values range, for example, from 1 to 10
item_costs = np.random.uniform(low=0, high=1, size=num_items)
maximum_cost = 50

# Algorithm parameters
generations = 200
population_size = 1000
representation = 'binary'

# Create fitness function
fitness = be.Fitness(fitness_function)

# Use a pre-defined algorithm
basic_ga_alg = be.use_algorithm(
    'GA1', fitness=fitness, population_size=population_size, individual_representation=representation,
    length=num_items, alg_id='GA1'
)

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)


# To show the convergence of the algorithm the display function can be useful
be.display(basic_ga_alg.report, path='./convergence/KnapsackProblem_GA1(buildings).pdf')

# Finally it is possible to get the best solution using get_best_solution function
solution = be.get_best_solution(basic_ga_alg)

print('Solution fitness: ', solution.fitness)
print('Items: ')
for idx, val in enumerate(solution.values):
    if val == 1:
        print(idx, end=", ")
print()

