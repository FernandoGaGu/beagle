"""
KNAPSACK PROBLEM
--------------------
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


# Definition of the steps of the algorithm
def step(parents: be.Population, fitness: be.Fitness) -> tuple:
    """
    The step function defines how an algorithm generation will be conducted. This function must receive a population and
    a fitness object and return another population. In this case we will define the parameters of the algorithm within
    the function itself and use report objects to monitor the evolution of the population.

    In this algorithm the main steps consists of:

    1. Get elite -> Elite
    2. Apply tournament selection -> Best individuals
    3. Apply one point cross over to best individuals -> Offspring
    4. Mutate offspring
    5. Evaluate offspring
    6. Merge elite and offspring -> Population for next generation
    """
    # Put parameters
    recombination_schema = 'one_point_b'         # Alternatives: 'n_point_b' or 'uniform_b'
    mutation_schema = 'bit_flip'
    mutation_probability = 0.1
    ranking_selection_schema = 'tournament'      # Alternatives: 'roulette' or 'sus'
    tournament_k = 2
    tournament_w = 1
    tournament_replacement = False
    elitism_percentage = 0.2

    # Get elite
    elite = be.survivor_selection(population=parents, schema='elitism', select=elitism_percentage)

    # Apply selection to get the mating pool
    mating_pool = be.ranking_selection(
        population=parents, n=len(parents) - len(elite), schema=ranking_selection_schema,
        w=tournament_w, k=tournament_k, replacement=tournament_replacement)

    # Generate offspring
    offspring = be.recombination(population=mating_pool, n=len(mating_pool), schema=recombination_schema)

    # Mutate offspring
    be.mutation(population=offspring, probability=mutation_probability, schema=mutation_schema)

    # Evaluate offspring
    be.evaluate_parallel(population=offspring, fitness_function=fitness)

    # Merge elite and offspring
    next_generation = be.merge_populations(offspring, elite)

    report.create_report(population=next_generation, population_name='Population', increment_generation=True)

    # With this indicator we keep the best solution of each generation
    return next_generation, be.SaveBestSolution(next_generation)


np.random.seed(1997)   # Random seed

# Problem definition
num_items = 100
items = np.arange(num_items)
item_utility = np.random.uniform(low=1, high=10, size=num_items)   # Utility values range, for example, from 1 to 10
item_costs = np.random.uniform(low=0, high=1, size=num_items)
maximum_cost = 50

# Algorithm parameters
generations = 200
population_size = 1000
representation = 'binary'

# Create population
initial_population = be.Population(size=population_size, representation=representation, length=num_items)

# Create fitness function
fitness = be.Fitness(fitness_function)

# Create a Report object to monitor the fitness of populations (we were using this object in the step function)
report = be.Report()

# Create algorithm
basic_ga_alg = be.Algorithm(step=step, fitness=fitness, initial_population=initial_population, alg_id='KnapsackProblem')

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)

# To show the convergence of the algorithm the display function can be useful
be.display(report, path='./convergence/KnapsackProblem.pdf')

# Finally it is possible to get the best solution using get_best_solution function
solution = be.get_best_solution(basic_ga_alg)

print('Solution fitness: ', solution.fitness)
print('Items: ')
for idx, val in enumerate(solution.values):
    if val == 1:
        print(idx, end=", ")
print()
