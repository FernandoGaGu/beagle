"""
N-QUEENS PROBLEM
--------------------
  The problem is to position N queens on an N x N chess board so that none of the 8 queens is capable of killing
  any other.

  In this problem each solution, that is to say each individual in the algorithm, is a combination of positions in which
  to place each queen. Many of the possible solutions will break the condition of the problem since two or more queens
  will be able to kill each other. Therefore the quality of a genotype will be represented by the number of queens that
  can kill each other. So the lower this number the higher the fitness of the individuals. This problem can be adapted
  to a maximisation problem simply by considering the inverse of the fitness value.

- Representation

  The genotype will be represented as a permutation. Initially we know that there cannot be more than one queen in the
  same row or column as this directly violates the restriction of the problem. Therefore, we can assume that there will
  be a queen in each column and represent the problem as a permutation between rows. In this way, the genotype
  G = (1, 2, 3, 4, 5, 6, 7, 8) represents all queens placed in the diagonal (considering an N of 8).
  Thus, the problem consists in minimizing the number of diagonal restrictions since with the representation we are
  assuring that the horizontal and vertical restrictions are met.

  The steps followed to solve the problem are as follows:

    1. Choosing how to represent the problem (described above).

    2. Designing the fitness function. This is perhaps the most laborious step for many problems, that is, defining how 
       to evaluate each of the possible solutions. In this case the difficulty lies in how to determine which queens are 
       capable of killing each other through the diagonal of the board.

    3. Define the steps carried out by the algorithm.

As finding a single solution is relatively trivial in this script an adaptation has been made to look for all possible 
solutions. This way when a solution has been found it is saved and all individuals who have such a solution will get the 
minimum possible fitness. 
"""
import sys
sys.path.append('..')
import beagle as be
import numpy as np

solutions = []


# Definition of the fitness function
def fitness_function(values) -> float:
    """
    This fitness function counts the number of check mates based on the permutation received as an argument and returns
    the inverse of that value raised to 3 in order to treat the problem as a maximization problem.
    """
    def up_right(idx: int, val: int):
        idx += 1
        val -= 1

        return idx, val

    def up_left(idx: int, val: int):
        idx -= 1
        val -= 1
        
        return idx, val

    def bottom_right(idx: int, val: int):
        idx += 1
        val += 1
        
        return idx, val

    def bottom_left(idx: int, val: int):
        idx -= 1
        val += 1
        
        return idx, val

    def movement(idx: int, val: int, direction: str):
        directions = {
            'ur': up_right,
            'ul': up_left,
            'br': bottom_right,
            'bl': bottom_left
        }

        return directions[direction](idx, val)

    def count_checks(idx: int, val: int, values, dashboard_size: tuple) -> int:
        """Function that returns the number of checkmates for a given genotype position"""
        movements = ['ur', 'ul', 'br', 'bl']

        number_of_checks = 0

        for mov in movements:
            current_idx = idx
            current_val = val

            covered_positions = []

            while True:  # Add all diagonal positions

                current_idx, current_val = movement(current_idx, current_val, mov)

                # Out of the dashboard
                if current_idx < 0 or current_val < 0 or current_idx == dashboard_size[0] or \
                        current_val == dashboard_size[1]:  
                    break

                covered_positions.append((current_idx, current_val))

            for idx_, val_ in enumerate(values):

                if idx_ == idx: continue   # Same element

                if (idx_, val_) in covered_positions:
                    number_of_checks += 1

        return number_of_checks

    if values in solutions:   # Solution already found
        return 0.0

    total_checks = 0

    for index, value in enumerate(values):
        checks = count_checks(index, value, values, DASHBOARD_SIZE)
        total_checks += checks
    
    if total_checks > 0:
        return 1 / total_checks**3
    else:
        return np.inf


# Definition of the steps of the algorithm
def step(parents: be.Population, fitness: be.Fitness) -> tuple:
    """
    The step function defines how an algorithm generation will be conducted. This function must receive a population and 
    a fitness object and return another population. In this case we will define the parameters of the algorithm within 
    the function itself and use report objects to monitor the evolution of the population.
    In this algorithm for the selection of individuals we will use ranking selection and survival selection strategies. 
    Another option would be to employ strategies such as proportional selection to fitness. In survivor selection we are 
    going to use both elitism and annihilation.
    """
    recombination_schema = 'edge-3'             # Other possible options are: 'pmx', 'order' or 'cycle'
    mutation_schema = 'inversion'               # Other possible options are: 'swap', 'insert' or 'scramble'
    mutation_probability = 0.3 
    mutation_possible_events = 3
    ranking_selection_schema = 'tournament'     # Other possible options for ranking selection are: 'sus' or 'roulette'
    tournament_k = 2
    tournament_w = 1
    tournament_replacement = False
    elite_size = 0.1                            # Select the 10% of the best individuals for the next generation
    annihilation_size = 0.1                     # Remove the 10% of the least-fitted individuals

    # -- ALGORITHM STEPS -- #

    # Generate offspring (offspring size == parents size)
    offspring = be.recombination(population=parents, n=len(parents), schema=recombination_schema)

    # Mutate offspring
    be.mutation(population=offspring, probability=mutation_probability,
                possible_events=mutation_possible_events, schema=mutation_schema)

    # Evaluate offspring fitness
    be.evaluate(population=offspring, fitness_function=fitness)

    # Merge offspring and parents
    parents_offspring = be.merge_populations(parents, offspring)

    # Select elite population
    elite = be.survivor_selection(population=parents_offspring, schema='elitism', select=elite_size)

    # Annihilate least-fitted individuals
    parents_offspring = be.survivor_selection(
        population=parents_offspring, schema='annihilation', annihilate=annihilation_size)

    # Apply ranking selection (by selecting a population with a similar size to the parents minus the size of the elite)
    next_generation = be.ranking_selection(
        population=parents_offspring, n=len(parents) - len(elite), schema=ranking_selection_schema,
        w=tournament_w, k=tournament_k, replacement=tournament_replacement)

    # Adding the elite to the next generation population
    next_generation = be.merge_populations(next_generation, elite)

    # Create the population report
    report.create_report(population=next_generation, population_name='Basic GA population', increment_generation=True)

    # If we only wanted to return the first solution found, we could return an EarlyStopping object, which will indicate
    # to the algorithm that the execution is finished
    for individual in next_generation:
        if individual.fitness[0] == np.inf:
            return next_generation, be.EarlyStopping(individual)

    return next_generation, None


# Define random seed
np.random.seed(1997)

N = 20   # Chessboard size
DASHBOARD_SIZE = (N, N)

# Algorithm parameters
generations = 200
population_size = 100
representation = 'permutation'
representation_events = [n for n in range(N)]

# Create population
initial_population = be.Population(size=population_size, representation=representation, events=representation_events)

# Create fitness function
fitness = be.Fitness(fitness_function)

# Create a Report object to monitor the fitness of populations (we were using this object in the step function)
report = be.Report()

# Create algorithm
basic_ga_alg = be.Algorithm(step=step, fitness=fitness, initial_population=initial_population, alg_id='EightQueens_1')

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)


if basic_ga_alg.solution_found:
    print('Solution found:', basic_ga_alg.solutions[0].values)
else:
    print('Not solution found')

# Finally, to show the convergence of the algorithm the display function can be useful
be.display(report, path='./convergence/NQueens.pdf')
