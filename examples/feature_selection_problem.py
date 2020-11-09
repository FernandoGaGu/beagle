"""
FEATURE SELECTION
-----------------
  In this example code the beagle package will be used for feature selection. The selection of characteristics is
  carried out in the context of supervised learning models. In this line we start from an initial dataset with a set
  of variables. This dataset will contain redundant/irrelevant variables which usually affect negatively the performance
  of the supervised models, this is known as the curse of dimensionality. In this context we are interested in reducing
  the number of redundant/irrelevant variables as much as possible. This reduction leads firstly to an improvement in
  the performance of the classification/regression models, secondly the selection of those variables that are really
  relevant implies the generation of useful knowledge and finally the reduction in the number of variables translates
  into a reduction in the computational cost of the models. For the reasons presented it is frequent that the first
  step when facing a problem in the area of Machine Learning is to carry out a selection of characteristics to reduce
  the dimensionality.

  Evolutionary algorithms offer one of the best strategies known nowadays to deal with this type of problem while
  preserving the original meaning of the variables (unlike composition-based methods such as PCA).

  - Representation

  Each individual that forms part of the population can be represented as an array of integers of variable length
  where each number represents a variable. To evaluate the fitness of the individuals we can use a simple cross
  validation. In this way the output of the algorithm will consist of the combination of variables that give a better
  classification performance for the model provided, potentially including only those variables significant for the
  given problem.

  - Classification model

  As a classification model we will use a Bayesian Gaussian classifier since this type of classifier operates under
  the assumption of conditional independence of variables given the class and therefore is especially sensitive to
  redundant/irrelevant variables. In addition its computational cost is quite low compared to other types of supervised
  classification models.
"""
import sys
sys.path.append('..')
import beagle as be
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer    # Breast cancer dataset
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def cross_validation(x_data, y_data, model, k: int, repetitions: int, metric: str = 'accuracy') -> tuple:
    """
    Function that performs K-CV repeats with class stratification. This function receives a dataset, a classification
    model belonging to the scikit-learn package, the number of partitions to be created (k), and the number of repeats
    to be done.

    Parameters
    ----------
    :param x_data: np.array (2d)
        Predictor variables.
    :param y_data: np.array (1d)
        Class labels.
    :param model: sklearn.base.BaseEstimator
        Supervised model.
   :param k: int
        Number of partitions for cross-validation.
    :param repetitions: int
        Number of repetitions for cross-validation.
    :param metric: str (default accuracy)
        Metric to compute.

    Returns
    -------
    :return: tuple
        Tuple where the first element corresponds to the value of the metric and the second to the standard deviation
        of each test dataset.
    """

    metric = metric.lower()

    available_metrics = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'recall': recall_score,
        'precision': precision_score
    }

    if metric not in available_metrics: raise TypeError('Metric %s not available' % metric)

    rep_k_fold = RepeatedStratifiedKFold(n_splits=k, n_repeats=repetitions)

    metric_values = np.zeros(k*repetitions)

    for i, (train, test) in enumerate(rep_k_fold.split(x_data, y_data)):
        trained_model = model.fit(x_data[train], y_data[train])
        y_pred = trained_model.predict(x_data[test])
        metric_values[i] = available_metrics[metric](y_true=y_data[test], y_pred=y_pred)

    return np.mean(metric_values), np.std(metric_values)


def fitness_function(values) -> float:
    """
    Fitness function that performs the cross validation with repetitions and class stratification and returns a
    single value incorporating information on the standard deviation of the cross validation.
    Using a fitness value between 0 and 1 may cause the selective pressure to disappear when individuals start to have a
    fitness close to 1. Additionally, it may be that the cross validation will be especially bad for one of the
    partitions. Therefore in order to solve the first problem the value returned by the cross validation will be modeled
    by the function x^2 - std(x).
    """

    mean, std = cross_validation(x_data=X_DATA[:, values], y_data=Y_DATA, model=supervised_model, k=CROSS_VALIDATION_K,
                                 repetitions=CROSS_VALIDATION_REPETITIONS, metric=CROSS_VALIDATION_METRIC)

    return mean**2 - std


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
    6. Annihilate worst individuals in offspring and replace them with the best.
    7. Merge elite and offspring -> Population for next generation
    """
    # Put parameters
    recombination_schema = 'one_point_i'         # Alternatives: 'n_point_i' or 'uniform_i'
    mutation_schema = 'random_resetting'         # Alternatives: 'creep'
    mutation_probability = 0.1
    max_mutation_events = 2
    ranking_selection_schema = 'tournament'      # Alternatives: 'roulette' or 'sus'
    tournament_k = 2
    tournament_w = 1
    tournament_replacement = False
    elitism_percentage = 0.1

    # Get elite
    elite = be.survivor_selection(population=parents, schema='elitism', select=elitism_percentage)

    # Apply selection to get the mating pool
    mating_pool = be.ranking_selection(
        population=parents, n=len(parents) - len(elite), schema=ranking_selection_schema,
        w=tournament_w, k=tournament_k, replacement=tournament_replacement)

    # Generate offspring
    offspring = be.recombination(population=mating_pool, n=len(mating_pool), schema=recombination_schema)

    # Mutate offspring
    be.mutation(population=offspring, probability=mutation_probability, schema=mutation_schema,
                max_mutation_events=max_mutation_events)

    # Evaluate offspring
    be.evaluate_parallel(population=offspring, fitness_function=fitness)

    # Merge elite and offspring
    next_generation = be.merge_populations(offspring, elite)

    report.create_report(population=next_generation, population_name='Population', increment_generation=True)

    # With this indicator we keep the best solution of each generation
    return next_generation, be.SaveBestSolution(next_generation)


CROSS_VALIDATION_K = 5
CROSS_VALIDATION_REPETITIONS = 5
CROSS_VALIDATION_METRIC = 'f1'

supervised_model = GaussianNB()
data = load_breast_cancer()

X_DATA = data['data']
Y_DATA = data['target'].astype(int)   # Binary classification problem

# To make the problem more challenging we will add some noise (40 random variables)
for n in range(40):
    X_DATA = np.hstack([X_DATA, np.random.uniform(0, 10, size=X_DATA.shape[0]).reshape(-1, 1)])

# Algorithm settings
generations = 100
population_size = 100
representation = 'integer'
value_coding = {'Feature_num_%d' % n: n for n in range(X_DATA.shape[1])}
replacement = False

# Create population
initial_population = be.Population(
    size=population_size, representation=representation, value_coding=value_coding, replacement=replacement)

# Create fitness function
fitness = be.Fitness(fitness_function)

# Create a Report object to monitor the fitness of populations (we were using this object in the step function)
report = be.Report()

# Create algorithm
basic_ga_alg = be.Algorithm(
    step=step, fitness=fitness, initial_population=initial_population, alg_id='FeatureSelectionProblem')

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)

# To show the convergence of the algorithm the display function can be useful
be.display(report, path='./convergence/FeatureSelection.pdf')

# Finally it is possible to get the best solution using get_best_solution function
solution = be.get_best_solution(basic_ga_alg)

print('Solution fitness: ', solution.fitness)
print('Features: ')
for val in solution.values:

    print(val, end=" ")
print()
