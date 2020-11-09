"""
HYPERPARAMETER OPTIMIZATION
---------------------------
  In this example code the beagle package will be used for hyperparameter optimization in the context of supervised
  classification. Support Vector Machines (SVMs) from the scikit-learn framework will be used as a model for supervised
  classification. In order to introduce a greater number of parameters, a radial-basis function (Gaussian) kernel will
  be used. In this context, the hyperparameters to be optimised will be:

    1. Regularization parameter C (L2 regularization).
    2. Gamma parameter.

  For this purpose a fitness function will be defined which will generate a new SVM model based on the parameters
  received as an input (individual genotype). This model will be used on a dataset with labelled examples (supervised
  classification) and the classification performance, calculated by 5 cross validation repetitions, will be returned.
  Using a fitness value between 0 and 1 may cause the selective pressure to disappear when individuals start to have a
  fitness close to 1. Additionally, it may be that the cross validation will be especially bad for one of the
  partitions. Therefore in order to solve the first problem the value returned by the cross validation will be modeled
  by the function (x)^2 - std(x). Similarly, pretending to achieve solutions with a more homogeneous generalization 
  capacity (in terms of performance between each of the cross validation partitions) the standard deviation of the cross
  validation process will be discounted to the fitness value.

- Representation

  Regarding representation, we must consider that we are dealing with real values. Therefore the representation of
  individuals is quite clear. In this case we only have to choose the range of values to explore.
"""
import sys
sys.path.append('..')
import beagle as be
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer


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
    Fitness function that defines the SVM model based on the parameters contained in the individual's genotype,
    performs the cross validation with repetitions and class stratification and returns a single value incorporating
    information on the standard deviation of the cross validation process as indicated in the description.
    """
    C_idx = 0
    gamma_idx = 1

    model = SVC(kernel='rbf', C=values[C_idx], gamma=values[gamma_idx])

    mean, std = cross_validation(x_data=X_DATA, y_data=Y_DATA, model=model, k=CROSS_VALIDATION_K,
                                 repetitions=CROSS_VALIDATION_REPETITIONS, metric=CROSS_VALIDATION_METRIC)

    return mean**10


def step(parents: be.Population, fitness: be.Fitness) -> tuple:
    """
    The step function defines how an algorithm generation will be conducted. This function must receive a population and
    a fitness object and return another population. In this case we will define the parameters of the algorithm within
    the function itself and use report objects to monitor the evolution of the population.
    In this algorithm for the selection of individuals we will use fitness proportional selection. We will additionally
    replace individuals who have been in the population for more than 5 generations with the best individuals from the
    offspring (those selected through the selection strategy). In each generation we will keep the combination of
    hyperparameters that gives the best performance by using a SolutionFound Indicator.

    The operators' arguments will be left by default.
    """
    recombination_schema = 'blend'              # Other possible options are: 'one_point_r', 'n_point_r', 'arithmetic' or 'uniform_r'
    mutation_schema = 'uncorrelated_one_step'   # Other possible options are: 'uniform', 'non_uniform' or 'uncorrelated_n_step'
    selection_schema = 'sigma_scaling'          # Other possible options are: 'fps' or 'win_dowing'

    # -- ALGORITHM STEPS -- #

    # Increment population age
    parents.increment_age()

    # Generate offspring (offspring size == parents size)
    offspring = be.recombination(population=parents, n=len(parents), schema=recombination_schema)

    # Mutate offspring
    be.mutation(population=offspring, schema=mutation_schema)

    # Evaluate offspring fitness (For those problems where the evaluation of individuals is very time consuming
    # ,or when the population size is very large, the parallel evaluation of individuals reduces considerably
    # the time required for this process).
    be.evaluate_parallel(population=offspring, fitness_function=fitness)

    # Best offspring
    best_offspring = be.fitness_proportional_selection(population=offspring, n=int(len(parents)/2),
                                                       schema=selection_schema)

    parents.replace_older_individuals(new_individuals=best_offspring)

    # Create the population report
    report.create_report(population=parents, population_name='Population', increment_generation=True)

    return parents, be.SaveBestSolution(parents)


# Define random seed
np.random.seed(1997)

CROSS_VALIDATION_K = 2
CROSS_VALIDATION_REPETITIONS = 5
CROSS_VALIDATION_METRIC = 'accuracy'

# Load breast dataset
data = load_breast_cancer()
X_DATA = data['data']
Y_DATA = data['target']

indices = np.random.choice(569, size=200)  # Take only 200 random examples
X_DATA = X_DATA[indices, :]
Y_DATA = Y_DATA[indices]

# Algorithm parameters
generations = 100
population_size = 50
representation = 'real'
C_parameter_range = (0.01, 20.0)
gamma_parameter_range = (1e-03, 100)

# Create population
initial_population = be.Population(size=population_size, representation=representation,
                                   bounds=[C_parameter_range, gamma_parameter_range])

# Create fitness function
fitness = be.Fitness(fitness_function)

# Create a Report object to monitor the fitness of populations (we were using this object in the step function)
report = be.Report()

# Create algorithm
basic_ga_alg = be.Algorithm(step=step, fitness=fitness, initial_population=initial_population,
                            alg_id='HyperparameterOptimization')

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)

# Finally, to show the convergence of the algorithm the display function can be useful
be.display(report, path='./convergence/HyperparameterOptimization.pdf')

solution = be.get_best_solution(basic_ga_alg)

print('Fitness value: ', solution.fitness)
print('C = %.5f\nGamma = %.5f' % (solution.values[0], solution.values[1]))
