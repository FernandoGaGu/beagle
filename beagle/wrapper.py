import multiprocessing as mp
from .algorithm import Algorithm


class Wrapper:
    """
    Class representing the wrapper for a set of algorithms.
    """
    def __init__(self, algorithms):
        """
        Parameters
        ----------
        :param algorithms: dict
            Dictionary with beagle.Algorithm objects.
        """
        for alg in algorithms.values():
            assert isinstance(alg, Algorithm), 'Instances received as argument must belong to the Algorithm class.'

        self._algorithms = algorithms

    @property
    def algorithms(self):
        """
        Returns
        -------
        :return list
            List of beagle.Algorithms objects.
        """
        return list(self._algorithms.values())


def parallel(*algorithms, **kwargs):
    """
    Function that allows to execute several algorithms received as an argument in parallel during the specified number
    of generations. This function returns a Wrapper object with the algorithms (accessible through the 'algorithms'
    attribute).

    Note:
        If you are designing algorithms with custom step functions and you are going to execute several algorithms
        in parallel it is recommended to avoid using the function evaluate_parallel (in case of using pre-defined
        algorithms you can use the argument 'evaluate_in_parallel' = False) unless the system resources are sufficient,
        otherwise the performance may be worse.

    Parameters
    ----------
    :param *args
        One or more Algorithms objects.
    :param generations: int
        Number of generations to run the algorithms.

    Returns
    -------
    :return beagle.Wrapper
        Wrapper object from which the algorithms can be obtained by using the 'algorithms attribute.

    Example:
        # evaluate_in_parallel=False argument is optional but recommendable
        ga_01 = be.use_algorithm(..., evaluate_in_parallel=False)

        ga_02 = be.use_algorithm(..., evaluate_in_parallel=False)

        wrapper = be.experimental.parallel(ga_01, ga_02, generations=1000)    # Execute algorithms in parallel

        be.display(wrapper.algorithms[0].report, ...)
        be.display(wrapper.algorithms[1].report, ...)
    """
    def fit_algorithm(alg_id_, alg_, gen_, return_dict_):
        return_dict_[alg_id_] = alg_.run(gen_)

    if kwargs.get('generations', None) is None:
        raise TypeError(
            "It is necessary to provide the number of generations to run the algorithms using the 'generations' parameter")

    for alg in algorithms:
        assert isinstance(alg, Algorithm), 'Instances received as argument must belong to the Algorithm class.'

    # Initializes Manager to save object states
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create the processes that will be parallelize
    processes = [
        mp.Process(target=fit_algorithm, args=(alg_id, alg, kwargs['generations'], return_dict))
        for alg_id, alg in enumerate(algorithms)
    ]

    # Start parallelization
    for p in processes:
        p.start()

    # Join results
    for p in processes:
        p.join()

    return Wrapper({alg_id: alg for alg_id, alg in return_dict.items()})
