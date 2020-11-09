from ..population import Individual


# Static decorator to create function static variables (used in selection operators)
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# Helping function for mutation and recombination operators
def adjust_limits(individual: Individual, idx: int, min_bound: int or float, max_bound: int or float):
    """
    Function that adjust the new value to the required interval of values.
    """
    if individual[idx] < min_bound:
        individual[idx] = min_bound
    elif individual[idx] > max_bound:
        individual[idx] = max_bound
