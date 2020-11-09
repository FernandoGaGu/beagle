class IncompleteArguments(Exception):
    """
    Exception raised when incomplete arguments have been provided by the user.
    """
    def __init__(self, required: list, class_name: str):
        super(IncompleteArguments, self).__init__(
            'Not all the required arguments have been provided for the initialization of the class %s. Required: %s'
            % (class_name, ', '.join(f'{a1} ({a2})' for a1, a2 in required))
        )


class UnrecognisedParameter(Exception):
    """
    Exception raised when a provided parameter hasn't been recognised.
    """
    def __init__(self, parameter, function):
        super(UnrecognisedParameter, self).__init__(
            'Parameter %s in function %s not recognised' % (parameter, function)
        )


class InsufficientReportGenerations(Exception):
    """
    Exception raised when the number of generations in a report is 1.
    """
    def __init__(self):
        super(InsufficientReportGenerations, self).__init__(
            'To display a Report object, more populations with more than one generation are required. Be sure to '
            'increase the generation number in your step function (report_instance.increment_generation) or specify '
            'the increment_generation parameter as true when using the report_instance.create_report() function.'
        )


class InconsistentLengths(Exception):
    """
    Exception raised when two arrays that should have the same length do not have it.
    """
    def __init__(self, message: str = ''):
        super(InconsistentLengths, self).__init__(
            'Trying to operate with different length arrays when both are required to be the same size. %s' % message
        )


class FitnessFunctionError(Exception):
    """
    Exception raised when when the fitness function generates an error during the evaluation of an individual.
    """
    def __init__(self, genotype):
        super(FitnessFunctionError, self).__init__(
            'Impossible to evaluate the fitness of the individual with genotype: %s' % str(genotype)
        )


class StepFunctionError(Exception):
    """
    Exception raised when an error occurs in the step function of the Algorithm class.
    """
    def __init__(self, message: str = ''):
        super(StepFunctionError, self).__init__(
            'Error in the step function of the Algorithm class. %s' % message
        )

