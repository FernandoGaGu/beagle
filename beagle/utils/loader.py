import os
import pickle
from ..population import Individual, Population
from ..algorithm import Algorithm


def save_population(file: str, beagle_obj: Algorithm or Population):
    """
    Save the population into a file.

    Parameters
    ----------
    :param file: str
        Name of the output file. The output fill will contain the extension .be.
    :param beagle_obj: beagle.Algorithm or beagle.Population
        Algorithm.
    """
    if file.split('.')[-1] != 'be':   # Add .be extension to the file if it doesn't include this extension
        file += '.be'

    file_abs_path = os.path.abspath(file)

    if os.path.exists(file_abs_path):   # Update file name if the file already exists
        print('File %s already exists, name changed to' % file_abs_path, end=' ')
        file_abs_path = _update_file_name(file_abs_path)
        print(file_abs_path)

    with open(file_abs_path, 'wb') as out:
        if isinstance(beagle_obj, Algorithm):           # Save population from Algorithm object
            pickle.dump(beagle_obj.population, out)

        elif isinstance(beagle_obj, Population):          # Save population from Population object
            pickle.dump(beagle_obj, out)

        else:
            raise TypeError(
                'beagle_obj must be an instance of beagle.Population or beagle.Algorithm. Provided: ',
                str(type(beagle_obj)))


def load_population(file: str) -> Population:
    """
    Load a population from file.

    Parameters
    ----------
    :param file: str
        File name, this is a pickle object previously saved using beagle.save_population()
    """
    file_abs_path = os.path.abspath(file)

    assert os.path.exists(file), 'File %s not found.' % file_abs_path

    try:
        return pickle.load(open(file_abs_path, 'rb'))
    except Exception:
        raise Exception('Something goes wrong loading %s. Impossible to load file.' % file_abs_path)


# Private functions used for the above implemented functions
def _update_file_name(file: str):
    """
    Change the name of the file to avoid overwrite another existing file with the same name.
    """
    n = 1
    path_to_file = '/'.join(file.split('/')[:-1])
    file_name = file.split('/')[-1].split('.')[0]

    while os.path.exists('%s/%s_(%d).be' % (path_to_file, file_name, n)):
        n += 1

    return '%s/%s_(%d).be' % (path_to_file, file_name, n)



