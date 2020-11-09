# beagle

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<br />
<p align="center">
	<img src="https://github.com/FernandoGaGu/beagle/blob/main/img/beagleLogo.png" width="450" height="200" alt="Logo" >
</p>

> Python package with flexible implementations of genetic algorithms with support for different types of representations.

## Description

The beagle package is a native python implementation of evolutionary algorithms. This package has been designed with the idea of providing the user all the flexibility needed to experiment with different approaches of evolutionary algorithms. On the other hand, for those cases in which the flexibility provided by this library is not required, beagle also includes basic predefined algorithms as well as more sophisticated ones such as NSGAII or SPEA2. 

For more information about the module and its components, consult the Wiki pages.

## Install

### Dependencies

PyWinEA requires:
- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- tqdm (>= 4.42.1)
- matplotlib (>= 3.1.3)

```sh
git clone https://github.com/FernandoGaGu/beagle
```
(Soon available through pip)

## Usage

Examples of the basic use of the package can be found in the examples directory. For more advanced use it is recommended to look at the documentation and Wiki pages. 

The following is a basic example of maximizing a function from the file *rosenbrocks_function_problem.py*:

```python
import beagle as be

def fitness_function(values) -> float:
    return -1*((1 - values[0])**2 + 100*(values[1] - values[0]**2)**2)

# Algorithm parameters
generations = 60
population_size = 1000
representation = 'real'
x_parameter_range = (-100.0, 100.0)
y_parameter_range = (-100.0, 100.0)
bounds = [x_parameter_range, y_parameter_range]

# Create fitness function
fitness = be.Fitness(fitness_function)

# Use a pre-defined algorithm
basic_ga_alg = be.use_algorithm('GA1', 
                                fitness=fitness, 
                                population_size=population_size, 
                                individual_representation=representation,
                                bounds=bounds, 
                                alg_id='GA1')

# ... and execute the algorithm ;)
basic_ga_alg.run(generations)
```

For more examples it is recommended to take a look at the [examples](https://github.com/FernandoGaGu/beagle/tree/master/examples/).

[contributors-shield]: https://img.shields.io/github/contributors/FernandoGaGu/beagle.svg?style=flat-square
[contributors-url]: https://github.com/FernandoGaGu/beagle/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/FernandoGaGu/beagle.svg?style=flat-square
[forks-url]: https://github.com/FernandoGaGu/beagle/network/members
[stars-shield]: https://img.shields.io/github/stars/FernandoGaGu/beagle.svg?style=flat-square
[stars-url]: https://github.com/FernandoGaGu/beagle/stargazers
[issues-shield]: https://img.shields.io/github/issues/FernandoGaGu/beagle.svg?style=flat-square
[issues-url]: https://github.com/FernandoGaGu/beagle/issues
[license-shield]: https://img.shields.io/github/license/FernandoGaGu/beagle.svg?style=flat-square
[license-url]: https://github.com/FernandoGaGu/beagle/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/GarciaGu-Fernando
[product-screenshot]: img/beagleLogo.png