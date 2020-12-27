import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .exceptions import InsufficientReportGenerations
from .report import Report, MOEAReport
from .algorithm import Algorithm
from .base import ReportBase
from .wrapper import Wrapper


def display(report: Algorithm or Report, **kwargs):
    """
    Function that represents the information collected by a report generating graphs with the evolution of
    fitness in the population.

    Parameters
    ----------
    :param report: beagle.Algorithm or beagle.Report
    :param kwargs: (optional arguments)
        :key 'path'         (str) Directory to save the generated graphic.
        :key 'only_show'    (str) Indicates if the graphics are for display only or if they will also be saved. By
                                  default the graphics will be saved and not displayed.
        :key 'width'        (str) Figure width
        :key 'height'       (str) Figure height
        :key 'wspace'       (str) Horizontal pad
        :key 'hspace'       (str) Vertical pad
        :key 'legend_size'  (str) Legend size
        :key 'cmap'         (str) Matplotlib colour map name
        :key 'alpha_grid'   (str) Controls the opacity of the grid lines

    Example:

    import beagle as be

    # ... create algorithms

    alg.run(1000)

    be.display(alg, ...)
    # or
    be.display(alg.report, ...)

    """
    if isinstance(report, Algorithm):
        report = report.report

    assert isinstance(report, ReportBase), \
        'report must an instance of whether ReportBase subclass or Algorithm class. Provided: %s' % str(type(report))

    if report.current_generation == 1:
        raise InsufficientReportGenerations()

    if isinstance(report, Report):
        _display_basic(report, **kwargs)
    elif isinstance(report, MOEAReport):
        _display_moea(report, **kwargs)


def display_w(wrapper: Wrapper, **kwargs):
    """
    Function that allows to represent the convergence of an object belonging to the Wrapper class. This type of
    objects are returned when several algorithms are executed in parallel using of the parallel function.

    Parameters
    ----------
    :param wrapper: beagle.Wrapper
    :param kwargs: (optional arguments)
        :key 'width'        (str) Figure width
        :key 'height'       (str) Figure height
        :key 'wspace'       (str) Horizontal pad
        :key 'hspace'       (str) Vertical pad
        :key 'legend_size'  (str) Legend size
        :key 'cmap'         (str) Matplotlib colour map name
        :key 'alpha_grid'   (str) Controls the opacity of the grid lines

    Example:

    import beagle as be

    # ... create algorithms

    wrapper = be.parallel(alg1, alg2, alg3, alg4, generations=1000)

    be.display_w(wrapper, ...)
    """
    assert isinstance(wrapper, Wrapper), 'wrapper must be an object of the Wrapper class returned by the parallel function.'

    basic_alg, moea_alg = [], []

    for i, alg in enumerate(wrapper.algorithms):
        if isinstance(alg.report, Report):
            basic_alg.append(i)
        if isinstance(alg.report, MOEAReport):
            moea_alg.append(i)

    if basic_alg:
        _display_basic_w([wrapper.algorithms[i] for i in basic_alg], **kwargs)
    if moea_alg:
        _display_moea_w([wrapper.algorithms[i] for i in moea_alg], **kwargs)


# ----- BASIC DISPLAY FUNCTIONS ---- #
def _display_basic(report: Report, **kwargs):
    """"Function that generates the convergence plot of a basic algorithm in the display function."""

    # Optional arguments
    fig_width = kwargs.get('width', 8)
    fig_height = kwargs.get('height', 15)
    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.9)
    legend_size = kwargs.get('legend_size', 12)
    cmap_name = kwargs.get('cmap', 'viridis')
    alpha_grid = kwargs.get('alpha_grid', 0.2)

    num_plots = 3
    num_colors = 4
    output_name = kwargs.get('path', './%s.pdf' % str(time.time()))
    cmap = plt.cm.get_cmap(cmap_name, num_colors)

    titles = {
        'total': 'Total fitness',
        'average': 'Average fitness',
        'best': 'Best fitness'
    }

    fig, ax = plt.subplots(num_plots, figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    _render_report(report, ax, cmap, titles, 1, '')

    for n, sub_title in enumerate(titles.values()):
        ax[n].legend()
        ax[n].set_title(sub_title)
        ax[n].spines['top'].set_visible(False)
        ax[n].spines['right'].set_visible(False)
        handles, labels = ax[n].get_legend_handles_labels()
        ax[n].legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, prop={'size': legend_size})

        ax[n].xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer numbers in x-tick
        ax[n].grid(alpha=alpha_grid)

        ax[n].set_xlabel('Generation')

    if kwargs.get('only_show', False):
        plt.show()
    else:
        plt.savefig(output_name)


def _display_moea(report: MOEAReport, **kwargs):
    """"Function that generates the convergence plot of a MOEA in the display function."""

    # Optional arguments
    fig_width = kwargs.get('width', 8)
    fig_height = kwargs.get('height', 15)
    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.9)
    legend_size = kwargs.get('legend_size', 12)
    cmap_name = kwargs.get('cmap', 'viridis')
    alpha_grid = kwargs.get('alpha_grid', 0.2)

    num_plots = report.num_objectives + 1
    num_colors = 4
    output_name = kwargs.get('path', './%s.pdf' % str(time.time()))
    cmap = plt.cm.get_cmap(cmap_name, num_colors)

    titles = {'hypervolume': 'Hypervolume'}
    for n in range(report.num_objectives):
        titles['objective_%d' % n] = 'Fitness function %d' % n

    fig, ax = plt.subplots(num_plots, figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    _render_moea_report(report, ax, cmap, titles, 1, '')

    for n, sub_title in enumerate(titles.values()):
        ax[n].legend()
        ax[n].set_title(sub_title)
        ax[n].spines['top'].set_visible(False)
        ax[n].spines['right'].set_visible(False)
        handles, labels = ax[n].get_legend_handles_labels()
        ax[n].legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, prop={'size': legend_size})

        ax[n].xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer numbers in x-tick
        ax[n].grid(alpha=alpha_grid)

        ax[n].set_xlabel('Generation')

    if kwargs.get('only_show', False):
        plt.show()
    else:
        plt.savefig(output_name)


# ----- WRAPPER FUNCTIONS ----- #
def _display_basic_w(algorithms: list, **kwargs):
    """Function that generates the convergence plot of basic algorithms in the display_w function."""

    # Optional arguments
    fig_width = kwargs.get('width', 8)
    fig_height = kwargs.get('height', 15)
    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.9)
    legend_size = kwargs.get('legend_size', 12)
    cmap_name = kwargs.get('cmap', 'viridis')
    alpha_grid = kwargs.get('alpha_grid', 0.2)

    num_plots = 3
    num_colors = len(algorithms)
    output_name = kwargs.get('path', './%s.pdf' % str(time.time()))
    cmap = plt.cm.get_cmap(cmap_name, num_colors)

    titles = {
        'total': 'Total fitness',
        'average': 'Average fitness',
        'best': 'Best fitness'
    }

    fig, ax = plt.subplots(num_plots, figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    for report_idx, alg in enumerate(algorithms):
        if alg.report.current_generation == 1:
            raise InsufficientReportGenerations()

        _render_report(alg.report, ax, cmap, titles, report_idx, alg.id)

    for n, sub_title in enumerate(titles.values()):
        ax[n].legend()
        ax[n].set_title(sub_title)
        ax[n].spines['top'].set_visible(False)
        ax[n].spines['right'].set_visible(False)
        handles, labels = ax[n].get_legend_handles_labels()
        ax[n].legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, prop={'size': legend_size})

        ax[n].xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer numbers in x-tick
        ax[n].grid(alpha=alpha_grid)

        ax[n].set_xlabel('Generation')

    if kwargs.get('only_show', False):
        plt.show()
    else:
        plt.savefig(output_name)


def _display_moea_w(algorithms: list, **kwargs):
    """Function that generates the convergence plot of MOEAs in the display_w function."""

    # Optional arguments
    fig_width = kwargs.get('width', 9)
    fig_height = kwargs.get('height', 17)
    wspace = kwargs.get('wspace', 0.1)
    hspace = kwargs.get('hspace', 0.9)
    legend_size = kwargs.get('legend_size', 12)
    cmap_name = kwargs.get('cmap', 'viridis')
    alpha_grid = kwargs.get('alpha_grid', 0.2)

    num_plots = algorithms[0].report.num_objectives + 1
    num_colors = len(algorithms)
    output_name = kwargs.get('path', './%s.pdf' % str(time.time()))
    cmap = plt.cm.get_cmap(cmap_name, num_colors)

    titles = {
        'total': 'Total fitness',
        'average': 'Average fitness',
        'best': 'Best fitness'
    }

    fig, ax = plt.subplots(num_plots, figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    for report_idx, alg in enumerate(algorithms):
        if alg.report.current_generation == 1:
            raise InsufficientReportGenerations()

        titles = {'hypervolume': 'Hypervolume'}
        for n in range(alg.report.num_objectives):
            titles['objective_%d' % n] = 'Fitness function %d' % n

        _render_moea_report(alg.report, ax, cmap, titles, report_idx, alg.id)

    for n, sub_title in enumerate(titles.values()):
        ax[n].legend()
        ax[n].set_title(sub_title)
        ax[n].spines['top'].set_visible(False)
        ax[n].spines['right'].set_visible(False)
        handles, labels = ax[n].get_legend_handles_labels()
        ax[n].legend(
            handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, prop={'size': legend_size})

        ax[n].xaxis.set_major_locator(MaxNLocator(integer=True))  # force integer numbers in x-tick
        ax[n].grid(alpha=alpha_grid)

        ax[n].set_xlabel('Generation')

    if kwargs.get('only_show', False):
        plt.show()
    else:
        plt.savefig(output_name)


# ----- RENDER FUNCTIONS ----- #
def _render_report(report: Report, ax, cmap, titles, report_idx, alg_id: str):
    """Rendering function for Report objects"""

    populations = _get_unique_populations(report)

    total_fitness = {population: [] for population in populations}
    average_fitness = {population: [] for population in populations}
    std_fitness = {population: [] for population in populations}
    best_fitness = {population: [] for population in populations}

    for generation in report.report.keys():
        for population in populations:
            total_fitness[population].append([
                report.total_fitness(generation=generation, population=population, fitness_idx=fitness_idx)
                for fitness_idx in range(len(report.report[generation][population][0].fitness))
            ])

            average_fitness[population].append([
                report.average_fitness(generation=generation, population=population, fitness_idx=fitness_idx)
                for fitness_idx in range(len(report.report[generation][population][0].fitness))
            ])

            std_fitness[population].append([
                report.std_fitness(generation=generation, population=population, fitness_idx=fitness_idx)
                for fitness_idx in range(len(report.report[generation][population][0].fitness))
            ])

            best_fitness[population].append([
                report.best_fitness(generation=generation, population=population, fitness_idx=fitness_idx)
                for fitness_idx in range(len(report.report[generation][population][0].fitness))
            ])

    for population in populations:
        total_fitness[population] = np.array(total_fitness[population]).T
        average_fitness[population] = np.array(average_fitness[population]).T
        std_fitness[population] = np.array(std_fitness[population]).T
        best_fitness[population] = np.array(best_fitness[population]).T

    generations = [n for n in range(report.current_generation - 1)]

    idx_cmap = 0

    for i, population in enumerate(populations):
        for j, (total_f, average_f, std_f, best_f) in enumerate(
                zip(total_fitness[population], average_fitness[population],
                    std_fitness[population], best_fitness[population])):
            ax[0].plot(
                generations, total_f, color=cmap(idx_cmap + report_idx),
                label='%s %s (fitness %d) %s' % (titles['total'], population, j, alg_id)
            )

            ax[1].plot(
                generations, average_f, color=cmap(idx_cmap + report_idx),
                label='%s %s (fitness %d) %s' % (titles['average'], population, j, alg_id)
            )

            ax[1].fill_between(
                generations, y1=average_f - std_f, y2=average_f + std_f,
                color=cmap(idx_cmap + report_idx), alpha=0.1,
            )

            ax[2].plot(
                generations, best_f, color=cmap(idx_cmap + report_idx),
                label='%s %s (fitness %d) %s' % (titles['best'], population, j, alg_id)
            )

            idx_cmap += 1


def _render_moea_report(report: MOEAReport, ax, cmap, titles, report_idx, alg_id: str):
    """Rendering function for MOEAReport objects"""

    populations = _get_unique_populations(report)

    hypervolume = {population: [] for population in populations}
    fitness_values = {population: [] for population in populations}

    for generation in report.report.keys():
        for population in populations:
            hypervolume[population].append([
                report.hypervolume(generation=generation, population=population)
            ])
            fitness_values[population].append(
                report.fitness_values(generation=generation, population=population)
            )

    for population in populations:
        hypervolume[population] = np.array(hypervolume[population]).T
        fitness_values[population] = np.array(fitness_values[population]).T

    generations = [n for n in range(report.current_generation - 1)]

    idx_cmap = 0

    # Plot hypervolume convergence
    for i, population in enumerate(populations):
        for hypervol in hypervolume[population]:
            # Hypervolume
            ax[0].plot(
                generations, hypervol, color=cmap(idx_cmap + report_idx),
                label='%s of %s (max %.2f) %s' % (titles['hypervolume'], population, np.max(hypervol), alg_id)
            )

        idx_cmap += 1

    idx_cmap = 0

    # Plot fitness values convergence
    for i, population in enumerate(populations):
        for j, fitness in enumerate(fitness_values[population]):

            ax[j+1].plot(
                generations, fitness, color=cmap(idx_cmap + report_idx),
                label='%s of %s (fitness %d) %s' % (titles['objective_%d' % j], population, j, alg_id)
            )

        idx_cmap += 1


# ----- AUXILIARY FUNCTIONS ----- #
def _get_unique_populations(report: Report or MOEAReport) -> list:
    """
    Return a list of the names of the populations present in the report.

    Parameters
    ----------
    :param report: beagle.Report or beagle.MOEAReport

    Returns
    -------
    :return: list
        List of string corresponding to the unique population names.
    """
    unique_populations = []

    for population in report.report.values():
        for key in population.keys():

            if key not in unique_populations:
                unique_populations.append(key)

    return unique_populations
