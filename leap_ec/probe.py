""" Probes are pipeline operators to instrument state that passes through the
pipeline such as populations or individuals.
"""
import csv
import sys
import json
import contextlib

from typing import Dict, Iterator

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from leap_ec.util import wrap_curry
from leap_ec import Individual
from leap_ec.global_vars import context
from leap_ec import ops as op
from leap_ec.ops import iteriter_op, listlist_op


##############################
# print_probe
##############################
@wrap_curry
@listlist_op
def print_probe(population, probe, stream=sys.stdout, prefix=''):
    """ pipeline operator for printing the given population

    This is really a wrapper around `probe` that, itself, gets passed te
    entire population.

    The optional prefix is used to tag the output.  For example, you may want
    to print 'before' to indicate that the population is before an operator
    is applied.

    :param population: to be printed
    :param probe: secondary probe that gets the poplation as input and for
        which the output is passed to `stream`
    :param stream: to write output
    :param prefix: optional string prefix to prepend to output
    :return: population
    """
    val = prefix + str(probe(population))
    stream.write(val)
    return population


##############################
# print_individual
##############################
@wrap_curry
@iteriter_op
def print_individual(next_individual: Iterator, prefix='',
                     stream=sys.stdout) -> Iterator:
    """ Just echoes the individual from within the pipeline

    Uses next_individual.__str__

    :param next_individual: iterator for next individual to be printed
    :param prefix: prefix appended to the start of the line
    :param stream: File object passed to print
    :return: the same individual, unchanged
    """

    while True:
        individual = next(next_individual)

        print(f'{prefix}{individual!s}', file=stream)

        yield individual


###############################
# Function print_population
###############################
def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population: The population of individuals to be printed
    :param generation: The generation of the population
    :return: None
    """
    for individual in population:
        print(generation, individual.genome, individual.fitness)


##############################
# Class BestSoFarProbe
##############################
class BestSoFarProbe(op.Operator):
    """  This probe takes an list of individuals as input and will track the
         best-so-far (BSF) individual across all the population it has seen.

        Insert an object of this class into a pipeline to have it track the
        the best individual it sees so far.  It will write the current best
        individual for each __call__ invocation to a given stream in CSV
        format.

        Like many operators, this operator checks the context object to
        retrieve the current generation number for output purposes.

        >>> from leap_ec import context, data
        >>> from leap_ec import probe
        >>> pop = data.test_population
        >>> context['leap']['generation'] = 12

        The probe will write its output to the provided stream (default is stdout,
        but we illustrate here with a StringIO stream):

        >>> import io
        >>> stream = io.StringIO()
        >>> probe = BestSoFarProbe(stream=stream)
        >>> new_pop = probe(pop)
        >>> print(stream.getvalue()) # doctest: +NORMALIZE_WHITESPACE
        step,bsf
        12,4
        <BLANKLINE>

        This operator does not change the state of the population:
        >>> new_pop == pop
        True

    """
    def __init__(self, stream=sys.stdout, header=True, context=context):
        """

        :param stream: to which to write best-so-far individuals
        :param header: True if want CSV header
        :param context: from which we get current generation (step)
        """
        self.bsf = None
        self.context = context
        self.writer = csv.DictWriter(stream, fieldnames=['step', 'bsf'])

        if header:
            self.writer.writeheader()

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        ind = max(population)
        assert(ind.fitness is not None), f"Probe expects individuals to have fitness values, but found one that doesn't."
        if self.bsf is None or (ind > self.bsf):
            self.bsf = ind

        self.writer.writerow({'step': self.context['leap']['generation'],
                            'bsf' : self.bsf.fitness
                            })

        return population


##############################
# Class BestSoFarIterProbe
##############################
class BestSoFarIterProbe(op.Operator):
    """  This probe takes an iterator as input and will track the
         best-so-far (BSF) individual in the all the individuals it sees.

        Insert an object of this class into a pipeline to have it track the
        the best individual it sees so far.  It will write the current best
        individual for each __call__ invocation to a given stream in CSV
        format.

        Like many operators, this operator checks the context object to
        retrieve the current generation number for output purposes.

        >>> from leap_ec import context, data
        >>> from leap_ec import probe
        >>> pop = data.test_population
        >>> context['leap']['generation'] = 12


        The probe will write its output to the provided stream (default is stdout,
        but we illustrate here with a StringIO stream):

        >>> import io
        >>> stream = io.StringIO()
        >>> probe = BestSoFarIterProbe(stream=stream)
        >>> bsf_output_iter = probe(iter(pop))
        >>> x = next(bsf_output_iter)
        >>> x = next(bsf_output_iter)
        >>> x = next(bsf_output_iter)
        >>> print(stream.getvalue()) # doctest: +NORMALIZE_WHITESPACE
        step,bsf
        12,...
        12,...
        12,...
        <BLANKLINE>

    """
    def __init__(self, stream=sys.stdout, header=True, context=context):
        """

        :param stream: to which to write best-so-far individuals
        :param header: True if want CSV header
        :param context: from which we get current generation (step)
        """
        self.bsf = None
        self.context = context
        self.writer = csv.DictWriter(stream, fieldnames=['step', 'bsf'])

        if header:
            self.writer.writeheader()

    def __call__(self, next_individual):
        assert (next_individual is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        while True:
            ind = next(next_individual)
            assert(ind.fitness is not None), f"Probe expects individuals to have fitness values, but found one that doesn't."
            if self.bsf is None or (ind > self.bsf):
                self.bsf = ind

            self.writer.writerow({'step': self.context['leap']['generation'],
                                'bsf' : self.bsf.fitness
                                })

            yield ind


##############################
# Class FitnessStatsCSVProbe
##############################
class FitnessStatsCSVProbe(op.Operator):
    """A probe that records basic fitness statistics for a population
    to a text stream in CSV format.

    This is meant to capture the "bread and butter" values you'll typically
    want to see in any population-based optimization experiment.  If you
    want additional columns with custom values, you can pass in a dict of
    `notes` with constant values or `extra_metrics` with functions to
    compute them.

    :param stream: the file object to write to (defaults to sys.stdout)
    :param header: whether to print column names in the first line
    :param extra_metrics: a dict of `'column_name': function` pairs, to compute
        optional extra columns.  The functions take a the population as input
        as a list of individuals, and their return value is printed in the column.
    :param job: optional constant job ID, which will be printed as the
        first column
    :param str notes: a dict of optional constant-value columns to include in
        all rows (ex. to identify and experiment or parameters)
    :param context: a LEAP context object, used to retrieve the current generation
        from the EA state (i.e. from `context['leap']['generation']`)

    In this example, we'll set up two three inputs for the probe: an output stream,
    the generation number, and a population.

    We use a `StringIO` stream to print the results here, but in practice you
    often want to use `sys.stdout` (the default) or a file object:

    >>> import io
    >>> stream = io.StringIO()

    The probe also relies on LEAP's algorithm `context` to determine the generation number:

    >>> from leap_ec.global_vars import context
    >>> context['leap']['generation'] = 100

    Here's how we'd compute fitness statistics for a test population.  The population
    is unmodified:

    >>> from leap_ec.data import test_population
    >>> probe = FitnessStatsCSVProbe(stream=stream, job=15, notes={'description': 'just a test'})
    >>> probe(test_population) == test_population
    True

    and the output has the following columns:
    >>> print(stream.getvalue())
    job, description, step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness
    15, just a test, 100, 4, 2.5, 1.11803..., 1, 4
    <BLANKLINE>

    To add custom columns, use the `extra_metrics` dict.  For example, here's a function
    that computes the median fitness value of a population:

    >>> import numpy as np
    >>> median = lambda p: np.median([ ind.fitness for ind in p ])

    We can include it in the fitness stats report like so:

    >>> stream = io.StringIO()
    >>> extras_probe = FitnessStatsCSVProbe(stream=stream, job="15", extra_metrics={'median_fitness': median})
    >>> extras_probe(test_population) == test_population
    True

    >>> print(stream.getvalue())
    job, step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness, median_fitness
    15, 100, 4, 2.5, 1.11803..., 1, 4, 2.5
    <BLANKLINE>

    """
    comment_character = '#'

    time_col='step'
    default_metric_cols=('bsf', 'mean_fitness', 'std_fitness', 'min_fitness', 'max_fitness')

    def __init__(self, stream=sys.stdout,
                 header=True,
                 extra_metrics=None,
                 comment=None,
                 job: str = None,
                 notes: Dict = None,
                 modulo: int = 1,
                 context: Dict = context):
        assert (stream is not None)
        assert (hasattr(stream, 'write'))
        assert (context is not None)

        self.stream = stream
        self.context = context
        self.bsf_ind = None
        self.modulo = modulo
        self.notes = notes if notes else {}
        self.extra_metrics = extra_metrics if extra_metrics else {}
        self.job = job
        self.comment = comment

        if header:
            self.write_comment(stream)
            self.write_header(stream)

    def write_header(self, stream):
        job_header = 'job, ' if self.job is not None else ''
        note_extras = '' if not self.notes else ', '.join(self.notes.keys()) + ', '
        extras = '' if not self.extra_metrics else ', ' + ', '.join(
            self.extra_metrics.keys())
        stream.write(
            job_header + note_extras + 'step, bsf, mean_fitness, std_fitness, min_fitness, max_fitness'
            + extras + '\n')

    def write_comment(self, stream):
        if self.comment:
            commented_lines = []
            for line in self.comment.split('\n'):
                commented_lines.append(f"{self.comment_character} {line}")
            stream.write('\n'.join(commented_lines) + '\n')

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        
        # Always update the best-so-far variable
        best_ind = best_of_gen(population)
        if self.bsf_ind is None or (best_ind > self.bsf_ind):
            self.bsf_ind = best_ind

        # Check if we've reached a measurement interval
        generation = self.context['leap']['generation']
        if generation % self.modulo != 0:
            # If not, don't write fitness info
            return population
        else:
            # Do write fitness info
            if self.job is not None:
                self.stream.write(str(self.job) + ', ')
            for _, v in self.notes.items():
                self.stream.write(str(v) + ', ')

            self.stream.write(str(generation) + ', ')

            self.stream.write(str(self.bsf_ind.fitness) + ', ')

            fitnesses = [x.fitness for x in population]
            self.stream.write(str(np.mean(fitnesses)) + ', ')
            self.stream.write(str(np.std(fitnesses)) + ', ')
            self.stream.write(str(np.min(fitnesses)) + ', ')
            self.stream.write(str(np.max(fitnesses)))
            for _, f in self.extra_metrics.items():
                self.stream.write(', ' + str(f(population)))
            self.stream.write('\n')
            return population


##############################
# Class AttributesCSVProbe
##############################
class AttributesCSVProbe(op.Operator):
    """
    An operator that records the specified attributes for all the individuals
    (or just the best individual) in `population` in CSV-format to the
    specified stream and/or to a DataFrame.

    :param attributes: list of attribute names to record, as found in the
        individuals' `attributes` field
    :param stream: a file object to write the CSV rows to (defaults to sys.stdout).
        Can be `None` if you only want a DataFrame
    :param bool do_dataframe: if True, data will be collected in memory as a
        Pandas DataFrame, which can be retrieved by calling the `dataframe` property
        after (or during) the algorithm run. Defaults to False, since this can
        consume a lot of memory for long-running algorithms.
    :param bool best_only: if True, attributes will only be recorded
        for the best-fitness individual; otherwise a row is recorded for every
        individual in the population
    :param bool header: if True (the default), a CSV header is printed as the
        first row with the column names
    :param bool do_fitness: if True, the individuals' fitness is
        included as one of the columns
    :param bool do_genomes: if True, the individuals' genome is
        included as one of the columns
    :param str notes: a dict of optional constant-value columns to include in
        all rows (ex. to identify and experiment or parameters)
    :param extra_metrics: a dict of `'column_name': function` pairs, to compute
        optional extra columns.  The functions take a the population as input
        as a list of individuals, and their return value is printed in the column.
    :param int job: a job ID that will be included as a constant-value column in
        all rows (ex. typically an integer, indicating the ith run out of many)
    :param context: the algorithm context we use to read the current generation
        from (so we can write it to a column)

    Individuals contain some build-in attributes (namely fitness, genome),
    and also a `dict` of additional custom attributes called, well,
    `attributes`.  This class allows you to log all of the above.

    Most often, you will want to record only the best individual in the
    population at each step, and you'll just want to know its fitness and
    genome.  You can do this with this class's boolean flags.  For example,
    here's how you'd record the best individual's fitness and genome to a
    dataframe:

    >>> from leap_ec.global_vars import context
    >>> from leap_ec.data import test_population
    >>> probe = AttributesCSVProbe(do_dataframe=True, best_only=True,
    ...                            do_fitness=True, do_genome=True)
    >>> context['leap']['generation'] = 100
    >>> probe(test_population) == test_population
    True

    You can retrieve the result programatically from the `dataframe` property:

    >>> probe.dataframe
       step  fitness           genome
    0   100        4  [0, 1, 1, 1, 1]

    By default, the results are also written to `sys.stdout`.  You can pass
    any file object you like into the `stream` parameter.

    Another common use of this task is to record custom attributes that are
    stored on individuals in certain kinds of experiments.  Here's how you
    would record the values of `ind.foo` and `ind.bar` for every individual
    in the population.  We write to a stream object this time to demonstrate
    how to use the probe without a dataframe:

    >>> import io
    >>> stream = io.StringIO()
    >>> probe = AttributesCSVProbe(attributes=['foo', 'bar'], stream=stream)
    >>> context['leap']['generation'] = 100
    >>> r = probe(test_population)
    >>> print(stream.getvalue())
    step,foo,bar
    100,GREEN,Colorless
    100,15,green
    100,BLUE,ideas
    100,72.81,sleep
    <BLANKLINE>
    """

    def __init__(self, attributes=(), stream=sys.stdout, do_dataframe=False,
                 best_only=False, header=True, do_fitness=False,
                 do_genome=False,
                 notes=None, extra_metrics=None, job=None,
                 context=context):
        assert ((stream is None) or hasattr(stream, 'write'))
        self.context = context
        self.stream = stream
        self.attributes = attributes
        self.best_only = best_only

        self.do_fitness = do_fitness
        self.do_genome = do_genome
        self.notes = notes if notes else {}
        self.extra_metrics = extra_metrics if extra_metrics else {}
        self.job = job
        self.do_dataframe = do_dataframe

        if (not do_dataframe) and stream is None:
            raise ValueError(
                "Both 'stream'=None and 'do_dataframe'=False, but at least one must be enabled.")

        fieldnames = []
        if job is not None:
            fieldnames.append('job')
        for name in self.notes.keys():
            fieldnames.append(name)
        fieldnames.append('step')
        fieldnames.extend(list(attributes))
        if do_fitness:
            fieldnames.append('fitness')
        if do_genome:
            fieldnames.append('genome')
        for name in self.extra_metrics.keys():
            fieldnames.append(name)

        self.fieldnames = fieldnames

        if self.do_dataframe:
            # We'll store rows of data as dicts in this list as we collect them
            self.data = []

        self.writer = None
        if stream is not None:
            # We'll write rows of data to this stream as we collect them
            self.writer = csv.DictWriter(
                stream, fieldnames=fieldnames, lineterminator='\n')
            if header:
                self.writer.writeheader()

    @property
    def dataframe(self):
        """Property for retrieving a Pandas DataFrame representation of the
        collected data. """
        if not self.do_dataframe:
            raise ValueError(
                'Tried to retrieve a dataframe of results, but this ' +
                f'{type(AttributesCSVProbe).__name__} was initialized with dataframe=False.')
        # We create the DataFrame on demand because it's inefficient to append to a DataFrame,
        # so we only want to create it after we are done generating data.
        return pd.DataFrame(self.data, columns=self.fieldnames)       
        
    def __call__(self, population):
        """When called (i.e. as part of an operator pipeline), take a
        population of individuals and collect data from it. """
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])

        individuals = [max(population)] if self.best_only else population
        
        for ind in individuals:
            row = self.get_row_dict(ind)
            if self.writer is not None:
                self.writer.writerow(row)

            if self.do_dataframe:
                self.data.append(row)

        return population

    def get_row_dict(self, ind):
        """Compute a full row of data from a given individual."""
        row = {'step': self.context['leap']['generation']}

        for attr in self.attributes:
            if attr not in ind.__dict__:
                raise ValueError(
                    'Attribute "{0}" not found in individual "{1}".'.format(
                        attr, ind.__repr__()))
            row[attr] = ind.__dict__[attr]

        if self.job is not None:
            row['job'] = self.job
        for k, v in self.notes.items():
            row[k] = v
        if self.do_fitness:
            row['fitness'] = ind.fitness
        if self.do_genome:
            row['genome'] = ind.genome
        for k, f in self.extra_metrics.items():
            row[k] = f(row)

        return row


##############################
# Class PopulationMetricsPlotProbe
##############################
class PopulationMetricsPlotProbe:

    def __init__(self, ax=None,
                 metrics=None,
                 xlim=None, ylim=None, modulo=1,
                 title='Population Metrics',
                 x_axis_value=None, context=context):
        """

        FIXME s/modulo/step/

        :param ax: matplotlib ax
        :param metrics: ???
        :param xlim: x axis bounds
        :param ylim: y axis bounds
        :param modulo: update interval
        :param title: for the plot
        :param x_axis_value: ???
        :param context: for accessing current generation
        """

        if ax is None:
            _, ax = plt.subplots()

        self.metrics = metrics
        self.modulo = modulo
        # x-axis defaults to generation
        if x_axis_value is None:
            x_axis_value = lambda: context['leap']['generation']
        self.x_axis_value = x_axis_value
        self.context = context

        # Set axis limits, and some variables we'll use for real-time scaling
        self.xlim = xlim
        self.ylim = ylim
        self.ax = ax
        plt.title(title)

        self.reset()

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.append(self.x, self.x_axis_value())

            for i, m in enumerate(self.metrics):
                self.y[i] = np.append(self.y[i], m(population))
                line = self.ax.lines[i]
                line.set_xdata(self.x)
                line.set_ydata(self.y[i])

            self._rescale_ax()
            self.ax.figure.canvas.draw()
            # With previous versions of matplotlib and/or jupyter, these lines were needed to
            #   trigger the plot to update in real time.  But as of the time of this comment,
            #   they are either not needed or caused problems.
            #plt.pause(0.000001)
            #plt.ion()
        return population

    def reset(self):
        # Create an empty line for each metric
        self.x = np.array([])
        self.y = [ np.array([]) for _ in range(len(self.metrics)) ]
        for _ in range(len(self.metrics)):
            self.ax.plot([], [])

    def _rescale_ax(self):
        l = np.min(self.x)
        r = np.max(self.x)
        if self.xlim is not None:
            l = min(self.xlim[0], l)
            r = max(self.xlim[1], r)

        b = np.min(self.y)
        t = np.max(self.y)
        if self.ylim is not None:
            b = min(self.ylim[0], b)
            t = max(self.ylim[1], t)

        self.ax.relim()
        self.ax.update_datalim(((l, b), (r, t)))
        self.ax.autoscale()


##############################
# Function pairwise_distance_metric()
##############################
def pairwise_squared_distance_metric(population: list):
    """Computes the genetic diversity of a population by considering the
    sum of squared Euclidean distances between individual genomes.

    We compute this in :math:`O(n)` by writing the sum in terms of
    distance from the population centroid :math:`c`:

    .. math::

       \\mathcal{D}(\\text{population}) = \\sum_{i=1}^n \\sum_{j=1}^n \\| x_i - x_j \\|^2 = 2n \\sum_{i=1}^n \\| x_i - c \\|^2
    """
    # Create one big matrix from the population
    genomes = [ ind.genome for ind in population ]
    genomes_matrix = np.stack(genomes)

    centroid = np.mean(genomes_matrix, axis=0)  # Compute c
    distances = genomes_matrix - centroid       # Compute x_i - c for all i
    norms = np.linalg.norm(distances, axis=1)   # Compute \|x_i - c\|
    sq_norms = np.power(norms, 2)               # Compute \|x_i - c\|^2

    return 2*len(population)*np.sum(sq_norms)  # Return 2n\sum_{i=1}^n \|x_i - c\|^2


##############################
# Function sum_of_variances_metric()
##############################
def sum_of_variances_metric(population: list):
    """Computes the genetic diversity of a population by considering the sum of
    the variances of each variable in the genome.

    .. math::

        \\mathcal{D}(\\text{population}) = \\sum_{i=1}^L \\mathbb{E}_{j \\in P}\\left[ x_j[i] - \\mathbb{E}[x_j[i]] \\right]

    This is a so-called "column-wise" metric, in the sense that it considers
    each element of the solution vectors independently.
    """
    # Create one big matrix from the population
    genomes = [ ind.genome for ind in population ]
    genomes_matrix = np.stack(genomes)

    variances = np.std(genomes_matrix, axis=0)**2

    return sum(variances)


##############################
# Function num_fixated_metrics()
##############################
def num_fixated_metric(population: list):
    """Computes the genetic diversity of the population by counting the number
    of variables in the genome that have zero variance.

    This is a so-called "column-wise" metric, in the sense that it considers
    each element of the solution vectors independently.
    """

    # Create one big matrix from the population
    genomes = [ ind.genome for ind in population ]
    genomes_matrix = np.stack(genomes)

    variances = np.std(genomes_matrix, axis=0)**2

    return sum(np.isclose(variances, 0))


##############################
# Class FitnessPlotProbe
##############################
class FitnessPlotProbe(PopulationMetricsPlotProbe):
    """
    Measure and plot a population's fitness trajectory.

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param int modulo: take and plot a measurement every `modulo` steps (
        default 1).
    :param title: title to print on the plot
    :param x_axis_value: optional function to define what value gets plotted
        on the x axis.  Defaults to pulling the 'generation' value out of the
        default `context` object.
    :param context: set a context object to query for the current generation.
        Defaults to the standard `leap_ec.context` object.

    Attach this probe to matplotlib :class:`Axes` and then insert it into an
    EA's operator pipeline.

    >>> import matplotlib.pyplot as plt
    >>> from leap_ec.probe import FitnessPlotProbe
    >>> from leap_ec.representation import Representation

    >>> f = plt.figure()  # Setup a figure to plot to
    >>> plot_probe = FitnessPlotProbe(ylim=(0, 70), ax=plt.gca())


    >>> # Create an algorithm that contains the probe in the operator pipeline
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec import ops
    >>> from leap_ec.real_rep.problems import SpheroidProblem
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> from leap_ec.real_rep.initializers import create_real_vector

    >>> from leap_ec.algorithm import generational_ea

    >>> l = 10
    >>> pop_size = 10
    >>> ea = generational_ea(max_generations=100, pop_size=pop_size,
    ...                      problem=SpheroidProblem(maximize=False),
    ...
    ...                      representation=Representation(
    ...                         individual_cls=Individual,
    ...                         decoder=IdentityDecoder(),
    ...                         initialize=create_real_vector(bounds=[[-5.12, 5.12]] * l)
    ...                      ),
    ...
    ...                      pipeline=[
    ...                         plot_probe,  # Insert the probe into the pipeline like so
    ...                         ops.tournament_selection,
    ...                         ops.clone,
    ...                         mutate_gaussian(std=0.2, expected_num_mutations='isotropic'),
    ...                         ops.evaluate,
    ...                         ops.pool(size=pop_size)
    ...                      ])
    >>> result = list(ea);


    .. plot::

        import matplotlib.pyplot as plt
        from leap_ec.probe import FitnessPlotProbe
        from leap_ec.representation import Representation

        f = plt.figure()  # Setup a figure to plot to
        plot_probe = FitnessPlotProbe(ylim=(0, 70), ax=plt.gca())


        # Create an algorithm that contains the probe in the operator pipeline
        from leap_ec.individual import Individual
        from leap_ec.decoder import IdentityDecoder
        from leap_ec import ops
        from leap_ec.real_rep.problems import SpheroidProblem
        from leap_ec.real_rep.ops import mutate_gaussian
        from leap_ec.real_rep.initializers import create_real_vector

        from leap_ec.algorithm import generational_ea

        l = 10
        pop_size = 10
        ea = generational_ea(max_generations=100, pop_size=pop_size,
                             problem=SpheroidProblem(maximize=False),

                             representation=Representation(
                                individual_cls=Individual,
                                decoder=IdentityDecoder(),
                                initialize=create_real_vector(bounds=[[-5.12, 5.12]] * l)
                             ),

                             pipeline=[
                                 plot_probe,  # Insert the probe into the pipeline like so
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_gaussian(std=0.2, expected_num_mutations='isotropic'),
                                 ops.evaluate,
                                 ops.pool(size=pop_size)
                             ])
        result = list(ea);


    To get a live-updated plot that words like a real-time video of the EA's
    progress, use this probe in conjunction with the `%matplotlib notebook`
    magic for Jupyter Notebook (as opposed to `%matplotlib inline`,
    which only allows static plots).
    """
    def __init__(self, ax=None, xlim=None, ylim=None, modulo=1, title='Best-of-Generation Fitness',  x_axis_value=None, context=context):
        super().__init__(ax=ax,
                 metrics=[ lambda pop: best_of_gen(pop).fitness ],
                 xlim=xlim, ylim=ylim, modulo=modulo, title=title,
                 x_axis_value=x_axis_value, context=context)


##############################
# Class CartesianPhenotypePlotProbe
##############################
class CartesianPhenotypePlotProbe:
    """
    Measure and plot a scatterplot of the populations' location in a 2-D
    phenotype space.

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param ~leap.problem.Problem contours: a problem defining a 2-D fitness
        function (this will be used to draw fitness contours in the background of
        the scatterplot).
    :param float granularity: (Optional) spacing of the grid to sample points
        along while drawing the fitness contours. If none is given, then the
        granularity will default to 1/50th of the range of the function's
        `bounds` attribute.
    :param int modulo: take and plot a measurement every `modulo` steps (
        default 1).
    :param pad: A list of extra gene values, used to fill in the hidden
        dimensions with contants while drawing fitness contours.

    Attach this probe to matplotlib :class:`Axes` and then insert it into an
    EA's operator pipeline to get a live phenotype plot that updates every
    `modulo` steps.

    >>> import matplotlib.pyplot as plt
    >>> from leap_ec.probe import CartesianPhenotypePlotProbe
    >>> from leap_ec.representation import Representation

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.algorithm import generational_ea

    >>> from leap_ec import ops
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.real_rep.problems import CosineFamilyProblem
    >>> from leap_ec.real_rep.initializers import create_real_vector
    >>> from leap_ec.real_rep.ops import mutate_gaussian

    >>> # The fitness landscape
    >>> problem = CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])

    >>> # If no axis is provided, a new figure will be created for the probe to write to
    >>> trajectory_probe = CartesianPhenotypePlotProbe(contours=problem,
    ...                                        xlim=(0, 1), ylim=(0, 1),
    ...                                        granularity=0.025)

    >>> # Create an algorithm that contains the probe in the operator pipeline

    >>> pop_size = 100
    >>> ea = generational_ea(max_generations=20, pop_size=pop_size,
    ...                      problem=problem,
    ...
    ...                      representation=Representation(
    ...                         individual_cls=Individual,
    ...                         initialize=create_real_vector(bounds=[[0.4, 0.6]] * 2),
    ...                         decoder=IdentityDecoder()
    ...                      ),
    ...
    ...                      pipeline=[
    ...                         trajectory_probe,  # Insert the probe into the pipeline like so
    ...                         ops.tournament_selection,
    ...                         ops.clone,
    ...                         mutate_gaussian(std=0.05, expected_num_mutations='isotropic', bounds=(0, 1)),
    ...                         ops.evaluate,
    ...                         ops.pool(size=pop_size)
    ...                      ])
    >>> result = list(ea);

    .. plot::

        import matplotlib.pyplot as plt
        from leap_ec.probe import CartesianPhenotypePlotProbe
        from leap_ec.representation import Representation

        from leap_ec.individual import Individual
        from leap_ec.algorithm import generational_ea

        from leap_ec import ops
        from leap_ec.decoder import IdentityDecoder
        from leap_ec.real_rep.problems import CosineFamilyProblem
        from leap_ec.real_rep.initializers import create_real_vector
        from leap_ec.real_rep.ops import mutate_gaussian

        # The fitness landscape
        problem = CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])

        # If no axis is provided, a new figure will be created for the probe to write to
        trajectory_probe = CartesianPhenotypePlotProbe(contours=problem,
                                                        xlim=(0, 1), ylim=(0, 1),
                                                        granularity=0.025)

        # Create an algorithm that contains the probe in the operator pipeline

        pop_size = 100
        ea = generational_ea(max_generations=20, pop_size=pop_size,
                             problem=problem,

                             representation=Representation(
                                individual_cls=Individual,
                                initialize=create_real_vector(bounds=[[0.4, 0.6]] * 2),
                                decoder=IdentityDecoder()
                             ),

                             pipeline=[
                                 trajectory_probe,  # Insert the probe into the pipeline like so
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_gaussian(std=0.05, expected_num_mutations='isotropic', bounds=(0, 1)),
                                 ops.evaluate,
                                 ops.pool(size=pop_size)
                             ])
        result = list(ea);


    """

    def __init__(self, ax=None, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12),
                 contours=None, granularity=None, title='Cartesian Phenotypes',
                 modulo=1, context=context, pad=()):
        if ax is None:
            _, ax = plt.subplots()
        if contours:
            @np.vectorize
            def v_fun(x, y):
                phenome = np.concatenate((np.hstack((x,y)), pad))
                return contours.evaluate(phenome)

            if granularity is None:
                granularity = (contours.bounds[1] - contours.bounds[0]) / 50.
            x = np.arange(xlim[0], xlim[1], granularity)
            y = np.arange(ylim[0], ylim[1], granularity)
            xx, yy = np.meshgrid(x, y)
            ax.contour(xx, yy, v_fun(xx, yy))

        self.sc = ax.scatter([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.ax = ax
        self.left, self.right = xlim
        self.bottom, self.top = ylim
        self.x = np.array([])
        self.y = np.array([])
        plt.title(title)
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.array([ind.decode()[0] for ind in population])
            self.y = np.array([ind.decode()[1] for ind in population])
            self.sc.set_offsets(np.c_[self.x, self.y])
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population

    def __rescale_ax(self):
        if np.min(self.x) < self.left:
            self.ax.set_xlim(left=np.min(self.x))
        if np.max(self.x) > self.right:
            self.ax.set_xlim(right=np.max(self.x))
        if np.min(self.y) < self.bottom:
            self.ax.set_ylim(bottom=np.min(self.y))
        if np.max(self.y) > self.top:
            self.ax.set_ylim(top=np.max(self.y))


##############################
# Class HistPhenotypePlotProbe
##############################
class HistPhenotypePlotProbe():
    """A visualization probe that uses matplotlib to show a live histogram
    of the population's phenotypes.

    This typically makes the most since for 1-dimensional genotypes.
    """
    def __init__(self, ax=None, title='Histogram of Phenotypes',
                 modulo=1, context=context):
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax

        ax.set_title(title)
        self.title = title
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            phenomes = [ ind.decode() for ind in population ]
            self.ax.cla()
            self.ax.hist(phenomes)
            self.ax.set_title(self.title)
            #self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population


##############################
# HeatMapPhenotypeProbe
##############################
class HeatMapPhenotypeProbe():
    """
    """
    def __init__(self, ax=None, title='HeatMap of Phenotypes',
                 modulo=1, context=context):
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax

        ax.set_title(title)
        self.title = title
        self.modulo = modulo
        self.context = context

        self.map = np.empty((10,0))

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            phenomes = [ ind.decode() for ind in population ]
            hist, _ = np.histogram(phenomes)
            self.map
            self.ax.cla()
            self.ax.hist(phenomes)
            self.ax.set_title(self.title)
            #self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population



##############################
# Class SumPhenotypePlotProbe
##############################
class SumPhenotypePlotProbe:
    """
    Plot the population's location on a fitness landscape that is defined
    over the sum of a vector phenotype's elements.  This is useful for visualizing
    OneMax functions and similar functions that can be understood in
    terms of a graph with "the number of ones" along the x axis.

    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param xlim: Bounds of the horizontal axis.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param ~leap.problem.Problem problem: a problem that will be used to draw
        a fitness curve.
    :param float granularity: (Optional) spacing of the grid to sample points
        along while drawing the fitness contours. If none is given, then the
        granularity will default to 1.0.
    :param int modulo: take and plot a measurement every `modulo` steps (
        default 1).

    Attach this probe to matplotlib :class:`Axes` and then insert it into an
    EA's operator pipeline to get a live phenotype plot that updates every
    `modulo` steps.

    >>> import matplotlib.pyplot as plt
    >>> from leap_ec.probe import SumPhenotypePlotProbe
    >>> from leap_ec.representation import Representation

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.algorithm import generational_ea

    >>> from leap_ec import ops
    >>> from leap_ec.binary_rep.problems import DeceptiveTrap
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> from leap_ec.binary_rep.ops import mutate_bitflip

    >>> # The fitness landscape
    >>> problem = DeceptiveTrap()

    >>> # If no axis is provided, a new figure will be created for the probe to write to
    >>> dimensions = 20
    >>> trajectory_probe = SumPhenotypePlotProbe(problem=problem,
    ...                                        xlim=(0, dimensions), ylim=(0, dimensions))

    >>> # Create an algorithm that contains the probe in the operator pipeline

    >>> pop_size = 100
    >>> ea = generational_ea(max_generations=20, pop_size=pop_size,
    ...                      problem=problem,
    ...
    ...                      representation=Representation(
    ...                         individual_cls=Individual,
    ...                         initialize=create_binary_sequence(length=dimensions)
    ...                      ),
    ...
    ...                      pipeline=[
    ...                         trajectory_probe,  # Insert the probe into the pipeline like so
    ...                         ops.tournament_selection,
    ...                         ops.clone,
    ...                         mutate_bitflip(expected_num_mutations=1),
    ...                         ops.evaluate,
    ...                         ops.pool(size=pop_size)
    ...                      ])
    >>> result = list(ea);

    .. plot::

        import matplotlib.pyplot as plt
        from leap_ec.probe import SumPhenotypePlotProbe
        from leap_ec.representation import Representation

        from leap_ec.individual import Individual
        from leap_ec.algorithm import generational_ea

        from leap_ec import ops
        from leap_ec.binary_rep.problems import DeceptiveTrap
        from leap_ec.binary_rep.initializers import create_binary_sequence
        from leap_ec.binary_rep.ops import mutate_bitflip

        # The fitness landscape
        problem = DeceptiveTrap()

        # If no axis is provided, a new figure will be created for the probe to write to
        trajectory_probe = SumPhenotypePlotProbe(problem=problem,
                                                    xlim=(0, 1), ylim=(0, 1))

        # Create an algorithm that contains the probe in the operator pipeline

        pop_size = 100
        dimensions = 20
        ea = generational_ea(max_generations=20, pop_size=pop_size,
                             problem=problem,

                             representation=Representation(
                                individual_cls=Individual,
                                initialize=create_binary_sequence(length=dimensions)
                             ),

                             pipeline=[
                                 trajectory_probe,  # Insert the probe into the pipeline like so
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_bitflip(expected_num_mutations=1),
                                 ops.evaluate,
                                 ops.pool(size=pop_size)
                             ])
        result = list(ea);


    """

    def __init__(self, ax=None, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12),
                 problem=None, granularity=1, title='Sum Phenotypes',
                 modulo=1, context=context):
        if ax is None:
            _, ax = plt.subplots()

        if problem:
            # If a problem is provided, plot the fitness function

            # First we need to generate a series of genomes to
            # feed into the fitnes function
            max_number_of_ones = int(xlim[1])
            def bitstring_with_ones(num_ones):
                """Generate a bitstring with n initial ones, and
                otherwise filled with zeroes."""
                assert(num_ones <= max_number_of_ones)
                return np.array([1]*num_ones + [0]*(max_number_of_ones - num_ones))
            x = np.arange(int(xlim[0]), max_number_of_ones + 1, int(granularity))

            # Now plot the function over them
            y = np.array([ problem.evaluate(bitstring_with_ones(i)) for i in x ])
            ax.plot(x, y, color='black', linewidth=3)

        self.sc = ax.scatter([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.ax = ax
        self.left, self.right = xlim
        self.bottom, self.top = ylim
        self.x = np.array([])
        self.y = np.array([])
        plt.title(title)
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            self.x = np.array([np.sum(ind.decode()) for ind in population])
            self.y = np.array([ind.fitness for ind in population])
            self.sc.set_offsets(np.c_[self.x, self.y])
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population

    def __rescale_ax(self):
        if np.min(self.x) < self.left:
            self.ax.set_xlim(left=np.min(self.x))
        if np.max(self.x) > self.right:
            self.ax.set_xlim(right=np.max(self.x))
        if np.min(self.y) < self.bottom:
            self.ax.set_ylim(bottom=np.min(self.y))
        if np.max(self.y) > self.top:
            self.ax.set_ylim(top=np.max(self.y))


##############################
# best_of_gen function
##############################
def best_of_gen(population):
    """
    Syntactic sugar to select the best individual in a population.

    :param population: a list of individuals
    :param context: optional `dict` of auxiliary state (ignored)

    >>> from leap_ec.data import test_population
    >>> print(best_of_gen(test_population))
    Individual<...> with fitness 4
    """
    assert (len(population) > 0)
    return max(population)
