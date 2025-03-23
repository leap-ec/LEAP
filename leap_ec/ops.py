"""Fundamental evolutionary operators.

This module provides many of the most important functions that we string
together to create EAs out of operator pipelines. You'll find many
traditional selection and reproduction strategies here, as well as components
for classic algorithms like island models and cooperative coevolution.

Representation-specific operators tend to reside within their own subpackages,
rather than here.  See for example :py:mod:`leap_ec.real_rep.ops` and
:py:mod:`leap_ec.binary_rep.ops`.
"""
import abc
import collections
from copy import copy
import csv
import itertools
from functools import wraps
import logging
import random
from statistics import mean
from typing import Iterator, List, Union, Tuple

import numpy as np
import toolz
from leap_ec.util import wrap_curry

from leap_ec import leap_logger_name
from leap_ec.global_vars import context


# Set up a logger using LEAP's global logger name
logger = logging.getLogger(leap_logger_name)


##############################
# Class Operator
##############################
class Operator(abc.ABC):
    """Abstract base class that documents the interface for operators in a
    LEAP pipeline.

    LEAP treats operators as functions of two arguments: the population,
    and a "context" `dict` that may be used in some algorithms to maintain
    some global state or parameters independent of the population.

    TODO The above description is outdated. --Siggy
    TODO Also this is for a *population* based operator.  We also have operators
    *for individuals*

    You can inherit from this class to define operators as classes.  Classes
    support operators that take extra arguments at construction time (such as
    a mutation rate) and maintain some internal private state, and they allow
    certain special patterns (such as multi-function operators).

    But inheriting from this class is optional.  LEAP can treat any
    `callable` object that takes two parameters as an operator.  You may
    define your custom operators as closures (which also allow for
    construction-time arguments and internal state), as simple functions (
    when no additional arguments are necessary), or as curried functions (
    i.e. with the help of `toolz.curry(...)`.

    """

    @abc.abstractmethod
    def __call__(self, pop_generator):
        """
        The basic interface for a pipeline operator in LEAP.

        :param pop_generator: a list or generator of individuals to be operated upon
        """
        pass


##############################
# Decorators for type checking
##############################
def iteriter_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives an iterator as its first argument, and that it
    returns an iterator.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs a list to an operator
    that expects an iterator, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(next_individual: Iterator, *args, **kwargs) -> Iterator:
        if not isinstance(next_individual, collections.abc.Iterator):
            if isinstance(next_individual, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(next_individual)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population iterator."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(next_individual)} as input, "
                    f"but expected an iterator.")

        result = f(next_individual, *args, **kwargs)

        if not isinstance(result, collections.abc.Iterator):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected an iterator.")

        return result

    return typecheck_f


def listlist_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives a list as its first argument, and that it returns
    a list.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs an iterator to an
    operator that expects a list, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(population: List, *args, **kwargs) -> List:
        if not isinstance(population, list):
            if isinstance(population, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(population)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population list."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(population)} as input, but "
                    f"expected a list.")

        result = f(population, *args, **kwargs)

        if not isinstance(result, list):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected a list.")

        return result

    return typecheck_f


def listiter_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives a list as its first argument, and that it returns
    an iterator.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs an iterator to an
    operator that expects a list, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(population: List, *args, **kwargs) -> Iterator:
        if not isinstance(population, list):
            if isinstance(population, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(population)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population list."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(population)} as input, but "
                    f"expected a list.")

        result = f(population, *args, **kwargs)

        if not isinstance(result, collections.abc.Iterator):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, but "
                f"expected an iterator.")

        return result

    return typecheck_f


def iterlist_op(f):
    """This decorator wraps a function with runtime type checking to ensure
    that it always receives an iterator as its first argument, and that it
    returns a list.

    We use this to make debugging operator pipelines easier in EAs: if you
    accidentally hook up, say an operator that outputs a list to an operator
    that expects an iterator, we'll throw an exception that pinpoints the
    issue.

    :param f function: the function to wrap
    """

    @wraps(f)
    def typecheck_f(next_individual: Iterator, *args, **kwargs) -> List:
        if not isinstance(next_individual, collections.abc.Iterator):
            if isinstance(next_individual, toolz.functoolz.curry):
                raise ValueError(
                    f"While executing operator {f}, an incomplete curry object was received ({type(next_individual)}).\n" + \
                    "This usually means that you forgot to specify a required argument for an upstream operator, " + \
                    "so a partly-curried function got passed down the pipeline instead of a population iterator."
                )
            else:
                raise ValueError(
                    f"Operator {f} received a {type(next_individual)} as input, "
                    f"but expected an iterator.")

        result = f(next_individual, *args, **kwargs)

        if not isinstance(result, list):
            raise ValueError(
                f"Operator {f} produced a {type(result)} as output, "
                f"but expected a list.")

        return result

    return typecheck_f


##############################
# evaluate operator
##############################
@wrap_curry
@iteriter_op
def evaluate(next_individual: Iterator) -> Iterator:
    """ Evaluate and returns the next individual in the pipeline

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> import numpy as np

    We need to specify the decoder and problem so that evaluation is possible.

    >>> genome = np.array([1, 1])
    >>> ind = Individual(genome, decoder=IdentityDecoder(), problem=MaxOnes())

    >>> evaluated_ind = next(evaluate(iter([ind])))

    :param next_individual: iterator pointing to next individual to be evaluated

    :param kwargs: contains optional context state to pass down the pipeline
       in context dictionaries

    :return: the evaluated individual
    """
    while True:
        individual = next(next_individual)
        individual.evaluate()

        yield individual


##############################
# evaluate operator
##############################
@wrap_curry
@listlist_op
def grouped_evaluate(population: list, max_individuals_per_chunk: int = None) -> list:
    """Evaluate the population by sending groups of multiple individuals to
    a fitness function so they can be evaluated simultaneously.

    This is useful, for example, as a way to evaluate individuals in parallel
    on a GPU."""
    if max_individuals_per_chunk is None:
        max_individuals_per_chunk = len(population)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    problem = population[0].problem
    assert(all([ind.problem == problem for ind in population])), f"Two or more individuals in the population have different problem references; cannot perform grouped evaluation!"

    fitnesses = []
    for chunk in chunks(population, max_individuals_per_chunk):
        # XXX Always passing individuals along to the problem.
        #     Does this create problems with dask, even when we aren't using individuals?
        fit = problem.evaluate_multiple([c.phenome for c in chunk], individuals=chunk)
        fitnesses.extend(fit)

    for fit, ind in zip(fitnesses, population):
        ind.fitness = fit

    return population


##############################
# const_evaluate operator
##############################
@wrap_curry
@listlist_op
def const_evaluate(population: List, value) -> List:
    """An evaluator that assigns a constant fitness to every individual.

    This ignores the `Problem` associated with each individual for the
    purpose of assigning a constant fitness.

    This is useful for algorithms that need to assign an arbitrary initial
    fitness value before using their normal evaluation method.  Some forms of
    cooperative coevolution are an example.
    """
    for ind in population:
        ind.fitness = value

    return population


##############################
# clone operator
##############################
@wrap_curry
@iteriter_op
def clone(next_individual: Iterator) -> Iterator:
    """ clones and returns the next individual in the pipeline

    The clone's fitness is set to None, its parents are set to the individual
    from which it was cloned (i.e., the parent), and it is assigned its own
    UUID.

    >>> from leap_ec.individual import Individual
    >>> import numpy as np

    Create a common decoder and problem for individuals.

    >>> genome = np.array([1, 1])
    >>> original = Individual(genome)

    >>> cloned_generator = clone(iter([original]))

    :param next_individual: iterator for next individual to be cloned
    :return: copy of next_individual
    """

    while True:
        individual = next(next_individual)

        yield individual.clone()


##############################
# Crossover base class
##############################
class Crossover(Operator):

    def __init__(self, persist_children, p_xover):
        self.persist_children = persist_children
        self.second_child = None
        self.p_xover = p_xover

    @abc.abstractmethod
    def recombine(self, parent_a, parent_b):
        """
        Perform recombination between two parents to produce two new individuals.
        """
        raise NotImplementedError

    def __call__(self, next_individual):
        """ Performs successive in-pipeline recombinations.

        :param next_individual: where we get the next individual
        :return: two recombined individuals (with probability p_xover), or two
            unmodified individuals (with probability 1 - p_xover)
        """
        # There has to be an inner function so we can properly use @iteriter_op
        @iteriter_op
        def _call(next_individual):
            if self.persist_children and self.second_child is not None:
                # Return a child left from another run
                # Swap with None has to happen before the yield to be certain its executed
                ret_child, self.second_child = self.second_child, None
                yield ret_child

            while True:
                parent_a = next(next_individual)
                parent_b = next(next_individual)

                if np.random.uniform() > self.p_xover:
                    first_child, self.second_child = parent_a, parent_b
                else:
                    first_child, self.second_child = self.recombine(parent_a, parent_b)
                    
                    first_child.parents |= self.second_child.parents
                    self.second_child.parents |= first_child.parents

                    first_child.fitness = self.second_child.fitness = None

                # Generators only execute code if necessary, so children will only be generated
                # if the first child needs to be yielded. That's why it doesn't need to be stored
                # in the class as well.
                yield first_child

                # Remove the second child from the class and yield it if the generator does
                # get requested for it
                ret_child, self.second_child = self.second_child, None
                yield ret_child

        return _call(next_individual)


##############################
# Uniform Crossover class
##############################
class UniformCrossover(Crossover):
    """Parameterized uniform crossover iterates through two parents' genomes
    and swaps each of their genes with the given probability.

    In a classic paper, De Jong and Spears showed that this operator works
    particularly well when the swap probability `p_swap` is set to about 0.2.  LEAP
    thus uses this value as its default.

        De Jong, Kenneth A., and W. Spears. "On the virtues of parameterized uniform crossover."
        *Proceedings of the 4th international conference on genetic algorithms.* Morgan Kaufmann Publishers, 1991.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import UniformCrossover, naive_cyclic_selection
    >>> import numpy as np

    >>> genome1 = np.array([0, 0])
    >>> genome2 = np.array([1, 1])
    >>> first = Individual(genome1)
    >>> second = Individual(genome2)
    >>> pop = [first, second]
    >>> select = naive_cyclic_selection(pop)
    >>> op = UniformCrossover()
    >>> result = op(select)
    >>> new_first = next(result)
    >>> new_second = next(result)

    The probability can be tuned via the `p_swap` parameter:
    >>> op = UniformCrossover(p_swap=0.1)
    >>> result = op(select)

    If `persist_children` is True and there is a child that was made by crossover but isn't
    used in the first call, it will be yielded in a future call.

    >>> op = UniformCrossover(p_xover=0.0, persist_children=True)
    >>>
    >>> next(op(select)) is first  # Create an iterator loop with op(select) and consume 1 individual
    True
    >>> next(op(select)) is second # Create a different iterator loop with op(select)
    True

    With `persist_children` set to False, the second child will not be yielded if the iterator
    is consumed an odd number of times. Instead, on the next call the loop is started anew.

    >>> op = UniformCrossover(p_xover=0.0, persist_children=False)
    >>>
    >>> next(op(select)) is first  # Create an iterator loop with op(select) and consume 1 individual
    True
    >>> next(op(select)) is second # Create a different iterator loop with op(select)
    False

    :param p_swap: how likely are we to swap each pair of genes when crossover
        is performed
    :param float p_xover: the probability that crossover is performed in the
        first place
    :param bool persist_children: whether unyielded children should persist between calls.
        This is useful for `leap_ec.distrib.asynchronous.steady_state`, where the pipeline
        may only produce one individual at a time.
    :return: a pipeline operator that returns two recombined individuals (with probability
        p_xover), or two unmodified individuals (with probability 1 - p_xover)
    """

    def __init__(self, p_swap: float=0.2, p_xover: float=1.0, persist_children=False):
        super().__init__(p_xover=p_xover, persist_children=persist_children)
        self.p_swap = p_swap

    def recombine(self, parent_a, parent_b):
        """
        Perform recombination between two parents to produce two new individuals.
        """
        assert(isinstance(parent_a.genome, np.ndarray))
        assert(isinstance(parent_b.genome, np.ndarray))

        # generate which indices we should swap
        min_length = min(parent_a.genome.shape[0], parent_b.genome.shape[0])
        indices_to_swap = random_bernoulli_vector(min_length, self.p_swap)

        # perform swap
        tmp = parent_a.genome[indices_to_swap]
        parent_a.genome[indices_to_swap] = parent_b.genome[indices_to_swap]
        parent_b.genome[indices_to_swap] = tmp

        return parent_a, parent_b


##############################
# N-Ary Crossover class
##############################
class NAryCrossover(Crossover):
    """ Do crossover between individuals between N crossover points.

    1 < n < genome length - 1

    We also assume that the passed in individuals are *clones* of parents.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import NAryCrossover
    >>> import numpy as np

    >>> genome1 = np.array([0, 0])
    >>> genome2 = np.array([1, 1])
    >>> first = Individual(genome1)
    >>> second = Individual(genome2)
    >>> pop = [first, second]
    >>> select = naive_cyclic_selection(pop)

    >>> op = NAryCrossover()
    >>> result = op(select)

    >>> new_first = next(result)
    >>> new_second = next(result)


    If `persist_children` is True and there is a child that was made by crossover but isn't
    used in the first call, it will be yielded in a future call.

    >>> op = NAryCrossover(p_xover=0.0, persist_children=True)
    >>>
    >>> next(op(select)) is first  # Create an iterator loop with op(select) and consume 1 individual
    True
    >>> next(op(select)) is second # Create a different iterator loop with op(select)
    True

    With `persist_children` set to False, the second child will not be yielded if the iterator
    is consumed an odd number of times. Instead, on the next call the loop is started anew.

    >>> op = NAryCrossover(p_xover=0.0, persist_children=False)
    >>>
    >>> next(op(select)) is first  # Create an iterator loop with op(select) and consume 1 individual
    True
    >>> next(op(select)) is second # Create a different iterator loop with op(select)
    False

    :param num_points: how many crossing points do we use?  Defaults to 2, since
        2-point crossover has been shown to be the least disruptive choice for
        this value.
    :param p: the probability that crossover is performed.
    :param bool persist_children: whether unyielded children should persist between calls.
        This is useful for `leap_ec.distrib.asynchronous.steady_state`, where the pipeline
        may only produce one individual at a time.
    :return: a pipeline operator that returns two recombined individuals (with probability
        p), or two unmodified individuals (with probability 1 - p)
    """

    def __init__(self, num_points=2, p_xover=1.0, persist_children=False):
        super().__init__(p_xover=p_xover, persist_children=persist_children)
        self.num_points = num_points

    def _pick_crossover_points(self, genome_size):
        """
        Randomly choose (without replacement) crossover points.
        """
        # See De Jong, EC, pg 145
        pp = np.arange(genome_size, dtype=int)

        xpts = np.random.choice(pp, size=(self.num_points,), replace=False)
        xpts.sort()
        xpts = [0] + list(xpts) + [genome_size]  # Add start and end

        return xpts

    def recombine(self, parent_a, parent_b):
        if len(parent_a.genome) < self.num_points or \
                len(parent_b.genome) < self.num_points:
            raise RuntimeError(
                'Invalid number of crossover points for n_ary_crossover')

        children = [parent_a, parent_b]
        # store each section of the genome to concatenate later
        genome1_sections = []
        genome2_sections = []
        # Used to toggle which sub-test_sequence is copied between offspring
        src1, src2 = 0, 1

        # Pick crossover points
        xpts = self._pick_crossover_points(len(parent_a.genome))

        for start, stop in toolz.itertoolz.sliding_window(2, xpts):
            genome1_sections.append(children[src1].genome[start:stop])
            genome2_sections.append(children[src2].genome[start:stop])

            # Now swap crossover direction
            src1, src2 = src2, src1

        # allows for crossover in both simple representations
        # and segmented representations, respectively
        if isinstance(parent_a.genome, np.ndarray):
            parent_a.genome = np.concatenate(genome1_sections)
            parent_b.genome = np.concatenate(genome2_sections)
        else:
            parent_a.genome = list(
                itertools.chain.from_iterable(genome1_sections))
            parent_b.genome = list(
                itertools.chain.from_iterable(genome2_sections))

        return parent_a, parent_b


##############################
# Function proportional_selection
##############################
@wrap_curry
@listiter_op
def proportional_selection(population: List, offset=0, exponent: int = 1,
                           key=lambda x: x.fitness) -> Iterator:
    """ Returns an individual from a population in direct proportion to their
        fitness or another given metric.

        To deal with negative fitness values use `offset='pop-min'` or set a
        custom offset. A `ValueError` is thrown if the result of adding
        `offset` to a fitness value results in a negative number. The value
        of an individual is calculated as follows

        `value = (fitness + offset)^exponent`

        :param population: the population to select from.
            Should be a list, not an iterator.
        :param offset: the offset from zero. If negative fitness values are
            possible and the minimum is unknown use `offest='pop-min'` for
            an adaptive offset. Defaults to 0.
        :param int exponent: the power to which fitness values are raised to.
            This can be tuned to increase or decrease selection pressure by
            creating larger or smaller differences between fitness values in
            the population. Defaults to 1.
        :param key: a function that computes the metric used to compare
            individuals. Defaults to fitness.
        :return: a random individual based on the proportion of the given
            metric in the population.

        >>> from leap_ec import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import proportional_selection
        >>> import numpy as np

        >>> genome1 = np.array([0, 0, 0])
        >>> genome2 = np.array([0, 0, 1])
        >>> pop = [Individual(genome1, problem=MaxOnes()),
        ...        Individual(genome2, problem=MaxOnes())]
        >>> pop = Individual.evaluate_population(pop)
        >>> selected = proportional_selection(pop)
    """
    # scale and shift to account for possible negative values
    values = compute_population_values(population, offset=offset,
                                       exponent=exponent, key=key)
    assert(values.shape[0] == len(population))

    # throw error on negative values since the algorithm does not
    # work otherwise
    if (values < 0.0).any():
        raise ValueError('negative value found after applying offset.')

    population_total = np.sum(values)
    proportions = values / population_total

    while True:
        choices = random.choices(population, weights=proportions)
        yield choices[0]


##############################
# Function sus_selection
##############################
@wrap_curry
@listiter_op
def sus_selection(population: List, n=None, shuffle: bool = True,
                  offset=0, exponent: int = 1,
                  key=lambda x: x.fitness) -> Iterator:
    """ Returns an individual from a population in proportion to their
        fitness or another given metric using the stochastic universal
        sampling algorithm.

        To deal with negative fitness values use `offset='pop-min'` or set a
        custom offset. A `ValueError` is thrown if the result of adding
        `offset` to a fitness value results in a negative number. The value
        of an individual is calculated as follows

        `value = (fitness + offset)^exponent`

        :param population: the population to select from.
            Should be a list, not an iterator.
        :param n: the number of evenly spaced points to use in the algorithm.
            Default is None which uses `len(population)`.
        :param bool shuffle: if True, `n` points are resampled after one full
            pass over them. If False, selection repeats over the same `n`
            points. Defaults to True.
        :param offset: the offset from zero. If negative fitness values are
            possible and the minimum is unknown use `offset='pop-min'` for
            an adaptive offset. Defaults to 0.
        :param int exponent: the power to which fitness values are raised to.
            This can be tuned to increase or decrease selection pressure by
            creating larger or smaller differences between fitness values in
            the population. Defaults to 1.
        :param key: a function that computes the metric used to compare
            individuals. Defaults to fitness.
        :return: a random individual based on the proportion of the given
            metric in the population.

        >>> from leap_ec import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import sus_selection
        >>> import numpy as np

        >>> genome1 = np.array([0, 0, 0])
        >>> genome2 = np.array([1, 1, 1])
        >>> pop = [Individual(genome1, problem=MaxOnes()),
        ...        Individual(genome2, problem=MaxOnes())]
        >>> pop = Individual.evaluate_population(pop)
        >>> selected = sus_selection(pop)
    """
    # determine number of points to sample if not specified
    if n is None:
        n = len(population)

    # check for non-positive number of points
    if n <= 0:
        raise ValueError(f'cannot sample {n} number of points')

    # scale and shift to account for possible negative values
    values = compute_population_values(population, offset=offset,
                                       exponent=exponent, key=key)
    assert(values.shape[0] == len(population))

    # throw error on negative values since the algorithm does not
    # work otherwise
    if (values < 0.0).any():
        raise ValueError('negative value found after applying offset.')

    population_total = np.sum(values)
    even_spacing = population_total / n
    random_start = np.random.uniform(low=0.0, high=even_spacing)
    selection_points = [random_start + i*even_spacing for i in range(0, n)]
    selection_idx = 0
    population_idx = 0
    running_sum = 0.0
    while True:
        # check if all points have been selected
        if selection_idx == len(selection_points):
            # reset to allow for continuous selection
            if shuffle:
                random_start = np.random.uniform(low=0.0, high=even_spacing)
                selection_points = [random_start + i*even_spacing
                                    for i in range(0, n)]
            selection_idx = 0
            running_sum = 0.0
            population_idx = 0

        current_point = selection_points[selection_idx]
        # continue until the running sum is greater than the point
        while running_sum < current_point:
            running_sum += values[population_idx]
            population_idx += 1
        selection_idx += 1

        # yield the individual that caused the running_sum
        # to move past the current_point
        yield population[population_idx-1]


##############################
# Function truncation_selection
##############################
@wrap_curry
@listlist_op
def truncation_selection(offspring: List, size: int,
                         parents: List = None,
                         key = None) -> List:
    """ return the `size` best individuals from the given population

        This defaults to (mu, lambda) if `parents` is not given.

        >>> from leap_ec.individual import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import truncation_selection
        >>> import numpy as np

        >>> pop = [Individual(np.array([0, 0, 0]), problem=MaxOnes()),
        ...        Individual(np.array([0, 0, 1]), problem=MaxOnes()),
        ...        Individual(np.array([1, 1, 0]), problem=MaxOnes()),
        ...        Individual(np.array([1, 1, 1]), problem=MaxOnes())]

        We need to evaluate them to get their fitness to sort them for
        truncation.

        >>> pop = Individual.evaluate_population(pop)

        >>> truncated = truncation_selection(pop, 2)

        TODO Do we want an optional context to over-ride the 'parents' parameter?

        :param offspring: offspring to truncate down to a smaller population
        :param size: is what to resize population to
        :param second_population: is optional parent population to include
                                  with population for downsizing
        :return: truncated population
    """
    if key:
        if parents is not None:
            return list(toolz.itertoolz.topk(
                size, itertools.chain(offspring, parents), key=key))
        else:
            return list(toolz.itertoolz.topk(size, offspring, key=key))
    else:
        if parents is not None:
            return list(toolz.itertoolz.topk(
                size, itertools.chain(offspring, parents)))
        else:
            return list(toolz.itertoolz.topk(size, offspring))


##############################
# Function elitist_survival
##############################
@wrap_curry
@listlist_op
def elitist_survival(offspring: List, parents: List, k: int = 1, key = None) -> List:
    """ This allows k best parents to compete with the offspring.

        >>> from leap_ec.individual import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> import numpy as np

        First, let's make a "pretend" population of parents using the MaxOnes
        problem.

        >>> pretend_parents = [Individual(np.array([0, 0, 0]), problem=MaxOnes()),
        ...                    Individual(np.array([1, 1, 1]), problem=MaxOnes())]

        Then a "pretend" population of offspring. (Pretend in that we're
        pretending that the offspring came from the parents.)

        >>> pretend_offspring = [Individual(np.array([0, 0, 0]), problem=MaxOnes()),
        ...                      Individual(np.array([1, 1, 0]), problem=MaxOnes()),
        ...                      Individual(np.array([1, 0, 1]), problem=MaxOnes()),
        ...                      Individual(np.array([0, 1, 1]), problem=MaxOnes()),
        ...                      Individual(np.array([0, 0, 1]), problem=MaxOnes())]

        We need to evaluate them to get their fitness to sort them for
        elitist_survival.

        >>> pretend_parents = Individual.evaluate_population(pretend_parents)
        >>> pretend_offspring = Individual.evaluate_population(pretend_offspring)

        This will take the best parent, which has [1,1,1], and replace the
        worst offspring, which has [0,0,0] (because this is the MaxOnes problem)
        >>> survivors = elitist_survival(pretend_offspring, pretend_parents)

        >>> assert pretend_parents[1] in survivors # yep, best parent is there
        >>> assert pretend_offspring[0] not in survivors # worst guy isn't

        We orginally ordered 5 offspring, so that's what we better have.
        >>> assert len(survivors) == 5

        Please note that the literature has a number of variations of elitism
        and other forms of overlapping generations.  For example, this may be a
        good starting point:

        De Jong, Kenneth A., and Jayshree Sarma. "Generation gaps revisited."
        In Foundations of genetic algorithms, vol. 2, pp. 19-28. Elsevier, 1993.

    :param offspring: list of created offpring, probably from pool()
    :param parents: list of parents, usually the ones that offspring came from
    :param k: how many elites from parents to keep?
    :param key: optional key criteria for selecting; e.g., can be used to impose
        parsimony pressure
    :return: surviving population, which will be offspring with offspring
        replaced by any superior parent elites
    """
    # We save this because we're going to truncate back down to this number
    # for the final survivors
    original_num_offspring = len(offspring)

    # Append the requested number of best parents to the offspring.
    if key:
        elites = list(toolz.itertoolz.topk(k, parents, key=key))
    else:
        elites = list(toolz.itertoolz.topk(k, parents))
    offspring.extend(elites)

    # Now return the offspring (plus possibly an elite) truncating the least
    # fit individual.
    if key:
        return list(toolz.itertoolz.topk(original_num_offspring, offspring, key))
    else:
        return list(toolz.itertoolz.topk(original_num_offspring, offspring))


##############################
# Function tournament_selection
##############################
@wrap_curry
@listiter_op
def tournament_selection(population: list, k: int = 2, key = None, select_worst: bool=False, indices = None) -> Iterator:
    """Returns an operator that selects the best individual from k individuals randomly selected from
        the given population.

        Like other selection operators, this assumes that if one individual is "greater than" another, then it is
        "better than" the other.  Whether this indicates maximization or minimization isn't handled here: the
        `Individual` class determines the semantics of its "greater than" operator.

        :param population: the population to select from.  Should be a list, not an iterator.
        :param int k: number of contestants in the tournament.  k=2 does binary tournament
            selection, which approximates linear ranking selection in the expectation.  Higher
            values of k yield greedier selection strategies—k=3, for instance, is equal to
            quadratic ranking selection in the expectation.
        :param key: an optional function that computes keys to sort over.  Defaults to None,
            in which case Individuals are compared directly.
        :param bool select_worst: if True, select the worst individual from the tournament instead
            of the best.
        :param list indices: an optional list that will be populated with the index of the
            selected individual.
        :return: the best of k individuals drawn from population

        >>> from leap_ec import Individual
        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.ops import tournament_selection
        >>> import numpy as np

        >>> pop = [Individual(np.array([0, 0, 0]), problem=MaxOnes()),
        ...        Individual(np.array([0, 0, 1]), problem=MaxOnes())]
        >>> pop = Individual.evaluate_population(pop)
        >>> best = tournament_selection(pop)
    """
    assert((indices is None) or (isinstance(indices, list))), f"Only a list should be passed to tournament_selection() for indices, but received {indices}."

    while True:
        choices_idx = random.choices(range(len(population)), k=k)
        judge = min if select_worst else max
        if key:
            best_idx = judge(choices_idx, key=lambda x: key(population[x]))
        else:
            best_idx = judge(choices_idx, key=lambda x: population[x])

        if indices is not None:
            indices.clear()  # Nuke whatever is in there
            indices.append(best_idx)  # Add the index of the individual we're about to return

        yield population[best_idx]


##############################
# Function insertion_selection
##############################
@wrap_curry
@listlist_op
def insertion_selection(offspring: List, parents: List, key = None) -> List:
    """ do exclusive selection between offspring and parents

    This is typically used for Ken De Jong's EV algorithm for survival
    selection.  Each offspring is deterministically selected and a random
    parent is selected; if the offspring wins, then it replaces the parent.

    Note that we make a _copy_ of the parents and have the offspring compete
    with the parent copies so that users can optionally preserve the original
    parents.  You may wish to do that, for example, if you want to analyze the
    composition of the original parents and the modified copy.

    :param offspring: population to select from
    :param parents: parents that are copied and which the copies are
           potentially updated with better offspring
    :param key: optional key for determining max() by other criteria such as
        for parsimony pressure
    :return: the updated parent population
    """
    copied_parents = copy(parents)
    for child in offspring:
        selected_parent_index = random.randrange(len(copied_parents))
        if key:
            copied_parents[selected_parent_index] = max(child,
                                                        copied_parents[
                                                            selected_parent_index],
                                                        key=key)
        else:
            copied_parents[selected_parent_index] = max(child,
                                                        copied_parents[
                                                            selected_parent_index])

        return copied_parents


##############################
# Function naive_cyclic_selection
##############################
@wrap_curry
@listiter_op
def naive_cyclic_selection(population: List, indices: List = None) -> Iterator:
    """ Deterministically returns individuals, and repeats the same test_sequence
    when exhausted.

    This is "naive" because it doesn't shuffle the population between complete
    tours to minimize bias.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import naive_cyclic_selection
    >>> import numpy as np

    >>> pop = [Individual(np.array([0, 0])),
    ...        Individual(np.array([0, 1]))]

    >>> cyclic_selector = naive_cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """

    for i, ind in itertools.cycle(enumerate(population)):
        if indices is not None:
            indices.clear()  # Nuke whatever is in there
            indices.append(i)  # Add the index of the individual we're about to return

        yield ind


##############################
# Function cyclic_selection
##############################
@wrap_curry
@listiter_op
def cyclic_selection(population: List) -> Iterator:
    """ Deterministically returns individuals in order, then shuffles the
    test_sequence, returns the individuals in that new order, and repeats this
    process.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import cyclic_selection
    >>> import numpy as np

    >>> pop = [Individual(np.array([0, 0])),
    ...        Individual(np.array([0, 1]))]

    >>> cyclic_selector = cyclic_selection(pop)

    :param population: from which to select
    :return: the next selected individual
    """
    # this is essentially itertools.cycle() that just shuffles
    # the saved test_sequence between cycles.
    saved = []
    for individual in population:
        yield individual
        saved.append(individual)
    while saved:
        # randomize the test_sequence between cycles to remove this source of sample
        # bias
        random.shuffle(saved)
        for individual in saved:
            yield individual


##############################
# Function random_selection
##############################
@wrap_curry
@listiter_op
def random_selection(population: List, indices = None) -> Iterator:
    """ return a uniformly randomly selected individual from the population

    :param population: from which to select
    :return: a uniformly selected individual
    """
    while True:
        choice_idx = random.choice(range(len(population)))

        if indices is not None:
            indices.clear()  # Nuke whatever is in there
            indices.append(choice_idx)  # Add the index of the individual we're about to return

        yield population[choice_idx]


##############################
# Function pool
##############################
@wrap_curry
@iterlist_op
def pool(next_individual: Iterator, size: int) -> List:
    """ 'Sink' for creating `size` individuals from preceding pipeline source.

    Allows for "pooling" individuals to be processed by next pipeline
    operator.  Typically used to collect offspring from preceding set of
    selection and birth operators, but could also be used to, say, "pool"
    individuals to be passed to an EDA as a training set.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import naive_cyclic_selection
    >>> import numpy as np

    >>> pop = [Individual(np.array([0, 0])),
    ...        Individual(np.array([0, 1]))]

    >>> cyclic_selector = naive_cyclic_selection(pop)

    >>> pool = pool(cyclic_selector, 3)

    print(pool)
    [Individual([0, 0], None, None), Individual([0, 1], None, None), Individual([0, 0], None, None)]

    :param next_individual: generator for getting the next offspring
    :param size: how many kids we want
    :return: population of `size` offspring
    """
    return [next(next_individual) for _ in range(size)]


##############################
# Function migrate
##############################
def migrate(topology, emigrant_selector,
            replacement_selector, migration_gap,
            customs_stamp=lambda x, _: x,
            metric=None,
            context=context):
    """
    A migration operator for use in island models.

    This operator works with multi-population algorithms,
    and is thus meant to used with :py:class:`leap_ec.algorithm.multi_population_ea`.

    Specifically, it assumes that

     1. the `population` argument passed into the returned function
        is a particular sub-population that we want to process
        "emigration" out of and "immigration" into,
     2. the `context` state object contains an integer field
        `context['leap']['generation']` indicating the current
        generation count of the algorithm, and
     3. the `context` also contains a integer field
        `context['leap']['current_subpopulation']` indicating the
        index of the subpopulation that is currently being processed
        in the overall collection of subpopulations (i.e. the one
        that `population` belongs to).

    These assumptions are essentially what :py:class:`leap_ec.algorithm.multi_population_ea`
    implements.

    >>> import networkx as nx
    >>> from leap_ec import ops, context
    >>> from leap_ec.data import test_population
    >>> pop0 = test_population[:]  # Shallow copy
    >>> pop1 = test_population[:]

    >>> op = migrate(topology=nx.complete_graph(2),
    ...              emigrant_selector=ops.tournament_selection,
    ...              replacement_selector=ops.random_selection,
    ...              migration_gap=50)
    >>> context['leap']['generation'] = 0
    >>> context['leap']['current_subpopulation'] = 0
    >>> op(pop0)
    [Individual<...>(...), Individual<...>(...), Individual<...>(...), Individual<...>(...)]

    >>> context['leap']['current_subpopulation'] = 1
    >>> op(pop1)
    [Individual<...>(...), Individual<...>(...), Individual<...>(...), Individual<...>(...)]

    This operator is a stateful closure: it maintains an
    internal list of all the out-going "emigrations" that
    occurred in the previous time step, so that it can
    process them as "immigrations" in the current time step.

    :param topology: a `networkx` topology defining the connectivity among islands
    :param emigrant_selector: a selection operator for choosing individuals to
        leave an island
    :param replacement_selector: a selection operator choosing contestants that
        will be replaced by an incoming immigrant if the immigrant has higher fitness
    :param int migration_gap: migration will occur regularly after every `migration_gap`
        evolutionary steps
    :param customs_stamp: an optional function to transfrom an individual upon its
        arrival to a new island.  This can be used, for example, to change the
        individual's decoder or problem in a heterogeneous island model.
    :param metric: an optional function of the form `f(generation, immigrant_individual, contestant_indidivudal, success)`
        for recording information about migration events.
    :param context: the context object to check for EA state, such as the current
        generation number, and the ID of the subpopulation that is currently
        being processed.

    """
    num_islands = topology.number_of_nodes()

    # We wrap a closure around some persistent state to keep trag of
    # immigrants as the move between populations
    immigrants = [[] for i in range(num_islands)]

    @listlist_op
    def do_migrate(population: List) -> List:
        current_subpop = context['leap']['current_subpopulation']
        logger.debug(f"Migration operator called on subpop {current_subpop} (generation: {context['leap']['generation']})")

        generation = context['leap']['generation']

        # Immigration
        for i, imm in enumerate(immigrants[current_subpop]):
            logger.debug(f"Processing immigrant {i+1} of {len(immigrants[current_subpop])} for subpop {current_subpop}.")
            # Do island-specific transformation
            # For example, this callback might update the individuals 'problem'
            # field to point to a new fitness function for the island, and
            # re-evalute its fitness.
            imm = customs_stamp(imm, current_subpop)

            # Compete for a place in the new population
            indices = [] # List to collect the selected index
            contestant = next(replacement_selector(population, indices=indices))
            contestant_index = indices[0]

            success = (imm >= contestant)
            if success:
                # Replace the contestant with the immgrant at the same position
                population[contestant_index] = imm

            if metric:
                metric(generation, imm, contestant, success)

        immigrants[current_subpop] = []

        # Emigration
        if generation % migration_gap == 0:
            logger.debug(f"migration_gap reached: doing emigration on subpop {current_subpop}.")
            # Choose an emigrant individual
            sponsor = next(emigrant_selector(population))
            logger.debug(f"Sponsor individual selected by emigrant_selector: {sponsor}")
            # Clone it and copy fitness
            emi = sponsor.clone()
            emi.fitness = sponsor.fitness
            logger.debug(f"Emigrant individual (copy of sponsor): {emi}")
            neighbors = topology.neighbors(
                current_subpop)  # Get neighboring islands
            # Randomly select a neighboring island
            dest = random.choice(list(neighbors))
            logger.debug(f"Destination island: {dest}")
            # Add the emigrant to its immigration list
            immigrants[dest].append(emi)

        return population

    return do_migrate


def migration_metric(stream, header: bool = True, notes: dict = None):
    """
    Returns a function that can be used to record migration events.

    The purpose of a migration metric is to record information about
    migrations that occur inside a migration operator.  Because these
    events take place inside the operator (rather than across operators),
    they cannot be recorded by a LEAP pipeline probe.

    In general, the interface for a migration metric function takes
    four parameters:

        - `generation`: the current generation
        - `immigrant_ind`: the individual that is attempting to migrate
        - `contestant_ind`: the individual that has been chosen to be replaced
        - `success`: True if the migration is successful, False otherwise

    The metric included here records the fitness of both individuals and writes
    them (along with the `generation` and `success` values) to a CSV.  You can
    write your own metric if you need to record other information (such as, say,
    genomes).

    >>> import sys
    >>> from leap_ec import Individual
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> m = migration_metric(sys.stdout,
    ...                      header=True,
    ...                      notes={'run': 0, 'description': 'Test output'}
    ... )
    run,description,generation,migrant_fitness,contestant_fitness,success

    >>> ind1 = Individual(np.array([1, 1, 1]), problem=MaxOnes())
    >>> f = ind1.evaluate()
    >>> contestant = Individual(np.array([0, 1, 1]), problem=MaxOnes())
    >>> f = contestant.evaluate()
    >>> m(0, ind1, contestant, True)
    0,Test output,0,3,2,True

    :param stream: file object to write the CSV data to
    :param bool header: a CSV header will be written if True
    :param dict notes: a dict specifying additional constant-value
        columns to include in the CSV output
    """
    notes = {} if notes is None else notes

    # Set up data collection if we're given a stream to write to
    if stream is None:
        writer = None
    else:
        fields = list(notes.keys()) + ['generation', 'migrant_fitness', 'contestant_fitness', 'success']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        if header:
            writer.writeheader()

    def measure_migration(generation, migrant, contestant, success: bool):
        """Write a row recording the given migration event."""
        if writer is not None:
            row_dict = {
                **notes,
                'generation': generation,
                'migrant_fitness': migrant.fitness,
                'contestant_fitness': contestant.fitness,
                'success': success
            }
            writer.writerow(row_dict)

    return measure_migration


##############################
# Class CooperativeEvaluate
##############################
def concat_combine(collaborators):
    """Combine a list of individuals by concatenating their genomes.

    You can choose whether this or some other function is used for combining
    collaborators by passing it into the `CooperativeEvaluate` constructor. """
    # Clone one of the evaluators so we can use its problem and decoder later
    combined_ind = collaborators[0].clone()

    genomes = [ind.genome for ind in collaborators]
    combined_ind.genome = np.concatenate(genomes)  # Concatenate
    return combined_ind


class CooperativeEvaluate(Operator):
    """A simple, non-parallel implementation of cooperative coevolutionary
    fitness evaluation.

    :param int num_trials: the number of combined solutions & fitness estimates
        to collect when computing a partial solution's fitness.
    :param collaborator_selector: a selection operator that we use to choose
        individuals from the *other* subpopulations to create a combined solution.
    :param context: the algorithm's state context.  Used to access
        subpopulation information.
    :param log_stream: optional file object to collect statistics about
        combined individuals to.
    :param combine: the function used to combine partial solutions into
        combined solutions.
    """

    def __init__(self, num_trials: int, collaborator_selector,
                 log_stream=None, combine=concat_combine,context=context):
        self.context = context
        self.num_trials = num_trials
        self.collaborator_selector = collaborator_selector
        self.combine = combine

        # Set up the CSV writier
        if log_stream is not None:
            self.log_writer = csv.DictWriter(
                log_stream,
                fieldnames=[
                    'generation',
                    'subpopulation',
                    'individual_type',
                    'collaborator_subpopulation',
                    'genome',
                    'fitness'])
            # We print the header at construction time
            self.log_writer.writeheader()
        else:
            self.log_writer = None

    def __call__(self, next_individual: Iterator) -> Iterator:
        """Execute the evaluation operator on a subpopulation."""
        while True:
            current_ind = next(next_individual)

            # Pull references to all subpopulations from the context object
            subpopulations = self.context['leap']['subpopulations']
            current_subpop = self.context['leap']['current_subpopulation']

            # Create iterators that select individuals from each subpopulation
            selectors = [self.collaborator_selector(
                subpop) for subpop in subpopulations]

            # Choose collaborators and evaulate
            fitnesses = []
            for i in range(self.num_trials):
                collaborators = self._choose_collaborators(
                    current_ind, subpopulations, current_subpop, selectors)
                combined_ind = self.combine(collaborators)
                fitness = combined_ind.evaluate()
                # Optionally write out data about the collaborations
                if self.log_writer is not None:
                    self._log_trial(
                        self.log_writer,
                        collaborators,
                        combined_ind,
                        i,
                        context=self.context)

                fitnesses.append(fitness)

            current_ind.fitness = mean(fitnesses)

            yield current_ind

    @staticmethod
    def _choose_collaborators(current_ind, subpopulations,
                              current_subpop, selectors):
        """Choose collaborators from the subpopulations."""
        collaborators = []
        for i in range(len(subpopulations)):
            if i != current_subpop:
                # Select a fellow collaborator from the other subpopulations
                ind = next(selectors[i])
                # Make sure we actually got something with a genome back
                assert (hasattr(ind, 'genome'))
                collaborators.append(ind)
            else:
                # Stick this subpop's individual in as-is
                collaborators.append(current_ind)

        assert (len(collaborators) == len(subpopulations))
        return collaborators

    @staticmethod
    def _log_trial(writer, collaborators, combined_ind, trial_id,
                   context=context):
        """Record information about a batch of collaborators to a CSV writer."""
        for i, collab in enumerate(collaborators):
            writer.writerow({'generation'                : context['leap']['generation'],
                             'subpopulation'             : context['leap']['current_subpopulation'],
                             'individual_type'           : 'Collaborator',
                             'collaborator_subpopulation': i,
                             'genome'                    : collab.genome,
                             'fitness'                   : collab.fitness})

        writer.writerow({'generation'                : context['leap']['generation'],
                         'subpopulation'             : context['leap']['current_subpopulation'],
                         'individual_type'           : 'Combined Individual',
                         'collaborator_subpopulation': None,
                         'genome'                    : combined_ind.genome,
                         'fitness'                   : combined_ind.fitness})


##############################
# function compute_expected_probability
##############################
def compute_expected_probability(expected_num_mutations: float,
                                 individual_genome: List) \
        -> float:
    """ Computed the probability of mutation based on the desired average
    expected mutation and genome length.

    The equation here is :math:`p = 1/L * \\texttt{expected_num_mutations}`.  To see why this is
    correct, note that the number of mutations performed is characterized by
    a binomial distribution with :math:`n=L` trials (one weighted "coin flip" per gene),
    and that the mean (expected_num_mutations) of a binomial distribution
    is given by :math:`n*p = expected_num_mutations`.

    :param expected_num_mutations: times individual is to be mutated on average
    :param individual_genome: genome for which to compute the probability
    :return: the corresponding probability of mutation
    """
    if expected_num_mutations > len(individual_genome):
        raise ValueError(f"Tried to compute a mutation probability for a parameter of expected_num_mutations={expected_num_mutations}, but this is greater than the genome length, {len(individual_genome)}, which is not allowed.")
    return 1.0 / len(individual_genome) * expected_num_mutations


##############################
# function compute_population_values
##############################
def compute_population_values(population: List, offset=0, exponent: int = 1,
                     key=lambda x: x.fitness) -> np.ndarray:
    """ Returns a list of values where the zero-point of the population is
        shifted and the values are scaled by exponentiation.

        :param population: the population to compute values from.
        :param offset: the offset from zero. Specifying `offset='pop-min'`
            will use the population's minimum value as the new zero-point.
            Defaults to 0.
        :param int exponent: the power to which values are raised to.
            Defaults to 1.
        :param key: a function that computes a metric based
            on an `Individual`.
        :return: a numpy array of values that have been shifted by `offset` and
            scaled by `exponent` corresponding to each individual in the
            population.
    """
    values = np.array([key(ind) for ind in population])
    if offset == 'pop-min':
        offset = -values.min(axis=0)
    return (values + offset) ** exponent


##############################
# function bernoulli_process
##############################
def random_bernoulli_vector(shape: Union[int, Tuple], p: float = 0.5) -> np.ndarray:
    """Generates a random vector of Boolean balues from a Bernoulli process—that is, from a 
    sequence of weighted coin flips.

    We use this function throughout LEAP because its implementation was found to
    be much faster than, say, just calling `np.random.choice([0, 1])`.

    >>> from leap_ec.ops import random_bernoulli_vector
    >>> random_bernoulli_vector(5, p=0.4)
    array([..., ..., ..., ..., ...])

    :param shape: shape of the random vector—can be an integer or a tuple.
    :param p: success probability of the bernoulli trials.
    :return: boolean numpy array 
    """
    assert(p >= 0 and p <= 1)
    shape = (shape,) if isinstance(shape, int) else shape
    return np.random.rand(*shape) < p