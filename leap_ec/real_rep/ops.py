#!/usr/bin/env python3
"""
    Pipeline operators for real-valued representations
"""
import math
from collections.abc import Iterable
from typing import Iterator, List, Tuple, Union

import numpy as np

from leap_ec.util import wrap_curry
from leap_ec.ops import compute_expected_probability, iteriter_op, random_bernoulli_vector, Crossover


##############################
# Function mutate_gaussian
##############################
@wrap_curry
@iteriter_op
def mutate_gaussian(next_individual: Iterator,
                    std,
                    expected_num_mutations: Union[int, str] = None,
                    bounds=(-math.inf, math.inf),
                    transform_slope: float = 1.0,
                    transform_intercept: float = 0.0) -> Iterator:
    """Mutate and return an Individual with a real-valued representation.

    This operators on an iterator of Individuals:

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> import numpy as np
    >>> pop = iter([Individual(np.array([1.0, 0.0]))])

    Mutation can either use the same parameters for all genes:

    >>> op = mutate_gaussian(std=1.0, expected_num_mutations='isotropic', bounds=(-5, 5))
    >>> mutated = next(op(pop))

    Or we can specify the `std` and `bounds` independently for each gene:

    >>> pop = iter([Individual(np.array([1.0, 0.0]))])
    >>> op = mutate_gaussian(std=[0.5, 1.0],
    ...                      expected_num_mutations='isotropic',
    ...                      bounds=[(-1, 1), (-10, 10)]
    ... )
    >>> mutated = next(op(pop))

    :param next_individual: to be mutated
    :param std: standard deviation to be equally applied to all individuals;
        this can be a scalar value or a "shadow vector" of standard deviations
    :param expected_num_mutations: if an int, the *expected* number of mutations per
        individual, on average.  If 'isotropic', all genes will be mutated.
    :param bounds: to clip for mutations; defaults to (- ∞, ∞)
    :return: a generator of mutated individuals.
    """
    if expected_num_mutations is None:
        raise ValueError("No value given for expected_num_mutations.  Must be either a float or the string 'isotropic'.")
    
    genome_mutator = genome_mutate_gaussian(std=std,
                                expected_num_mutations=expected_num_mutations,
                                bounds=bounds,
                                transform_slope=transform_slope,
                                transform_intercept=transform_intercept)

    while True:
        individual = next(next_individual)

        individual.genome = genome_mutator(individual.genome)
        # invalidate fitness since we have new genome
        individual.fitness = None

        yield individual


##############################
# Function genome_mutate_gaussian
##############################
@wrap_curry
def genome_mutate_gaussian(genome,
                           std: float,
                           expected_num_mutations,
                           bounds: Tuple[float, float] =
                             (-math.inf, math.inf),
                           transform_slope: float = 1.0,
                           transform_intercept: float = 0.0):
    """Perform Gaussian mutation directly on real-valued genes (rather than
    on an Individual).

    This used to be inside `mutate_gaussian`, but was moved outside it so that
    `leap_ec.segmented.ops.apply_mutation` could directly use this function,
    thus saving us from doing a copy-n-paste of the same code to the segmented
    sub-package.

    :param genome: of real-valued numbers that will potentially be mutated
    :param std: the mutation width—either a single float that will be used for
        all genes, or a list of floats specifying the mutation width for
        each gene individually.
    :param expected_num_mutations: on average how many mutations are expected
    :return: mutated genome
    """
    assert(std is not None)
    assert(isinstance(std, Iterable) or (std >= 0.0))
    assert(expected_num_mutations is not None)

    if isinstance(std, Iterable):
        std = np.array(std)

    # compute actual probability of mutation based on expe cted number of
    # mutations and the genome length

    if not isinstance(genome, np.ndarray):
        raise ValueError(("Expected genome to be a numpy array. "
                        f"Got {type(genome)}."))

    if expected_num_mutations == 'isotropic':
        # Default to isotropic Gaussian mutation
        p = 1.0
    else:
        p = compute_expected_probability(expected_num_mutations, genome)

    # select which indices to mutate at random
    indices_to_mutate = random_bernoulli_vector(shape=genome.shape, p=p)

    # Pick out just the std values we need for the mutated genes
    std_selected = std if not isinstance(std, Iterable) else std[indices_to_mutate]

    # Apply additive Gaussian noise to the selected genes
    new_gene_values = transform_slope * (genome[indices_to_mutate] \
                                                    + np.random.normal(size=sum(indices_to_mutate)) \
                                                    # scalar multiply if scalar; element-wise if std is an ndarray
                                                    * std_selected) \
                                + transform_intercept
    genome[indices_to_mutate] = new_gene_values

    # Implement hard bounds
    genome = apply_hard_bounds(genome, bounds)

    return genome


##############################
# Function apply_hard_bounds
##############################
def apply_hard_bounds(genome, hard_bounds):
    """A helper that ensures that every gene is contained within the given bounds.

    :param genome: list of values to apply bounds to.
    :param hard_bounds: if a `(low, high)` tuple, the same bounds will be used for every gene.
        If a list of tuples is given, then the ith bounds will be applied to the ith gene.

    Both sides of the range are inclusive:

    >>> genome = np.array([0, 10, 20, 30, 40, 50])
    >>> apply_hard_bounds(genome, hard_bounds=(20, 40))
    array([20, 20, 20, 30, 40, 40])

    Different bounds can be used for each locus by passing in a list of tuples:

    >>> bounds= [ (0, 1), (0, 1), (50, 100), (50, 100), (0, 100), (0, 10) ]
    >>> apply_hard_bounds(genome, hard_bounds=bounds)
    array([ 0,  1, 50, 50, 40, 10])
    """
    assert(genome is not None)
    assert(isinstance(genome, Iterable))
    assert(hard_bounds is not None)

    if not isinstance(hard_bounds[0], Iterable):
        # scalar bounds apply to each gene
        low = hard_bounds[0]
        high = hard_bounds[1]
    elif isinstance(hard_bounds, np.ndarray):
        low = hard_bounds[:, 0]
        high = hard_bounds[:, 1]
    else:
        # Looping through twice here is faster than converting to
        # numpy array and slicing for the column
        low = [bound[0] for bound in hard_bounds]
        high = [bound[1] for bound in hard_bounds]

    return np.clip(genome, a_min=low, a_max=high)


##############################
# Function mutate_bpm
##############################
@wrap_curry
@iteriter_op
def mutate_bpm(next_individual: Iterator,
               eta: float,
               expected_num_mutations: Union[int, str] = None,
               bounds: Tuple[float, float] = None) -> Iterator:
    """Mutate and return an Individual with a real-valued representation.

    This operators on an iterator of Individuals:

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.real_rep.ops import mutate_bpm
    >>> import numpy as np
    >>> pop = iter([Individual(np.array([1.0, 0.0]))])

    Mutation can either use the same parameters for all genes:

    >>> op = mutate_bpm(eta=10, expected_num_mutations='isotropic', bounds=(-5, 5))
    >>> mutated = next(op(pop))

    Or we can specify the `eta` and `bounds` independently for each gene:

    >>> pop = iter([Individual(np.array([1.0, 0.0]))])
    >>> op = mutate_bpm(eta=[10, 20],
    ...                      expected_num_mutations='isotropic',
    ...                      bounds=[(-1, 1), (-10, 10)]
    ... )
    >>> mutated = next(op(pop))

    :param next_individual: to be mutated
    :param eta: eta value of the mutation, higher values are more closely distributed
        to the original parameters
    :param expected_num_mutations: if an int, the *expected* number of mutations per
        individual, on average.  If 'isotropic', all genes will be mutated.
    :param bounds: to clip for mutations, also factors into the distribution
    :return: a generator of mutated individuals.
    """
    if expected_num_mutations is None:
        raise ValueError("No value given for expected_num_mutations.  Must be either a float or the string 'isotropic'.")
    if bounds is None:
        raise ValueError("No bounds given for Bounded Polynomial Mutation. Must be a tuple of bounds broadcastable to the genome.")
    
    genome_mutator = genome_mutate_bpm(eta=eta,
                                expected_num_mutations=expected_num_mutations,
                                bounds=bounds)

    while True:
        individual = next(next_individual)

        individual.genome = genome_mutator(individual.genome)
        individual.fitness = None

        yield individual


##############################
# Function genome_mutate_bpm
##############################
@wrap_curry
def genome_mutate_bpm(genome,
                      eta: float,
                      expected_num_mutations,
                      bounds: Tuple[float, float]):

    if not isinstance(genome, np.ndarray):
        raise ValueError(("Expected genome to be a numpy array. "
                        f"Got {type(genome)}."))

    if expected_num_mutations == 'isotropic':
        p = 1.0
    else:
        p = compute_expected_probability(expected_num_mutations, genome)

    indices_to_mutate = random_bernoulli_vector(shape=genome.shape, p=p)

    num_mut = np.sum(indices_to_mutate)
    if num_mut == 0:
        return genome

    etap1 = 1 + (eta if np.ndim(eta) == 0 else np.array(eta)[indices_to_mutate]) 
    broadcast_bounds = np.broadcast_to(bounds, (*genome.shape, 2))[indices_to_mutate]
    low, high = np.moveaxis(broadcast_bounds, -1, 0)
    diff = high - low
    
    v = genome[indices_to_mutate]
    r = np.random.random(size=num_mut) * 2 - 1

    xy = np.where(r < 0, (high-v), (v-low)) / diff

    b = 1 + np.abs(r) * (xy ** etap1 - 1)
    dv = np.copysign((1 - b ** (1/etap1)) * diff, r)
    
    genome[indices_to_mutate] += dv
    genome = apply_hard_bounds(genome, bounds)
    
    return genome


##############################
# Simulated Binary Crossover class
##############################
class SimulatedBinaryCrossover(Crossover):
    """Parameterized simulated binary crossover iterates through two parents' genomes,
    mutates them based on their compared values, and probabilistically swaps genes.

    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import naive_cyclic_selection
    >>> from leap_ec.real_rep.ops import SimulatedBinaryCrossover
    >>> import numpy as np

    >>> genome1 = np.array([0., 0.])
    >>> genome2 = np.array([1., 1.])
    >>> first = Individual(genome1)
    >>> second = Individual(genome2)
    >>> pop = [first, second]
    >>> select = naive_cyclic_selection(pop)
    >>> op = SimulatedBinaryCrossover(eta=1, bounds=(-10, 10))
    >>> result = op(select)
    >>> new_first = next(result)
    >>> new_second = next(result)

    The closeness of the mutation can be controlled by eta, the probability of performing
    simulated binary crossover with p_sbx, and the probability of swapping with p_swap:
    >>> op = SimulatedBinaryCrossover(eta=10, p_sbx=0.5, p_swap=0.1, bounds=(-10, 10))
    >>> result = op(select)
    
    :param eta: the eta value of the crossover, higher values are more closely distributed
        to the original parameters
    :param p_sbx: how likely we are to perform sbx on a pair of genes
    :param p_swap: how likely are we to swap each pair of genes when crossover
        is performed
    :param float p_xover: the probability that crossover is performed in the
        first place
    :param bounds: to clip for crossover, also factors into the distribution
    :param bool persist_children: whether unyielded children should persist between calls.
        This is useful for `leap_ec.distrib.asynchronous.steady_state`, where the pipeline
        may only produce one individual at a time.
    :return: a pipeline operator that returns two recombined individuals (with probability
        p_xover), or two unmodified individuals (with probability 1 - p_xover)
    """

    @staticmethod
    def apply(gene_a: np.ndarray, gene_b: np.ndarray, eta: float, p_swap: float, bounds: Tuple[float, float]):
        # Computation of sbx for the specified parameters
        is_a_high = gene_a > gene_b 
        gene_stack = np.stack((gene_a, gene_b), axis=0)
        gene_stack = np.where(is_a_high[None,], gene_stack, gene_stack[::-1])
        g_low, g_high = gene_stack

        bounds = np.moveaxis(bounds, -1, 0)
        diff = bounds[1] - bounds[0]
        etap1 = eta + 1
        r = np.random.random(gene_a.shape)

        beta = 1 + 2 * (g_high - bounds[0]) / diff
        v = 1 - r * (2 - beta ** -etap1)
        q = (1 - np.abs(v)) ** np.copysign(1/etap1, v)
        g_high += (1 - q) * diff / 2

        beta = 1 + 2 * (bounds[1] - g_low) / diff
        v = 1 - r * (2 - beta ** -etap1)
        q = (1 - np.abs(v)) ** np.copysign(1/etap1, v)
        g_low -= (1 - q) * diff / 2

        np.clip(gene_stack, *bounds, out=gene_stack)
        do_swap = np.random.random(gene_a.shape) < p_swap
        return np.where(is_a_high == do_swap, gene_stack, gene_stack[::-1])

    def __init__(self, eta: float, p_sbx:float=1.0, p_swap: float=0.2, bounds: Tuple[float, float]=None, p_xover: float=1.0, persist_children=False):
        if bounds is None:
            raise ValueError("No bounds given for SimulatedBinaryCrossover. Must be a tuple of bounds broadcastable to the genome.")
        
        super().__init__(p_xover=p_xover, persist_children=persist_children)
        self.eta = eta
        self.bounds = bounds
        self.p_sbx = p_sbx
        self.p_swap = p_swap

    def recombine(self, parent_a, parent_b):
        """
        Perform recombination between two parents to produce two new individuals.
        """
        assert(isinstance(parent_a.genome, np.ndarray))
        assert(isinstance(parent_b.genome, np.ndarray))

        # generate which indices we should sbx
        min_length = min(parent_a.genome.shape[0], parent_b.genome.shape[0])
        indices_to_sbx = random_bernoulli_vector(min_length, self.p_sbx)

        eta = self.eta if np.ndim(self.eta) == 0 else np.array(self.eta)[indices_to_sbx]
        bounds = np.broadcast_to(self.bounds, (min_length, 2))[indices_to_sbx]

        # perform sbx
        gene_a = parent_a.genome[indices_to_sbx]
        gene_b = parent_b.genome[indices_to_sbx]
        gene_a, gene_b = self.apply(gene_a, gene_b, eta, self.p_swap, bounds)

        parent_a.genome[indices_to_sbx] = gene_a
        parent_b.genome[indices_to_sbx] = gene_b

        return parent_a, parent_b


