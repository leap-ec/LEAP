#!/usr/bin/env python3
"""
    LEAP pipeline operators for multiobjective optimization.

    For now this just implements NSGA-II, but other multiobjective approaches
    will eventually be included.
"""
from typing import Iterator
import random
from itertools import chain

import toolz
from math import inf

import numpy as np

from leap_ec.ops import compute_expected_probability, listlist_op, iteriter_op
from leap_ec.util import wrap_curry
from .problems import MultiObjectiveProblem

##############################
# sort_by_dominance operator
##############################
@wrap_curry
@listlist_op
def sort_by_dominance(population: list) -> list:
    """ Sort population by rank and distance

        This presumes that fast_nondominated_sort() *and* crowding_distance_calc
        have been used on *all* individuals in `population`.

        :param population: to be sorted
        :returns: sorted population
    """
    return sorted(population, key=lambda x: (x.rank, -x.distance))


##############################
# fast_nondominated_sort operator
##############################
@wrap_curry
@listlist_op
def fast_nondominated_sort(population: list, parents: list = None) -> list:
    """ This implements the NSGA-II fast-non-dominated-sort()

    This is really *binning* the population by ranks.  In any case, the
    returned population will have an attribute, `rank`, that will denote
    the corresponding rank in which it is a member.

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.

    :param population: population to be ranked
    :param parents: optional parents population to be included with the ranking
        process
    :returns: individuals binned by ranks
    """
    # Ensure that we're dealing with a multi-objective Problem.
    assert isinstance(population[0].problem, MultiObjectiveProblem)

    # Have a separate working copy of the populations
    if parents is not None:
        working_pop = population + parents
    else:
        working_pop = population

    ranks = {1: []}  # rank 1 initially empty

    # First, find rank 1
    for individual in working_pop:
        individual.dominates = []
        individual.dominated_by = 0
        # there is no rank 0, we're just including in initialization as a
        # reality check; i.e., if we see rank zeroes floating around while
        # debugging, we know something went wrong
        individual.rank = 0

        for other_individual in working_pop:
            if individual is other_individual:
                continue
            if individual > other_individual:
                individual.dominates.append(other_individual)
            elif other_individual > individual:
                individual.dominated_by += 1

        if individual.dominated_by == 0:
            individual.rank = 1
            ranks[1].append(individual)

    # Now fill out the remaining ranks
    i = 1
    while ranks.get(i, []) != []:
        next_front = []
        for individual in ranks[i]:
            for other_individual in individual.dominates:
                other_individual.dominated_by -= 1
                if other_individual.dominated_by == 0:
                    other_individual.rank = i + 1
                    next_front.append(other_individual)

        i += 1
        ranks[i] = next_front

    # the parents will have been updated, too, but the next pipeline operator
    # will also look at them
    return working_pop

##############################
# rank_ordinal_sort operator
##############################
@wrap_curry
@listlist_op
def rank_ordinal_sort(population: list, parents: list = None) -> list:
    """ This implements Rank Ordinal Sort from Rank-based Non-dominated Sorting

    Produces identical `rank` values to `fast_nondominated_sort` from the original
    NSGA-II implementation, however performs much faster.

    - Bogdan Burlacu. 2022. Rank-based Non-dominated Sorting. arXiv.
      DOI:https://doi.org/10.48550/ARXIV.2203.13654

    :param population: population to be ranked
    :param parents: optional parents population to be included with the ranking
        process
    :returns: individuals binned by ranks
    """
    if parents is not None:
        population += parents

    # De-duplicate indentical fitness values, using the returned fitnesses for the sorting
    unique_fitnesses, orig_inv_idx = np.unique([
        ind.fitness * ind.problem.maximize
        for ind in population
    ], return_inverse=True, axis=0)

    # Determine the per-objective ordering of the population
    # Proper permutation ordering is the transposed and reversed argsort
    permutations = np.argsort(unique_fitnesses, axis=0, kind="stable").T[:, ::-1]

    # Determine objective ranks from the unique values of each fitness
    objective_ranks = np.zeros(permutations.shape[::-1], dtype=np.uint)
    for i, permute in enumerate(permutations):
        _, unique_idx, inv_idx = np.unique(unique_fitnesses[permute, i], return_index=True, return_inverse=True)
        objective_ranks[permute, i] = unique_idx[inv_idx]

    ranks = np.zeros(len(unique_fitnesses), dtype=np.uint)
    for i in permutations[0]:
        k = np.argmax(objective_ranks[i])

        # Get the set of values that may be dominable
        dominable_set = permutations[k, objective_ranks[i, k]:]

        # Get the subset of values that match i's rank, and therefore can be updated
        updatable_set = dominable_set[ranks[i] == ranks[dominable_set]]

        # Determine what of the subset is dominated and increment those value's rank
        is_dominated = (objective_ranks[i] <= objective_ranks[updatable_set]).all(1)
        ranks[updatable_set[is_dominated]] += 1

        # Since i is always in the dominable set, every value's rank is +1
        # This works out in the end by using np.zeros, and the return front rank is still rank == 1

    # Use the inverse idx from deduplicating fitnesses to assign rankings
    for ind, rank in zip(population, ranks[orig_inv_idx]):
        ind.rank = int(rank)

    return population

##############################
# crowding_distance_calc operator
##############################
def per_rank_crowding_calc(ranked_population: list, is_maximizing) -> list:
    """ Calculate crowding distance within rank
    :param ranked_population: A population of entirely one rank
    :returns: population with crowding distance calculate for one rank
    """
    if len(ranked_population) == 1:
        # There is just one individual in this rank, so set their distance to 0
        # since there are no neighbors.
        ranked_population[0].distance = 0
        return ranked_population

    # Presuming this is a population with homogeneous objectives, then the size of
    # the optimization directions array should be equal to the number of objectives.
    num_objectives = is_maximizing.shape[0]
    
    # minimum and maximum fitnesses by objective, so we initialize to the
    # infinities. At first we assume maximization for all of the objectives,
    # but then we fine-tune for minimization in the next step.
    f_min = np.full(num_objectives, np.inf)
    f_max = np.full(num_objectives, -np.inf)

    for objective in range(num_objectives):
        if is_maximizing[objective] == -1:
            f_min[objective] = -np.inf
            f_max[objective] = np.inf

    # Find ranges of fitness per objective
    for i in ranked_population:
        i.distance = 0  # init distances to zero to start
        for objective in range(num_objectives):  # update fitness ranges
            if is_maximizing[objective] == -1:
                # We are *maximizing* for this specific objective
                f_min[objective] = max(f_min[objective],
                                        i.fitness[objective])
                f_max[objective] = min(f_max[objective],
                                        i.fitness[objective])
            else:
                # We are *minimizing* for this specific objective
                f_min[objective] = min(f_min[objective],
                                        i.fitness[objective])
                f_max[objective] = max(f_max[objective],
                                        i.fitness[objective])

    objective_ranges = f_max - f_min

    sorted_pop = []

    for objective in range(num_objectives):
        if objective_ranges[objective] == 0 or objective_ranges[objective] == np.nan:
            continue
        
        # sort by objective being mindful that maximization vs. minimization may
        # be different for each objective
        if is_maximizing[objective] == -1:
            # If we're maximizing in ascending order, that actually means we
            # want descending order since the larger values are fitter.
            sorted_pop = sorted(ranked_population,
                                key=lambda ind: - ind.fitness[objective])
        else:
            sorted_pop = sorted(ranked_population,
                                key=lambda ind: ind.fitness[objective])

        # set first and last elements to infinity
        sorted_pop[0].distance = sorted_pop[-1].distance = inf

        # update the distance per individuals with a sliding window of
        # three fitnesses for the current objective starting from the second to
        # the second to last individual's
        for i in range(1, len(sorted_pop) - 1):
            sorted_pop[i].distance = sorted_pop[i].distance + \
                                        (sorted_pop[i + 1].fitness[objective] -
                                        sorted_pop[i - 1].fitness[
                                            objective]) / objective_ranges[
                                            objective]

    # return ranked_population not sorted since sometimes sorted_pop will be empty
    return ranked_population

@wrap_curry
@listlist_op
def crowding_distance_calc(population: list) -> list:
    """ This implements the NSGA-II crowding-distance-assignment()

    Note that this assumes that all the individuals have had their ranks
    computed since we do crowding distance calculations within ranks.

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.

    :param population: population to calculate crowding distances
    :returns: individuals with crowding distance calculated
    """

    # Ensure that we're dealing with a multi-objective Problem.
    assert isinstance(population[0].problem, MultiObjectiveProblem)

    # Check if we're maximizing or minimizing; we arbitrarily check the first
    # individual.
    # TODO We *might* have to check on a case by case basis if we have a weird
    # corner case whereby the population has a mix of different problems.  Or,
    # if there is a mix, that they're homogeneous with regards to maximizing.
    # Note that MultiObjectiveProblem.maximize is a numpy array where a -1 or 1
    # signifies whether we're dealing with maximizing or minimizing. A -1
    # means we're maximizing for that corresponding objective.
    is_maximizing = population[0].problem.maximize

    # First, divide up the population into sub-populations by rank
    pop_by_ranks = toolz.groupby(lambda x: x.rank, population)

    # Eventually the entire population after crowding distance calculated for
    # all the sub-populations by rank.
    all_crowd_dist_pop = []
    for rank in pop_by_ranks.keys():
        all_crowd_dist_pop += per_rank_crowding_calc(pop_by_ranks[rank], is_maximizing)

    return all_crowd_dist_pop
