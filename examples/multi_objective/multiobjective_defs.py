#!/usr/bin/env python3
"""
    External definitions of functions to resolve multiprocess issue in
    notebooks as suggested by this:

    https://stackoverflow.com/a/42383397/2197955
"""
import pandas as pd
import numpy as np
import time
import random



from leap_ec.representation import Representation
from leap_ec.ops import tournament_selection, clone, evaluate, pool, UniformCrossover
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian, apply_hard_bounds
from leap_ec.probe import print_individual

from leap_ec.multiobjective.nsga2 import generalized_nsga_2
from leap_ec.multiobjective.problems import SCHProblem
from leap_ec.multiobjective.ops import fast_nondominated_sort, rank_ordinal_sort

SEED=42

MAX_GENERATIONS=5


sch_problem = SCHProblem()
sch_representation = Representation(initialize=create_real_vector(bounds=[(-10, 10)]))


def benchmark_nsga2(pop_size, algorithm, iteration, *_):
    generator = np.random.MT19937(SEED)
    np.random.set_state(generator.jumped(iteration).state)
    random.seed(np.random.sample())

    pipeline = [
        tournament_selection,
        clone,
        UniformCrossover(p_swap=0.2),
        mutate_gaussian(std=0.5, expected_num_mutations=1),
        evaluate,
        pool(size=pop_size)
    ]

    t = time.time()
    generalized_nsga_2(
        max_generations=MAX_GENERATIONS,
        pop_size=pop_size,
        problem=sch_problem,
        representation=sch_representation,
        pipeline=pipeline,
        rank_func={
            "fast_nondominated": fast_nondominated_sort,
            "rank_ordinal": rank_ordinal_sort
        }[algorithm]
    )

    return time.time() - t


def benchmark_wrapper(row):
    """ Used for multiprocess pool imap() """
    return benchmark_nsga2(*row)
