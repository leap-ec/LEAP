#!/usr/bin/env python3
"""
    Ensures that the third code snippet in top-level README.md works.
"""
from toolz import pipe

from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.global_vars import context

import leap_ec.ops as ops
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec import util, probe

# create initial rand population of 5 individuals
parents = Individual.create_population(5,
                                       initialize=create_binary_sequence(4),
                                       decoder=IdentityDecoder(),
                                       problem=MaxOnes())
# Evaluate initial population
parents = Individual.evaluate_population(parents)

# print initial, random population
probe.print_population(parents, generation=0)

# generation_counter is an optional convenience for generation tracking
generation_counter = util.inc_generation(context=context)

while generation_counter.generation() < 6:
    offspring = pipe(parents,
                     ops.tournament_selection,
                     ops.clone,
                     mutate_bitflip(expected_num_mutations=1),
                     ops.UniformCrossover(p_swap=0.2),
                     ops.evaluate,
                     ops.pool(size=len(parents)))  # accumulate offspring

    parents = offspring

    generation_counter()  # increment to the next generation

    probe.print_population(parents, context['leap']['generation'])
