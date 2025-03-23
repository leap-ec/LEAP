#!/usr/bin/env python3
"""
    This implements a simple Evolutionary Programming (EP) system, but it
    does not evolve state machines as done with the original EP approach.

    TODO convert to a state machines problem

"""
import os

from toolz import pipe

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.real_rep.problems import SpheroidProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population:
    :param generation:
    :return: None
    """
    for individual in population:
        print(generation, individual.genome, individual.fitness)


BROOD_SIZE = 3  # how many offspring each parent will reproduce

if __name__ == '__main__':
    # Define the real value bounds for initializing the population. In this case,
    # we define a genome of four bounds.

    # the (-5.12,5.12) was what was originally used for this problem in
    # Ken De Jong's 1975 dissertation, so was used for historical reasons.
    bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    parents = Individual.create_population(5,
                                           initialize=create_real_vector(
                                               bounds),
                                           decoder=IdentityDecoder(),
                                           problem=SpheroidProblem(
                                               maximize=False))

    # Evaluate initial population
    parents = Individual.evaluate_population(parents)

    # print initial, random population
    print_population(parents, generation=0)

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        max_generation = 2
    else:
        max_generation = 100

    # Set up a generation counter using the default global context variable
    generation_counter = util.inc_generation()

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.cyclic_selection,
                         ops.clone,
                         mutate_gaussian(std=.1, expected_num_mutations='isotropic'),
                         ops.evaluate,
                         # create the brood
                         ops.pool(size=len(parents) * BROOD_SIZE),
                         # mu + lambda
                         ops.truncation_selection(size=len(parents),
                                                  parents=parents))
        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        print_population(parents, context['leap']['generation'])
