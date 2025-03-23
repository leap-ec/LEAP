#!/usr/bin/env python3
"""
    EV is a simple ES-like EA invented by Ken De Jong for educational purposes.

    Essentially, an EV uses a real-valued representation, and a pre-defined,
    static standard deviation applied to the Gaussian mutation operator.

    Note that there are sibling examples the demonstrate more true
    evolutionary strategy-like approaches. """
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
POPULATION_SIZE = 10


# When running the test harness, just run for two generations
# (we use this to quickly ensure our examples don't get bitrot)
if os.environ.get(test_env_var, False) == 'True':
    MAX_GENERATIONS = 2
else:
    MAX_GENERATIONS = 5


##############################
# Entry point
##############################
if __name__ == '__main__':
    # Define the real value bounds for initializing the population. In this
    # case, we define a genome of four bounds.

    # the (-5.12,5.12) was what was originally used for this problem in
    # Ken De Jong's 1975 dissertation, so was used for historical reasons.
    bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    parents = Individual.create_population(POPULATION_SIZE,
                                                initialize=create_real_vector(
                                                    bounds),
                                                decoder=IdentityDecoder(),
                                                problem=SpheroidProblem(maximize=False))

    # Evaluate initial population
    parents = Individual.evaluate_population(parents)

    # print initial, random population
    print_population(parents, generation=0)

    max_generation = MAX_GENERATIONS

    # Set up a generation counter using the default global context variable
    generation_counter = util.inc_generation()

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.random_selection,
                         ops.clone,
                         mutate_gaussian(std=.1, expected_num_mutations=1),
                         ops.evaluate,
                         ops.pool(
                             size=len(parents) * BROOD_SIZE),
                         # create the brood
                         ops.insertion_selection(parents=parents))

        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        print_population(parents, context['leap']['generation'])
