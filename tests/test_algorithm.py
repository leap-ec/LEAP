"""Unit tests for the algorithm package."""

import pytest

from leap_ec import Representation, ops, context
from leap_ec.island.algorithm import multi_population_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import MaxOnes

NUM_POPULATIONS = 4

GENERATIONS = 5

POP_SIZE = 10


##############################
# Function multi_pop_fixture()
##############################
@pytest.fixture
def multi_pop_fixture():
    """Set up a multi_population_ea() algorithm for testing."""
    pop_size = POP_SIZE
    generations = GENERATIONS
    num_populations = NUM_POPULATIONS
    # The number of genes in each subpopulation's individuals
    genes_per_subpopulation = [3, 4, 5, 8]

    def get_representation(length: int):
        """Return a representation that creates
        binary sequences of the given length."""
        assert (length > 0)
        return Representation(
            initialize=create_binary_sequence(length)
        )

    # Initialize a representation for each subpopulation
    representations = [get_representation(l) for l in genes_per_subpopulation]

    ea = multi_population_ea(max_generations=generations, pop_size=pop_size,
                             num_populations=num_populations,

                             # Fitness function
                             problem=MaxOnes(),

                             # Assign a poor initial fitness to individuals
                             init_evaluate=ops.const_evaluate(value=-100),

                             # Passing a list of representations causes
                             # different ones to be used for different subpops
                             representation=representations,

                             # Operator pipeline
                             shared_pipeline=[
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_bitflip(expected_num_mutations=1),
                                 ops.CooperativeEvaluate(
                                     num_trials=3,
                                     collaborator_selector=ops.random_selection),
                                 ops.pool(size=pop_size)
                             ])

    # Return the populations
    return ea


def test_multi_population_ea(multi_pop_fixture):
    """
    Ensure that at the end of a run, the final best-so-far individual that
    is returned for each subpop is the same as the best individual in context['leap']['subpopulations'].

    This is a basic check that the reference to the population in context is
    updated correctly.
    """
    # Run the algorithm and retrieve BSF records
    populations = multi_pop_fixture

    assert len(populations) == NUM_POPULATIONS

    # Check that the number of genes as specified are correct
    for pop in populations:
        for ind in pop:
            # TODO We should add a check that the populations have a
            # homogenous number of genes.
            assert len(ind.genome) in [3, 4, 5, 8]

    # Check that the global context is correct.
    context_pops = context['leap']['subpopulations']
    assert context_pops == populations
    assert (context['leap']['generation'] == GENERATIONS)
