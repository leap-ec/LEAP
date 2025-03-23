""" Unit tests for pipeline operators in the segmented representation package. """
import functools
import numpy as np
import pytest
from leap_ec.binary_rep.ops import genome_mutate_bitflip
from leap_ec.int_rep.ops import genome_mutate_binomial, individual_mutate_randint
from leap_ec.individual import Individual
from leap_ec.ops import NAryCrossover
from leap_ec.segmented_rep.ops import apply_mutation, remove_segment, \
    add_segment, copy_segment

##############################
# Test Fixtures
##############################
test_sequence = np.array(
    [2, 2])  # just an arbitrary sequence of twos for testing


@pytest.fixture
def gen_sequence():
    """Return a function that returns an arbitrary static test_sequence."""

    def f():
        return test_sequence

    return f


def in_possible_outcomes(test_seq, possible_outcomes):
    """ :returns: true if test_seq in possible_outcomes """
    return any([np.array_equiv(test_seq, x) for x in possible_outcomes])


##############################
# Tests for apply_mutation()
##############################
def test_apply_mutation():
    """Applying segment-wise mutation operators when the nested operator has
    expected_num_mutations=len(genome) should result in every gene of every
    segment being mutated."""
    mutation_op = apply_mutation(mutator=genome_mutate_bitflip(expected_num_mutations=2))
    original = Individual([np.array([0, 0]), np.array([1, 1])])
    mutated = next(mutation_op(iter([original])))

    assert np.all(mutated.genome[0] == [1, 1]) \
           and np.all(mutated.genome[1] == [0, 0])


def test_apply_mutation_uniform_int():
    """ Same test, but with integer mutation.  Setting the bounds artificially small
    to make the resulting mutations choose deterministic values for us to test."""
    mutator = individual_mutate_randint(bounds=[(10, 10), (110, 110)], expected_num_mutations=2)

    mutation_op = apply_mutation(mutator=mutator)
    original = Individual([np.array([0, 100]), np.array([1, 101])])
    mutated = next(mutation_op(iter([original])))

    assert np.all(mutated.genome[0] == [10, 110]) \
           and np.all(mutated.genome[1] == [10, 110])


def test_apply_mutation_binomial_int():
    """ Same test, but with integer mutation """
    mutator = genome_mutate_binomial(
        std=1.0,
        bounds=[(0, 10), (100, 110)],
        probability=1)
    mutation_op = functools.partial(apply_mutation,
                                    mutator=mutator)
    original = Individual([np.array([0, 100]), np.array([1, 101])])
    mutated = next(mutation_op(iter([original])))

    pass


##############################
# Tests for remove_segment()
##############################


def test_segmented_remove():
    original = Individual([np.array([0, 0]), np.array([1, 1])])
    mutated = next(remove_segment(iter([original]), probability=1.0))
    assert (mutated.genome[0] == [np.array([0, 0]), np.array([1, 1])]).any()


##############################
# Tests for add_segment()
##############################
def test_segmented_add(gen_sequence):
    # Test with append first
    original = Individual([np.array([0, 0]), np.array([1, 1])])
    mutated = next(add_segment(iter([original]),
                               seq_initializer=gen_sequence,
                               probability=1.0,
                               append=True))
    target = [np.array([0, 0]), np.array([1, 1]), test_sequence]
    assert np.array_equiv(mutated.genome, target)

    # Test without append, so segment can inserted in one of three locations
    possible_outcomes = [[test_sequence, np.array([0, 0]), np.array([1, 1])],
                         [np.array([0, 0]), test_sequence, np.array([1, 1])],
                         [np.array([0, 0]), np.array([1, 1]), test_sequence]]

    for i in range(20):
        original = Individual([np.array([0, 0]), np.array([1, 1])])
        mutated = next(add_segment(iter([original]),
                                   seq_initializer=gen_sequence,
                                   probability=1.0,
                                   append=False))

        assert in_possible_outcomes(mutated.genome, possible_outcomes)


##############################
# Tests for copy_segment()
##############################
def test_segmented_copy():
    original = Individual([np.array([0, 0]), np.array([1, 1])])
    mutated = next(copy_segment(iter([original]),
                                probability=1.0,
                                append=True))

    possible_outcomes = [[np.array([0, 0]), np.array([1, 1]), np.array([0, 0])],
                         [np.array([0, 0]), np.array([1, 1]), np.array([1, 1])],
                         ]

    assert in_possible_outcomes(mutated.genome, possible_outcomes)

    possible_outcomes = [[np.array([0, 0]), np.array([0, 0]), np.array([1, 1])],
                         [np.array([0, 0]), np.array([1, 1]), np.array([0, 0])],
                         [np.array([1, 1]), np.array([0, 0]), np.array([1, 1])],
                         [np.array([0, 0]), np.array([1, 1]), np.array([1, 1])],
                         ]

    # TODO: it would be better to build a histogram of expected outcomes and
    # do the chi square test
    for i in range(20):
        original = Individual([np.array([0, 0]), np.array([1, 1])])
        mutated = next(copy_segment(iter([original]),
                                    probability=1.0,
                                    append=False))

        assert in_possible_outcomes(mutated.genome, possible_outcomes)


##############################
# Tests for n_ary_crossover() on segmented genomes
##############################
def test_segmented_crossover():
    """ test that n-ary crossover works as expected for fixed and variable
        length segments
    """
    a = Individual([np.array([0, 0]), np.array([1, 1])])
    b = Individual([np.array([1, 1]), np.array([0, 0])])

    result = NAryCrossover()(iter([a, b]))
    c = next(result)
    d = next(result)

    possible_outcomes = [(np.array([0, 0]), np.array([1, 1])),
                         (np.array([1, 1]), np.array([0, 0])),
                         (np.array([0, 0]), np.array([0, 0])),
                         (np.array([1, 1]), np.array([1, 1]))]

    assert in_possible_outcomes(c.genome, possible_outcomes) and \
           in_possible_outcomes(d.genome, possible_outcomes)

    # Now for genomes of different lengths
    # TODO I need to *carefully* review the possible crossover possibilities
    possible_outcomes = [[],
                         [np.array([0, 0])],
                         [np.array([1, 1])],
                         [np.array([2, 2])],
                         [np.array([0, 0]), np.array([1, 1])],
                         [np.array([1, 1]), np.array([2, 2])],
                         [np.array([2, 2]), np.array([1, 1])],
                         [np.array([2, 2]), np.array([0, 0]), np.array([1, 1])],
                         [np.array([0, 0]), np.array([2, 2]), np.array([1, 1])],
                         [np.array([0, 0]), np.array([1, 1]), np.array([2, 2])],
                         ]

    for _ in range(20):
        a = Individual([np.array([0, 0]), np.array([1, 1])])
        b = Individual([[2, 2]])

        result = NAryCrossover(num_points=1)(iter([a, b]))
        c = next(result)
        d = next(result)

        assert in_possible_outcomes(c.genome, possible_outcomes)
        assert in_possible_outcomes(d.genome, possible_outcomes)
