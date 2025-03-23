"""
    Tests for leap_ec.distrib.evaluate.
"""
import numpy as np
import pytest
from distributed import Client, LocalCluster

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.distrib.evaluate import evaluate, is_viable
from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.global_vars import context
from leap_ec.individual import Individual


def test_good_eval():
    """
        This is for testing a plain ole good individual to ensure that
        leap_ec.distrib.evaluate works for normal circumstances.
    """
    # set up a basic dask local cluster
    with Client(LocalCluster(n_workers=1, threads_per_worker=1)) as client:
        # hand craft an individual that should evaluate fine
        # Let's try evaluating a single individual
        individual = Individual(np.array([1, 1]),
                                problem=MaxOnes())

        future = client.submit(evaluate(context=context),
                               individual)

        evaluated_individual = future.result()

        assert evaluated_individual.fitness == 2


def test_broken_individual_eval():
    """
        Test an individual that intentionally throws an exception during
        evaluation, which marks that individual has non-viable.

        TODO implement this
    """
    # set up a basic dask local cluster
    with Client(LocalCluster(n_workers=1, threads_per_worker=1)) as client:
        # hand craft an individual that will intentionally fail by not
        # assigning it a Problem class
        individual = DistributedIndividual(np.array([1, 1]),
                                           problem=None)

        future = client.submit(evaluate(context=context),
                               individual)

        evaluated_individual = future.result()

        assert not is_viable(evaluated_individual)
