"""Unit tests for real-valued reproductive operators."""
import pytest
from scipy import stats

import numpy as np

from leap_ec.individual import Individual
from leap_ec.real_rep import ops


##############################
# Tests for mutate_gaussian()
##############################
@pytest.mark.stochastic
def test_mutate_gaussian1():
    """If we apply isotropic Gaussian mutation with std=1.0 to a given genome a bunch of different times,
    the offsprings' genes should follow a Gaussian distribution around their parents' values."""
    N = 5000  # We'll sample 5,000 independent genomes

    gene0_values = []
    gene1_values = []
    op =ops.mutate_gaussian(std=1.0, expected_num_mutations='isotropic')

    for _ in range(N):
        # Set up two parents with fixed genomes, two genes each
        ind1 = Individual(np.array([0, 0.5]))
        population = iter([ind1])
        
        # Mutate
        result = op(population)
        result = next(result)  # Pulse the iterator

        gene0_values.append(result.genome[0])
        gene1_values.append(result.genome[1])

    # Use a Kolmogorov-Smirnoff test to verify that the mutations follow a
    # Gaussian distribution with zero mean and unit variance
    p_threshold = 0.01

    # Gene 0 should follow N(0, 1.0)
    _, p = stats.kstest(gene0_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should follow N(0.5, 1.0)
    gene1_centered_values = [ x - 0.5 for x in gene1_values ]
    _, p = stats.kstest(gene1_centered_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should *not* follow N(0, 1.0)
    _, p = stats.kstest(gene1_values, 'norm')
    print(p)
    assert(p <= p_threshold)


@pytest.mark.stochastic
def test_mutate_gaussian2():
    """If we apply isotropic Gaussian mutation with different std values for each gene to a 
    given genome a bunch of different times, the offsprings' genes should follow a 
    Gaussian distribution around their parents' values with the given std values."""
    N = 5000  # We'll sample 5,000 independent genomes
    std = np.array([10, 0.5])

    gene0_values = []
    gene1_values = []
    op = ops.mutate_gaussian(std=std, expected_num_mutations='isotropic')

    for _ in range(N):
        # Set up two parents with fixed genomes, two genes each
        ind1 = Individual(np.array([0.0, 100.0]))
        population = iter([ind1])
        
        # Mutate
        result = op(population)
        result = next(result)  # Pulse the iterator

        gene0_values.append(result.genome[0])
        gene1_values.append(result.genome[1])

    # Use a Kolmogorov-Smirnoff test to verify that the mutations follow a
    # Gaussian distribution with zero mean and unit variance
    p_threshold = 0.01

    # Gene 0 should follow N(0, 10)
    gene0_standardized_values = [ (x - 0)/10 for x in gene0_values ]
    _, p = stats.kstest(gene0_standardized_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should follow N(100, 0.5)
    gene1_standardized_values = [ (x - 100)/0.5 for x in gene1_values ]
    _, p = stats.kstest(gene1_standardized_values, 'norm')
    print(p)
    assert(p > p_threshold)

    # Gene 1 should *not* follow N(0, 1.0)
    _, p = stats.kstest(gene1_values, 'norm')
    print(p)
    assert(p <= p_threshold)

def test_mutate_bpm1():
    """If we apply bounded polynomial mutation to a genome,
    the values should differ from the original and stay constrained to the bounds."""
    N = 5000
    dim = 5
    def generate_from_op(op):
        res = []
        for _ in range(N):
            ind1 = Individual(np.zeros(dim))
            population = iter([ind1])
            res.append(next(op(population)).genome)
        return np.array(res)
    
    # Test differing bound modes with low eta (high variance)
    op = ops.mutate_bpm(eta=1, bounds=(-1, 1), expected_num_mutations='isotropic')
    samples = generate_from_op(op)
    assert np.all(samples != 0) and np.all(samples >= -1) and np.all(samples <= 1)

    bounds = np.stack((np.full(dim, -1), np.full(dim, 1)), axis=-1)
    op = ops.mutate_bpm(eta=1, bounds=bounds, expected_num_mutations='isotropic')
    samples = generate_from_op(op)
    assert np.all(samples != 0) and np.all(samples >= -1) and np.all(samples <= 1)

    # Test that we differ from 0 even for high eta
    op = ops.mutate_bpm(eta=1_000, bounds=(-1, 1), expected_num_mutations='isotropic')
    samples = generate_from_op(op)
    assert np.all(samples != 0)

@pytest.mark.stochastic
def test_mutate_bpm2():
    """If we apply bounded polynomial mutation to a genome,
    the variance of the children should have an inverse relationship with eta."""
    N = 5000
    etas = [1, 10, 100]
    def generate_from_op(op):
        res = []
        for _ in range(N):
            ind1 = Individual(np.zeros(1))
            population = iter([ind1])
            res.append(next(op(population)).genome)
        return np.array(res)
    
    variances = []
    for eta in etas:
        op = ops.mutate_bpm(eta=eta, bounds=(-1, 1), expected_num_mutations='isotropic')
        samples = generate_from_op(op)
        variances.append(np.var(samples))
    
    # Variances should be decreasing with the given etas
    assert np.all(np.diff(variances) < 0)

@pytest.mark.stochastic
def test_crossover_sbx1():
    """If we apply sbx the children should differ from the original
    and stay constrained to the bounds."""
    N = 5000
    dim = 5
    def generate_from_op(op):
        res1, res2 = [], []
        for _ in range(N):
            ind1 = Individual(np.zeros(dim))
            ind2 = Individual(np.ones(dim))
            children = op(iter([ind1, ind2]))
            res1.append(next(children).genome)
            res2.append(next(children).genome)
        return np.array(res1), np.array(res2)

    op = ops.SimulatedBinaryCrossover(1, p_swap=0., bounds=(-1, 2))
    samples_zero, samples_one = generate_from_op(op)
    assert np.all(samples_zero >= -1) and np.all(samples_zero <= 2) and np.all(samples_zero != 0) \
        and np.all(samples_one >= -1) and np.all(samples_one <= 2) and np.all(samples_one != 1)

    bounds = np.stack((np.full(dim, -1), np.full(dim, 2)), axis=-1)
    op = ops.SimulatedBinaryCrossover(1, p_swap=0., bounds=bounds)
    samples_zero, samples_one = generate_from_op(op)
    assert np.all(samples_zero >= -1) and np.all(samples_zero <= 2) and np.all(samples_zero != 0) \
        and np.all(samples_one >= -1) and np.all(samples_one <= 2) and np.all(samples_one != 1)

    # Test that children differ even for high eta
    op = ops.SimulatedBinaryCrossover(1_000, p_swap=0., bounds=(-1, 2))
    samples_zero, samples_one = generate_from_op(op)
    assert np.all(samples_zero != 0) and np.all(samples_one != 1)
    # Also test that the values are different at all
    assert np.all(samples_zero != samples_one)


@pytest.mark.stochastic
def test_crossover_sbx2():
    """If we apply sbx with guaranteed swap or no-swap, the children
    should be with probability > 0.5 be bounded by the opposing parent."""
    N = 5000
    def generate_from_op(op):
        res1, res2 = [], []
        for _ in range(N):
            ind1 = Individual(np.zeros(1))
            ind2 = Individual(np.ones(1))
            children = op(iter([ind1, ind2]))
            res1.append(next(children).genome)
            res2.append(next(children).genome)
        return np.array(res1), np.array(res2)
    
    # Never swap the values within sbx
    op = ops.SimulatedBinaryCrossover(1, p_swap=0., bounds=(-1, 2))
    samples_zero, samples_one = generate_from_op(op)
    assert np.mean(samples_zero <= 1) > 0.5 and np.mean(samples_one >= 0) > 0.5
    
    # Always swap the values within sbx
    op = ops.SimulatedBinaryCrossover(1, p_swap=1., bounds=(-1, 2))
    samples_one, samples_zero = generate_from_op(op)
    assert np.mean(samples_zero <= 1) > 0.5 and np.mean(samples_one >= 0) > 0.5

