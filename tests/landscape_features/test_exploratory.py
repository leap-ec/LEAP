"""Unit tests for the exploratory package, i.e. our exploratory landscape analysis features."""
import pytest

from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
from leap_ec.landscape_features import exploratory
from leap_ec.representation import Representation
from leap_ec.real_rep import initializers, problems
from leap_ec.statistical_helpers import equals_gaussian


##############################
# Test fixtures
##############################
def spheroid_sample():
    """A uniform sample of individuals on the spheroid function."""
    DIMENSIONS = 10
    N_SAMPLES = 50*DIMENSIONS
    problem = problems.SpheroidProblem()

    representation = Representation(
        initialize=initializers.create_real_vector(bounds=[(-5.12, 5.12)]*DIMENSIONS)
    )

    initial_sample = representation.create_population(N_SAMPLES, problem)
    Individual.evaluate_population(initial_sample)

    return problem, representation, initial_sample

def spheroid_convex():
    """Constructs an ELAConvexity instance from samples on the spheroid.

    We made this a fixture because building ELAConexity requires some computation,
    so we want to reuse a single instance in different tests.
    """
    problem, representation, initial_sample = spheroid_sample()

    return exploratory.ELAConvexity(problem, representation, design_individuals=initial_sample, num_convexity_tests=100)

@pytest.fixture
def spheroid_convex_fixture():
    return spheroid_convex()


##############################
# Tests for ELAConvexity
##############################
def test_results_length(spheroid_convex_fixture):
    """We run ELAConvexity requesting 100 tests, so it should generate data for
    100 pairs of points."""
    assert(len(spheroid_convex_fixture.pairs) == 100)
    assert(len(spheroid_convex_fixture.combinations) == 100)
    assert(len(spheroid_convex_fixture.deltas) == 100)

def test_spheroid_convexity(spheroid_convex_fixture):
    """If we run ELAConvexity on the spheroid function, it should come back as 100%
    convex and 0% linear."""
    assert(spheroid_convex_fixture.convex_p() == pytest.approx(1.0))
    assert(spheroid_convex_fixture.linear_p() == pytest.approx(0.0))


def test_spheroid_deviation(spheroid_convex_fixture):
    """If we run ELAConvexity on the spheroid function, the original and absolute
    linear_deviation values should be negative of each other."""
    assert(spheroid_convex_fixture.linear_deviation() == pytest.approx(-spheroid_convex_fixture.linear_deviation_abs()))

@pytest.mark.slow
@pytest.mark.stochastic
def test_spheroid_linear_deviation():
    """If we run ELAConvexity on the spheroid function, the output should be consistent
    with what comes out of version 1.8 of the R `flacco` package.

    This expected answer for this test is based on the following R script:

    .. code::

        library(flacco)

        do_features <- function() {
        ## (1) Create some example-data
        ctrl = list(init_sample.lower = -5.12,
                    init_sample.upper = 5.12)
        X = createInitialSample(n.obs = 500, dim = 10, control=ctrl)
        #f = function(x) sum(sin(x) * x^2 + (x - 0.5)^3)
        f = function(x) sum(x^2)
        y = apply(X, 1, f)
        
        ## (2) Compute the feature object
        feat.object = createFeatureObject(X = X, y = y, fun=f)
        
        ## (3) Have a look at feat.object
        print(feat.object)
        
        ## (4) Check, which feature sets are available
        listAvailableFeatureSets()
        
        ## (5) Calculate a specific feature set, e.g. the ELA meta model
        featureSet = calculateFeatureSet(feat.object, set = "ela_conv")
        
        featureSet
        }


        x <- replicate(100, do_features()$ela_conv.lin_dev.orig)
        hist(x)
        mean(x)
        # Output: -29.18858
        sd(x)
        # Output: 0.6097511

    
    """
    reference_mean = -29.18858
    reference_std = 0.6097511
    num_reference_observations = 100
    p = 0.05

    x = [ spheroid_convex().linear_deviation() for _ in range(num_reference_observations) ]

    assert(equals_gaussian(x, reference_mean, reference_std, num_reference_observations, p=p))
