"""An example of an evolutionary algorithm with a basic real-vectored solution
representation, and that logs several diversity metrics from the population to
its CSV output.

We use a generational EA with Gaussian mutation of 2-D genomes to minimize
the Langermann function.
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.problems import LangermannProblem


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann
    # This is defined over a real-valued space, but
    # we can also use it to evaluate integer-valued genomes.
    problem = LangermannProblem(maximize=False)

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1000

    l = 2
    pop_size = 10
    generational_ea(max_generations=generations,pop_size=pop_size,
                    problem=problem,  # Fitness function

                    # Representation
                    representation=Representation(
                        # Initialize a population of integer-vector genomes
                        initialize=create_real_vector(
                            bounds=[problem.bounds] * l)
                    ),

                    # Operator pipeline
                    pipeline=[
                        ops.tournament_selection(k=2),
                        ops.clone,

                        # Apply Gaussian mutation
                        mutate_gaussian(std=1.5, bounds=[problem.bounds]*l,
                                        expected_num_mutations=1),
                        ops.evaluate,
                        ops.pool(size=pop_size),

                        # Some visualization probes so we can watch what happens
                        probe.CartesianPhenotypePlotProbe(
                            xlim=problem.bounds,
                            ylim=problem.bounds,
                            contours=problem),
                        probe.FitnessPlotProbe(),

                        # Collect diversity metrics along with the standard CSV columns
                        probe.FitnessStatsCSVProbe(stream=sys.stdout,
                        extra_metrics={
                            'diversity_pairwise_dist': probe.pairwise_squared_distance_metric,
                            'diversity_sum_variance': probe.sum_of_variances_metric,
                            'diversity_num_fixated': probe.num_fixated_metric
                            })
                    ]
            )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
