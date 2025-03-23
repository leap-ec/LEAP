"""An example of an evolutionary algorithm that makes use of LEAP's integer
representation.

We use a generational EA with binomial mutation of integer genes to minimize an
integer version of the Langermann function.
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_binomial
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
        generations = 100
    
    l = 2
    pop_size = 10
    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                                problem=problem,  # Fitness function

                                # Representation
                                representation=Representation(
                                    # Initialize a population of integer-vector genomes
                                    initialize=create_int_vector(
                                        bounds=[problem.bounds] * l)
                                ),

                                # Operator pipeline
                                pipeline=[
                                    ops.tournament_selection(k=2),
                                    ops.clone,
                                    # Apply binomial mutation: this is a lot like
                                    # additive Gaussian mutation, but adds an integer
                                    # value to each gene
                                    mutate_binomial(std=1.5, bounds=[problem.bounds]*l,
                                                    expected_num_mutations=1),
                                    ops.evaluate,
                                    ops.pool(size=pop_size),

                                    # Some visualization probes so we can watch what happens
                                    probe.CartesianPhenotypePlotProbe(
                                            xlim=problem.bounds,
                                            ylim=problem.bounds,
                                            contours=problem),
                                    probe.FitnessPlotProbe(),

                                    probe.PopulationMetricsPlotProbe(
                                        metrics=[ probe.pairwise_squared_distance_metric ],
                                        title='Population Diversity'),

                                    probe.FitnessStatsCSVProbe(stream=sys.stdout)
                                ]
                            )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
