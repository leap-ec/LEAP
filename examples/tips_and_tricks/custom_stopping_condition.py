"""An example of passing in a custom stopping condition to an 
evolutionary algorithm.
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.problems import SpheroidProblem


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann
    # This is defined over a real-valued space, but
    # we can also use it to evaluate integer-valued genomes.
    problem = SpheroidProblem(maximize=False)

    # The thresholds to use for our stopping condition
    fitness_threshold = 0.0001
    diversity_threshold = 5

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        # To evolve until the stopping conditions are met, we
        # specific infinite max_generations
        generations = float('inf')
    
    l = 2
    pop_size = 10
    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                                problem=problem,  # Fitness function

                                # Stopping condition: we stop when fitness or diversity drops below a threshold
                                stop=lambda pop: (max(pop).fitness < fitness_threshold)
                                                or (probe.pairwise_squared_distance_metric(pop) < diversity_threshold),

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
                                    # Apply binomial mutation: this is a lot like
                                    # additive Gaussian mutation, but adds an integer
                                    # value to each gene
                                    mutate_gaussian(std=0.2, bounds=[problem.bounds]*l,
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
