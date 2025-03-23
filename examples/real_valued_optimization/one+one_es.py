"""An example of a (1+1)-style evolutionary algorithm on the Spheroid function.

We implement it as a generational_ea(), using a population size of 1 (to generate
1 offspring each generation), and with elitism of 1 (to enforce keeping the
best individual among the parent and the offspring).
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.problems import SpheroidProblem


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function.
    problem = SpheroidProblem(maximize=False)


    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 100
    
    l = 2
    pop_size = 1
    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                             k_elites=1,  # Keep the best individual
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 # Initialize a population of integer-vector genomes
                                 initialize=create_real_vector(
                                     bounds=[problem.bounds] * l)
                             ),

                             # Operator pipeline
                             pipeline=[

                                 # We put the probes first in this example, so that
                                 # we only measure parents selected by the elitism
                                 # mechanism (rather than transient offspring)
                                 probe.CartesianPhenotypePlotProbe(
                                        xlim=problem.bounds,
                                        ylim=problem.bounds,
                                        contours=problem),
                                        
                                 probe.FitnessPlotProbe(),

                                 probe.FitnessStatsCSVProbe(stream=sys.stdout),

                                 ops.cyclic_selection,
                                 ops.clone,
                                 mutate_gaussian(std=0.1, bounds=[problem.bounds]*l,
                                                 expected_num_mutations=1),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                             ]
                        )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
