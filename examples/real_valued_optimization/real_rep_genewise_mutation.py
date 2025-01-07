"""An example of an evolutionary algorithm that uses different mutation 
parameters for different genes.

We use a generational EA with Gaussian mutation of real-valued genes to 
minimize a function.
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.problems import LangermannProblem


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann function
    problem = LangermannProblem(maximize=False)

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 100
    
    pop_size = 10

    # Here we specify different bounds for each gene,
    # which we'll use for both initialization and hard-bounding below
    bounds = [(0, 10), (0, 5)]

    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 # Initialize a population of real-vector genomes
                                 initialize=create_real_vector(
                                     bounds=bounds)
                             ),

                             # Operator pipeline
                             pipeline=[
                                 ops.tournament_selection(k=2),
                                 ops.clone,
                                 # We pass two different std values to Gaussian mutation
                                 mutate_gaussian(std=[1.5, 0.5], bounds=bounds,
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
