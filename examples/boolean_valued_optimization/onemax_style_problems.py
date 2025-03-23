"""An example of an evolutionary algorithm that makes use of LEAP's binary
representation and visualization probes while solving problems in the
broader OneMax family.
"""
import os
import sys

from matplotlib import pyplot as plt

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import MaxOnes, DeceptiveTrap, TwoMax


##############################
# main
##############################
if __name__ == '__main__':
    l = 20
    pop_size = 10 

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 100

    # Our fitness function will be the venerable MaxOnes problem or
    # one of its variations
    #problem = MaxOnes()
    #problem = TwoMax()
    problem = DeceptiveTrap()
    

    ##############################
    # Visualizations
    ##############################
    # Setting up some visualization probes in advance
    # Doing it here allow us to use subplots to arrange them nicely
    plt.figure(figsize=(18, 5))

    plt.subplot(131)
    p1 = probe.SumPhenotypePlotProbe(
            xlim=(0, l),
            ylim=(0, l),
            problem=problem,
            ax=plt.gca())

    plt.subplot(132)
    p2 = probe.FitnessPlotProbe(ax=plt.gca(), xlim=(0, generations))

    plt.subplot(133)
    p3 = probe.PopulationMetricsPlotProbe(
            metrics=[ probe.pairwise_squared_distance_metric ],
            xlim=(0, generations),
            title='Population Diversity',
            ax=plt.gca())

    plt.tight_layout()
    viz_probes = [ p1, p2, p3 ]


    ##############################
    # Run!
    ##############################
    final_pop = generational_ea(max_generations=generations,pop_size=pop_size,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 # Initialize a population of integer-vector genomes
                                 initialize=create_binary_sequence(length=l)
                             ),

                             # Operator pipeline
                             pipeline=[
                                 ops.tournament_selection(k=2),
                                 ops.clone,
                                 # Apply binomial mutation: this is a lot like
                                 # additive Gaussian mutation, but adds an integer
                                 # value to each gene
                                 mutate_bitflip(expected_num_mutations=1),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 # Collect fitness statistics to stdout
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout),
                                 *viz_probes  # Inserting the additional probes we defined above
                             ]
                        )

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
