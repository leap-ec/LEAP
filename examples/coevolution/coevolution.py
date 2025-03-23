"""
A demonstration of cooperative coevolutionary algorithm applied
to a max-ones problem.

Usage:
  coevolution.py [options]
  coevolution.py -h | --help

Options:
  -h --help             Show this screen.
  -g --generations=<n>  Generations to run for. [default: 2000]
  --pop-size=<n>        Number of individuals in each subpopulation. [default: 20]
"""
import os

from docopt import docopt

from leap_ec import Representation, test_env_var
from leap_ec import ops
from leap_ec.algorithm import multi_population_ea
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip


##############################
# Function get_representation()
##############################
def get_representation(length: int):
    """Return a representation that creates
    binary sequences of the given length."""
    assert(length > 0)
    return Representation(
        initialize=create_binary_sequence(length)
    )


##############################
# Entry point
##############################
if __name__ == '__main__':
    arguments = docopt(__doc__)

    # CLI parameters
    pop_size = int(arguments['--pop-size'])
    generations = int(arguments['--generations'])

    # Fixed parameters
    num_populations = 4
    genes_per_subpopulation = [3, 4, 5, 8]  # The number of genes in each subpopulation's individuals

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2

    # Initialize a representation for each subpopulation
    representations = [get_representation(l) for l in genes_per_subpopulation]

    with open('./coop_stats.csv', 'w') as log_stream:
        ea = multi_population_ea(max_generations=generations, pop_size=pop_size,
                                 num_populations=num_populations,

                                 # Fitness function
                                 problem=MaxOnes(),

                                 # Assign a poor initial fitness to individuals
                                 init_evaluate=ops.const_evaluate(value=-100),

                                 # Passing a list of representations causes
                                 # different ones to be used for different subpops
                                 representation=representations,

                                 # Operator pipeline
                                 shared_pipeline=[
                                     ops.tournament_selection,
                                     ops.clone,
                                     mutate_bitflip(expected_num_mutations=1),
                                     ops.CooperativeEvaluate(
                                         num_trials=3,
                                         collaborator_selector=ops.random_selection,
                                         log_stream=log_stream),
                                     ops.pool(size=pop_size)
                                 ])

        # Dump out the sub-populations
        for i, pop in enumerate(ea):
            print(f'Population {i}:')
            for ind in pop:
                print(ind)
