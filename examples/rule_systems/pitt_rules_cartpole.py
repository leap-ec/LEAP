"""An example of solving a reinforcement learning problem with a Pitt-approach rule system."""
import os
import sys

import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np

from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep import problems, rules, executable
from leap_ec.int_rep.ops import individual_mutate_randint
from leap_ec.real_rep.ops import genome_mutate_gaussian


##############################
# Function build_probes()
##############################
def build_probes(genomes_file, decoder):
    """Set up probes for writings results to file and terminal and
    displaying live metric plots."""
    assert(genomes_file is not None)

    probes = []

    # Print fitness stats to stdout
    probes.append(probe.FitnessStatsCSVProbe(stream=sys.stdout))

    # Save genome of the best individual to a file
    probes.append(probe.AttributesCSVProbe(
                  stream=genomes_file,
                  best_only =True,
                  do_fitness=True,
                  do_genome=True))

    # Open a figure to plot a fitness curve to
    plt.figure()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.title("Best-of-Generation Fitness")
    probes.append(probe.FitnessPlotProbe(
                        ylim=(0, 1), xlim=(0, 1),
                        modulo=1, ax=plt.gca()))
                    
    # Visualize the first two conditions of every rule
    plt.figure()
    plt.ylabel("Sensor 1")
    plt.xlabel("Sensor 0")
    plt.title("Rule Coverage")
    probes.append(rules.PlotPittRuleProbe(decoder, xlim=(-1, 1), ylim=(-1, 1), ax=plt.gca()))

    return probes


##############################
# Entry point
##############################
if __name__ == '__main__':
    # Parameters
    runs_per_fitness_eval = 5
    simulation_steps = 500
    pop_size = 5
    num_rules = 20
    mutate_std = 0.05
    gui = False  # Change to true to watch the cart-pole visualization

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 100

    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v0')

    # The default observation_space for CartPole has erroneously huge bounds
    # So manually define a more reasonable input space for our rules to use
    input_space = spaces.Box(-1, 1, (4,), np.float32)

    # Setup a decoder to convert genomes into rule systems
    decoder = rules.PittRulesDecoder(  
                    input_space=input_space,
                    output_space=environment.action_space,
                    priority_metric=rules.PittRulesExecutable.PriorityMetric.RULE_ORDER)

    with open('./genomes.csv', 'w') as genomes_file:

        generational_ea(max_generations=generations, pop_size=pop_size,
                        # Solve a problem that executes agents in the
                        # environment and obtains fitness from it
                        problem=problems.EnvironmentProblem(
                            runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui),

                        representation=Representation(
                            initialize=decoder.initializer(num_rules),
                            decoder=decoder),

                        # The operator pipeline.
                        pipeline=[
                            ops.tournament_selection,
                            ops.clone,
                            decoder.mutator(
                                condition_mutator=genome_mutate_gaussian(
                                                            std=mutate_std,
                                                            bounds=decoder.condition_bounds,
                                                            expected_num_mutations=1/num_rules),
                                action_mutator=individual_mutate_randint(bounds=decoder.action_bounds,
                                                                            probability=1.0)
                            ),
                            ops.evaluate,
                            ops.pool(size=pop_size),
                            *build_probes(genomes_file, decoder)  # Inserting all the probes at the end
                        ])

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
