"""An example of solving a reinforcement learning problem by using evolution to 
tune the weights of a neural network."""
import os
import sys

import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np

from leap_ec import Individual, Representation, test_env_var
from leap_ec import probe, ops
from leap_ec.algorithm import generational_ea
from leap_ec.executable_rep import problems, executable, neural_network
from leap_ec.int_rep.ops import individual_mutate_randint
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


##############################
# Function build_probes()
##############################
def build_probes(genomes_file):
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

    # Open a figure to plot the best-of-gen network graph to
    plt.figure()
    probes.append(neural_network.GraphPhenotypeProbe(
                        modulo=1, ax=plt.gca(),
                        weights=True, weight_multiplier=3.0))

    return probes


##############################
# Entry point
##############################
if __name__ == '__main__':
    # Parameters
    runs_per_fitness_eval = 5
    simulation_steps = 500
    pop_size = 5
    num_hidden_nodes = 4
    mutate_std = 0.05
    gui = False  # Change to true to watch the cart-pole visualization

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1000

    # Load the OpenAI Gym simulation
    environment = gym.make('CartPole-v0')

    # Representation
    num_inputs = 4
    num_actions = environment.action_space.n
    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = executable.WrapperDecoder(
                wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(
                    shape=(num_inputs, num_hidden_nodes, num_actions)
                ),
                decorator=executable.ArgmaxExecutable)

    with open('./genomes.csv', 'w') as genomes_file:

        generational_ea(max_generations=generations, pop_size=pop_size,
                        # Solve a problem that executes agents in the
                        # environment and obtains fitness from it
                        problem=problems.EnvironmentProblem(
                            runs_per_fitness_eval, simulation_steps, environment, 'reward', gui=gui),

                        representation=Representation(
                            initialize=create_real_vector(bounds=([[-1, 1]]*decoder.wrapped_decoder.length)),
                            decoder=decoder),

                        # The operator pipeline.
                        pipeline=[
                            ops.tournament_selection,
                            ops.clone,
                            mutate_gaussian(std=mutate_std, bounds=(-1, 1), expected_num_mutations=1),
                            ops.evaluate,
                            ops.pool(size=pop_size),
                            *build_probes(genomes_file)  # Inserting all the probes at the end
                        ])

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
