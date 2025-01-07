"""Example demonstrating the use of Cartesion genetic programming (CGP) to
evolve logic circuits to solve Boolean functions.

This application provides both an evolutionary CGP solver, and an alternative
random search algorithm, so that the two may be compared.
"""
import os
import sys

import click
from matplotlib import pyplot as plt

from leap_ec.algorithm import generational_ea, random_search
from leap_ec import ops, probe, test_env_var
from leap_ec.ops import cyclic_selection, clone, evaluate, pool
from leap_ec.representation import Representation
from leap_ec.executable_rep import cgp, neural_network, problems


##############################
# CGP components
##############################

# The CGPDecoder is the heart of our CGP representation.
# We'll set it up first because it's needed as a parameter
# to a few different components.
cgp_decoder = cgp.CGPDecoder(
                    # Primitives may either be plain lambdas or FunctionPrimitive objects.
                    #   Here we use FunctinPrimitives, because it allows additional edges to
                    #   be pruned from the graph for cleanliness.
                    primitives=[ cgp.NAND(), cgp.NotX()],
                    num_inputs = 2,
                    num_outputs = 1,
                    num_layers=50,
                    nodes_per_layer=1,
                    max_arity=2
                )


# Our fitness function will compare our circuits to the
# full truth table for XOR
xor_problem = problems.TruthTableProblem(
                    boolean_function=lambda x: [ x[0] ^ x[1] ],  # XOR
                    num_inputs = 2,
                    num_outputs = 1
                )


cgp_representation = Representation(
                        decoder=cgp_decoder,
                        # We use a sepecial initializer that obeys the CGP constraints
                        initialize=cgp_decoder.initializer()
                    )


def cgp_visual_probes(modulo):
    """Set up the graphical probes that we'll use."""
    plt.figure()
    p1 = probe.FitnessPlotProbe(modulo=modulo, ax=plt.gca())
    plt.figure()
    p2 = neural_network.GraphPhenotypeProbe(modulo=modulo, ax=plt.gca())
    return [ p1, p2 ]


##############################
# CGP algorithm
##############################
def do_cgp(gens):
    pop_size = 5

    final_pop = generational_ea(gens, pop_size,

            representation=cgp_representation,

            # Our fitness function will be to solve the XOR problem
            problem=xor_problem,

            pipeline=[
                ops.tournament_selection,
                ops.clone,
                cgp.cgp_mutate(cgp_decoder, expected_num_mutations=1),
                # The check_constraints() operator is optional, but can
                # be useful if you are, say, writing your own operators and
                # just want to verify you aren't creating invalid CGP
                # individuals:
                cgp_decoder.check_constraints,
                ops.evaluate,
                ops.pool(size=pop_size),
                probe.FitnessStatsCSVProbe(stream=sys.stdout)
            ] + cgp_visual_probes(modulo=10)
    )


##############################
# cli entry point
##############################
@click.group(invoke_without_command=True)
def cli():
    """Example of Cartesian Genetic Programming."""

    # If no command is given, just run CGP

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 100
    do_cgp(generations)


##############################
# cgp command
##############################
@cli.command(name='cgp')
@click.option('--gens', default=100)
def cgp_cmd(gens):
    """Use an evolutionary CGP approach to solve the XOR function."""
    do_cgp(100)


##############################
# random command
##############################
@cli.command('random')
@click.option('--evals', default=500)
def random(evals):
    """Use random search over a CGP representation to solve the XOR function."""
    _ = random_search(evals,
                      representation=cgp_representation,

                      # Our fitness function will be to solve the XOR problem
                      problem=xor_problem,

                      pipeline=[cyclic_selection,
                                clone,
                                cgp.cgp_mutate(cgp_decoder, probability=1.0),
                                evaluate,
                                pool(size=1),
                                probe.FitnessStatsCSVProbe(stream=sys.stdout),
                                ] + cgp_visual_probes(modulo=10)
                      )


##############################
# main
##############################
if __name__ == '__main__':
    cli()

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()

    plt.close('all')
