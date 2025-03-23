"""
    Provides an island model example.
"""
import logging
import math
import os
import sys

from rich import print

from matplotlib import pyplot as plt
import networkx as nx

from leap_ec import Individual, Representation, context, test_env_var, \
    leap_logger_name
from leap_ec import ops, probe
from leap_ec.algorithm import multi_population_ea
from leap_ec.real_rep.problems import SchwefelProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.initializers import create_real_vector


##############################
# viz_plots function
##############################
def viz_plots(problems, modulo):
    """A convenience method that creates a figure with grid of subplots for
    visualizing the population genotypes and best-of-gen fitness for a number
    of different problems.

    :return: two lists of probe operators (for the phenotypes and fitness,
    respectively). Insert these into your algorithms to plot measurements to
    the respective subplots. """

    num_rows = min(4, len(problems))
    num_columns = math.ceil(len(problems) / num_rows)
    true_rows = int(len(problems) / num_columns)
    fig = plt.figure(figsize=(6 * num_columns, 2.5 * true_rows))
    fig.tight_layout()
    genotype_probes = []
    fitness_probes = []
    for i, p in enumerate(problems):
        plt.subplot(true_rows, num_columns * 2, 2 * i + 1)
        tp = probe.CartesianPhenotypePlotProbe(
            contours=p,
            xlim=p.bounds,
            ylim=p.bounds,
            modulo=modulo,
            ax=plt.gca())
        genotype_probes.append(tp)

        plt.subplot(true_rows, num_columns * 2, 2 * i + 2)
        fp = probe.FitnessPlotProbe(ylim=(0, 1), modulo=modulo, ax=plt.gca())
        fitness_probes.append(fp)

    plt.subplots_adjust(
        left=0.05,
        bottom=0.05,
        right=0.95,
        top=0.95,
        wspace=0.2,
        hspace=0.3)

    return genotype_probes, fitness_probes


##############################
# Entry point
##############################
if __name__ == '__main__':
    #########################
    # Parameters and Logging
    #########################
    l = 2
    pop_size = 10

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        test_mode = True
        generations = 2
    else:
        test_mode = False
        generations = 1000

    # Uncomment these lines to see logs of what genomes and fitness values are sent to your external process.
    # This is useful for debugging a simulation.
    # logging.getLogger().addHandler(logging.StreamHandler())  # Log to stderr
    # logging.getLogger(leap_logger_name).setLevel(logging.DEBUG) # Log debug messages

    #########################
    # Topology and Problem
    #########################
    # Set up up the network of connections between islands
    topology = nx.complete_graph(3)
    if not test_mode:
        # not need to draw the graph when running in test mode
        nx.draw_networkx(topology, with_labels=True)
    problem = SchwefelProblem(maximize=False)

    #########################
    # Visualization probes
    #########################
    if not test_mode:
        genotype_probes, fitness_probes = viz_plots(
            [problem] * topology.number_of_nodes(), modulo=10)
        subpop_probes = list(zip(genotype_probes, fitness_probes))
    else:
        subpop_probes = []


    def get_island(context):
        """Closure that returns a callback for retrieving the current island
        ID during logging."""
        return lambda _: context['leap']['current_subpopulation']


    #########################
    # Algorithm
    #########################
    with open('./example_migration_data.csv', 'w') as migration_file:
        ea = multi_population_ea(max_generations=generations,
                                 num_populations=topology.number_of_nodes(),
                                 pop_size=pop_size,
                                 problem=problem,  # Fitness function

                                 # Representation
                                 representation=Representation(
                                     individual_cls=Individual,
                                     initialize=create_real_vector(
                                         bounds=[problem.bounds] * l)
                                 ),

                                 # Operator pipeline
                                 shared_pipeline=[
                                     ops.tournament_selection,
                                     ops.clone,
                                     mutate_gaussian(
                                         std=30,
                                         expected_num_mutations=1,
                                         bounds=problem.bounds),
                                     ops.evaluate,
                                     ops.pool(size=pop_size),
                                     ops.migrate(topology=topology,
                                                 emigrant_selector=ops.tournament_selection,
                                                 replacement_selector=ops.random_selection,
                                                 migration_gap=50,
                                                 metric=ops.migration_metric(
                                                     stream=migration_file,
                                                     header=True
                                                 )),
                                     probe.FitnessStatsCSVProbe(
                                         stream=sys.stdout,
                                         extra_metrics={
                                             'island': get_island(context)})
                                 ],
                                 subpop_pipelines=subpop_probes)

        print(ea)

    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()

    plt.close('all')
