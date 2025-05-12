#!/usr/bin/env python3
"""
    Island model specific pipeline operators.
"""
import csv
import random
from typing import List

import networkx as nx
import numpy as np

from leap_ec import context, Individual
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.ops import listlist_op, logger

##############################
# Function migrate
##############################
def migrate(topology, emigrant_selector,
            replacement_selector, migration_gap,
            customs_stamp=lambda x, _: x,
            metric=None,
            context=context):
    """
    A migration operator for use in island models.

    This operator works with multi-population algorithms,
    and is thus meant to used with :py:class:`leap_ec.algorithm.multi_population_ea`.

    Specifically, it assumes that

     1. the `population` argument passed into the returned function
        is a particular sub-population that we want to process
        "emigration" out of and "immigration" into,
     2. the `context` state object contains an integer field
        `context['leap']['generation']` indicating the current
        generation count of the algorithm, and
     3. the `context` also contains a integer field
        `context['leap']['current_subpopulation']` indicating the
        index of the subpopulation that is currently being processed
        in the overall collection of subpopulations (i.e. the one
        that `population` belongs to).

    These assumptions are essentially what :py:class:`leap_ec.algorithm.multi_population_ea`
    implements.

    >>> import networkx as nx
    >>> from leap_ec import ops, context
    >>> from leap_ec.data import test_population
    >>> pop0 = test_population[:]  # Shallow copy
    >>> pop1 = test_population[:]

    >>> op = migrate(topology=nx.complete_graph(2),
    ...              emigrant_selector=ops.tournament_selection,
    ...              replacement_selector=ops.random_selection,
    ...              migration_gap=50)
    >>> context['leap']['generation'] = 0
    >>> context['leap']['current_subpopulation'] = 0
    >>> op(pop0)
    [Individual<...>(...), Individual<...>(...), Individual<...>(...), Individual<...>(...)]

    >>> context['leap']['current_subpopulation'] = 1
    >>> op(pop1)
    [Individual<...>(...), Individual<...>(...), Individual<...>(...), Individual<...>(...)]

    This operator is a stateful closure: it maintains an
    internal list of all the out-going "emigrations" that
    occurred in the previous time step, so that it can
    process them as "immigrations" in the current time step.

    :param topology: a `networkx` topology defining the connectivity among islands
    :param emigrant_selector: a selection operator for choosing individuals to
        leave an island
    :param replacement_selector: a selection operator choosing contestants that
        will be replaced by an incoming immigrant if the immigrant has higher fitness
    :param int migration_gap: migration will occur regularly after every `migration_gap`
        evolutionary steps
    :param customs_stamp: an optional function to transfrom an individual upon its
        arrival to a new island.  This can be used, for example, to change the
        individual's decoder or problem in a heterogeneous island model.
    :param metric: an optional function of the form `f(generation, immigrant_individual, contestant_indidivudal, success)`
        for recording information about migration events.
    :param context: the context object to check for EA state, such as the current
        generation number, and the ID of the subpopulation that is currently
        being processed.

    """
    num_islands = topology.number_of_nodes()

    # We wrap a closure around some persistent state to keep trag of
    # immigrants as the move between populations
    immigrants = [[] for i in range(num_islands)]

    @listlist_op
    def do_migrate(population: List) -> List:
        current_subpop = context['leap']['current_subpopulation']
        logger.debug(f"Migration operator called on subpop {current_subpop} (generation: {context['leap']['generation']})")

        generation = context['leap']['generation']

        # Immigration
        for i, imm in enumerate(immigrants[current_subpop]):
            logger.debug(f"Processing immigrant {i+1} of {len(immigrants[current_subpop])} for subpop {current_subpop}.")
            # Do island-specific transformation
            # For example, this callback might update the individuals 'problem'
            # field to point to a new fitness function for the island, and
            # re-evalute its fitness.
            imm = customs_stamp(imm, current_subpop)

            # Compete for a place in the new population
            indices = [] # List to collect the selected index
            contestant = next(replacement_selector(population, indices=indices))
            contestant_index = indices[0]

            success = (imm >= contestant)
            if success:
                # Replace the contestant with the immgrant at the same position
                population[contestant_index] = imm

            if metric:
                metric(generation, imm, contestant, success)

        immigrants[current_subpop] = []

        # Emigration
        if generation % migration_gap == 0:
            logger.debug(f"migration_gap reached: doing emigration on subpop {current_subpop}.")
            # Choose an emigrant individual
            sponsor = next(emigrant_selector(population))
            logger.debug(f"Sponsor individual selected by emigrant_selector: {sponsor}")
            # Clone it and copy fitness
            emi = sponsor.clone()
            emi.fitness = sponsor.fitness
            logger.debug(f"Emigrant individual (copy of sponsor): {emi}")
            neighbors = topology.neighbors(
                current_subpop)  # Get neighboring islands
            # Randomly select a neighboring island
            dest = random.choice(list(neighbors))
            logger.debug(f"Destination island: {dest}")
            # Add the emigrant to its immigration list
            immigrants[dest].append(emi)

        return population

    return do_migrate


##############################
# Function migration_metric
##############################
def migration_metric(stream, header: bool = True, notes: dict = None):
    """
    Returns a function that can be used to record migration events.

    The purpose of a migration metric is to record information about
    migrations that occur inside a migration operator.  Because these
    events take place inside the operator (rather than across operators),
    they cannot be recorded by a LEAP pipeline probe.

    In general, the interface for a migration metric function takes
    four parameters:

        - `generation`: the current generation
        - `immigrant_ind`: the individual that is attempting to migrate
        - `contestant_ind`: the individual that has been chosen to be replaced
        - `success`: True if the migration is successful, False otherwise

    The metric included here records the fitness of both individuals and writes
    them (along with the `generation` and `success` values) to a CSV.  You can
    write your own metric if you need to record other information (such as, say,
    genomes).

    >>> import sys
    >>> from leap_ec import Individual
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> m = migration_metric(sys.stdout,
    ...                      header=True,
    ...                      notes={'run': 0, 'description': 'Test output'}
    ... )
    run,description,generation,migrant_fitness,contestant_fitness,success

    >>> ind1 = Individual(np.array([1, 1, 1]), problem=MaxOnes())
    >>> f = ind1.evaluate()
    >>> contestant = Individual(np.array([0, 1, 1]), problem=MaxOnes())
    >>> f = contestant.evaluate()
    >>> m(0, ind1, contestant, True)
    0,Test output,0,3,2,True

    :param stream: file object to write the CSV data to
    :param bool header: a CSV header will be written if True
    :param dict notes: a dict specifying additional constant-value
        columns to include in the CSV output
    """
    notes = {} if notes is None else notes

    # Set up data collection if we're given a stream to write to
    if stream is None:
        writer = None
    else:
        fields = list(notes.keys()) + ['generation', 'migrant_fitness', 'contestant_fitness', 'success']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        if header:
            writer.writeheader()

    def measure_migration(generation, migrant, contestant, success: bool):
        """Write a row recording the given migration event."""
        if writer is not None:
            row_dict = {
                **notes,
                'generation': generation,
                'migrant_fitness': migrant.fitness,
                'contestant_fitness': contestant.fitness,
                'success': success
            }
            writer.writerow(row_dict)

    return measure_migration
