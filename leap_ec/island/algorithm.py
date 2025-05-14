#!/usr/bin/env python3
"""
    Island model specific functions.
"""

from typing import Iterable

from toolz import pipe

import leap_ec
from leap_ec import Individual, context, ops, Representation, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.real_rep import problems, create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


##############################
# Function multi_population_ea
##############################
def multi_population_ea(max_generations, num_populations, pop_size, problem,
                        representation, shared_pipeline,
                        subpop_pipelines=None, stop=lambda x: False,
                        init_evaluate=Individual.evaluate_population,
                        context=context):
    """
    An EA that maintains multiple (interacting) subpopulations, i.e. for
    implementing island models.

    This effectively executes several EAs concurrently that share the same
    generation counter, and which share the same representation (
    :py:class:`~leap.Individual`, :py:class:`~leap.Decoder`) and
    objective function (:py:class:`~leap.problem.Problem`), and which share
    all or part of the same operator pipeline.

    :param int max_generations: The max number of generations to run the algorithm for.
        Can pass in float('Inf') to run forever or until the `stop` condition is reached.
    :param int num_populations: The number of separate populations to maintain.
    :param int pop_size: Size of each initial subpopulation
    :param int stop: A function that accepts a list of populations and
        returns True iff it's time to stop evolving.
    :param `Problem` problem: the Problem that should be used to evaluate
        individuals' fitness
    :param representation: the `Representation` that governs the creation and decoding
        of individuals.  If a list of `Representation` objects is given, then
        different representations will be used for different subpopulations; else
        the same representation will be used for all subpopulations.
    :param list shared_pipeline: a list of operators that every population
        will use to create the offspring population at each generation
    :param list subpop_pipelines: a list of population-specific operator
        lists, the ith of which will only be applied to the ith population (after
        the `shared_pipeline`).  Ignored if `None`.
    :param init_evaluate: a function used to evaluate the initial population,
        before the main pipeline is run.  The default of
        `Individual.evaluate_population` is suitable for many cases, but you
        may wish to pass a different operator in for distributed evaluation
        or other purposes.

    :return: a list of lists of each of the subpopulations.

    To turn a multi-population EA into an island model, use the
    :py:func:`leap_ec.ops.migrate` operator in the shared pipeline.  This
    operator takes a `NetworkX` graph describing the topology of connections
    between islands as input.

    For example, here's how we might define a fully connected 4-island model
    that solves a :py:class:`leap_ec.real_rep.problems.SchwefelProblem` using a
    real-vector representation:

import leap_ec.island.ops    >>> import networkx as nx
    >>> from leap_ec.algorithm import multi_population_ea
    >>> from leap_ec import ops
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> from leap_ec.real_rep import problems
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.representation import Representation
    >>> from leap_ec.real_rep.initializers import create_real_vector
    >>>
    >>> topology = nx.complete_graph(4)
    >>> nx.draw_networkx(topology, with_labels=True)
    >>> problem = problems.SchwefelProblem(maximize=False)
    ...
    >>> l = 2  # Length of the genome
    >>> pop_size = 10
    >>> pops = multi_population_ea(max_generations=10,
    ...                            num_populations=topology.number_of_nodes(),
    ...                            pop_size=pop_size,
    ...
    ...                            problem=problem,
    ...
    ...                            representation=Representation(
    ...                                individual_cls=Individual,
    ...                                decoder=IdentityDecoder(),
    ...                                initialize=create_real_vector(bounds=[problem.bounds] * l)
    ...                                ),
    ...
    ...                            shared_pipeline=[
    ...                                ops.tournament_selection,
    ...                                ops.clone,
    ...                                mutate_gaussian(std=30,
    ...                                                expected_num_mutations='isotropic',
    ...                                                bounds=problem.bounds),
    ...                                ops.evaluate,
    ...                                ops.pool(size=pop_size),
    ...                                leap_ec.island.ops.migrate(topology=topology,
    ...                                            emigrant_selector=ops.tournament_selection,
    ...                                            replacement_selector=ops.random_selection,
    ...                                            migration_gap=5)
    ...                            ])
    >>> pops # doctest:+ELLIPSIS
    [[Individual<...>(...), ..., Individual<...>(...)], ..., [Individual<...>(...), ..., Individual<...>(...)]]

    We can now run the algorithm by pulling output from its generator,
    which gives us the best individual in each population at each generation:

    While each population is executing, `multi_population_ea` writes the
    index of the current subpopulation to `context['leap'][
    'subpopulation']`.  That way shared operators (such as
    :py:func:`leap.ops.migrate`) have the option of accessing the share
    context to learn which subpopulation they are currently working with.

    TODO find a way to use Dask to parallelize populations, likely by having a
    Dask worker for each sub-poplulation.
    """

    # If we are given a single problem, create a list assigning it to each subpop
    if not isinstance(problem, Iterable):
        problem = [problem for _ in range(num_populations)]
    # If we are given a single representation, create a list assigning it to each subpop
    if not isinstance(representation, Iterable):
        representation = [representation for _ in range(num_populations)]

    assert (len(representation) == len(problem))

    # Initialize & evaluate the initial subpopulations
    pops = [r.create_population(pop_size, problem=p) for r, p in
            zip(representation, problem)]
    pops = [init_evaluate(p) for p in pops]

    # Include a reference to the populations in the context object.
    # This allows operators to see all the subpopulations.
    context['leap']['subpopulations'] = pops

    # Set up a generation counter that records the current generation to the
    # context
    generation_counter = util.inc_generation(context=context)

    while (generation_counter.generation() < max_generations) and not stop(
            pops):
        # Execute each population serially
        for i, parents in enumerate(pops):
            # Indicate the subpopulation we are currently executing in the
            # context object. This allows operators to know which
            # subpopulation they are working with.
            context['leap']['current_subpopulation'] = i
            # Execute the operators to create a new offspring population
            operators = list(shared_pipeline) + \
                        (list(subpop_pipelines[i]) if subpop_pipelines else [])
            offspring = pipe(parents, *operators)

            pops[i] = offspring  # Replace parents with offspring

        generation_counter()  # Increment to the next generation


    return pops
