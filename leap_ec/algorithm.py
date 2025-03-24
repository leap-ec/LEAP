"""
    Provides convenient monolithic functions that wrap a lot of common function-
    ality.

    * generational_ea() for a typical generational model
    * multi_population_ea() for invoking an EA using sub-populations
    * random_search() for a more naive strategy
"""

from leap_ec import ops, util
from toolz import pipe
from typing import Iterable

from leap_ec.global_vars import context
from leap_ec.individual import Individual


##############################
# Function generational_ea
##############################
def generational_ea(max_generations: int,
                    pop_size: int, problem, representation,
                    pipeline,
                    stop=lambda x: False,
                    init_evaluate=Individual.evaluate_population,
                    k_elites: int=1,
                    start_generation: int=0,
                    context=context):
    """
    This function provides an evolutionary algorithm with a generational
    population model.

    When called this initializes and evaluates a population of size
    `pop_size` using the `init_evaluate` function and then pipes it through
    an operator `pipeline` (i.e. a list of operators) to obtain offspring.
    Wash, rinse, repeat.

    The algorithm is provided here at the "metaheuristic" level: in order to
    apply it to a particular problem, you must parameterize it with
    implementations of its various components. You must decide the population
    size, how individuals are represented and initialized, the pipeline of
    reproductive operators, etc. A metaheuristic template of this kind can be
    used to implement genetic algorithms, genetic programming, certain
    evolution strategies, and all manner of other (novel) algorithms by
    passing in appropriate components as parameters.

    :param int max_generations: The max number of generations to run the
        algorithm for. Can pass in float('Inf') to run forever or until
        the `stop` condition is reached.
    :param int pop_size: Size of the initial population
    :param Callable[[Iterable[Individual]], bool] stop: A function that accepts
        a population and returns True if it's time to stop evolving.
    :param `Problem` problem: the Problem that should be used to evaluate
        individuals' fitness
    :param representation: How the problem is represented in individuals
    :param list pipeline: a list of operators that are applied (in order) to
        create the offspring population at each generation
    :param init_evaluate: a function used to evaluate the initial population,
        before the main pipeline is run.  The default of
        `Individual.evaluate_population` is suitable for many cases, but you
        may wish to pass a different operator in for distributed evaluation
        or other purposes.
    :param k_elites: keep k elites
    :param start_generation: index of the first generation to count from
        (defaults to 0). You might want to change this, for example, in
        experiments that involve stopping and restarting an algorithm.

    :return: the final population

    The intent behind this kind of EA interface is to allow the complete
    configuration of a basic evolutionary algorithm to be defined in a clean
    and readable way.  If you define most of the components in-line when
    passing them to the named arguments, then the complete configuration of
    an algorithmic experiment forms one concise code block.  Here's what a
    basic (mu, lambda)-style EA looks like (that is, an EA that throws away
    the parents at each generation in favor of their offspring):

    >>> from leap_ec import Individual, Representation
    >>> from leap_ec.algorithm import generational_ea, stop_at_generation
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> from leap_ec.binary_rep.ops import mutate_bitflip
    >>> import leap_ec.ops as ops
    >>> pop_size = 5
    >>> final_pop = generational_ea(max_generations=100, pop_size=pop_size,
    ...
    ...                      problem=MaxOnes(),      # Solve a MaxOnes Boolean optimization problem
    ...
    ...                      representation=Representation(
    ...                          initialize=create_binary_sequence(length=10)  # Initial genomes are random binary sequences
    ...                      ),
    ...
    ...                      # The operator pipeline
    ...                      pipeline=[
    ...                          ops.tournament_selection,                     # Select parents via tournament selection
    ...                          ops.clone,                          # Copy them (just to be safe)
    ...                          mutate_bitflip(expected_num_mutations=1),     # Basic mutation with a 1/L mutation rate
    ...                          ops.UniformCrossover(p_swap=0.4),  # Crossover with a 40% chance of swapping each gene
    ...                          ops.evaluate,                       # Evaluate fitness
    ...                          ops.pool(size=pop_size)             # Collect offspring into a new population
    ...                      ])

    The algorithm runs immediately and returns the final population:

    >>> print(*final_pop, sep='\\n') # doctest:+ELLIPSIS
    Individual<...> ...
    Individual<...> ...
    Individual<...> ...
    ...
    Individual<...> ...

    You can get the best individual by using `max` (since comparison on
    individuals is based on the `Problem` associated with them, this will
    return the best individaul even on minimization problems):

    >>> max(final_pop) # doctest:+ELLIPSIS
    Individual<...>...

    """
    # Initialize a population of pop_size individuals of the same type as
    # individual_cls
    parents = representation.create_population(pop_size, problem=problem)

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(start_generation=start_generation, context=context)

    # Evaluate initial population
    parents = init_evaluate(parents)

    while (generation_counter.generation() < max_generations) and not stop(
            parents):
        # Execute the operators to create a new offspring population
        offspring = pipe(parents, *pipeline,
                         ops.elitist_survival(parents=parents,
                                              k=k_elites))

        parents = offspring  # Replace parents with offspring
        generation_counter()  # Increment to the next generation

    return parents


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

    >>> import networkx as nx
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
    ...                                ops.migrate(topology=topology,
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


##############################
# Function random_search
##############################
def random_search(evaluations, problem, representation, pipeline,
                  context=context):
    """This function performs random search of a solution space using the
    given representation and problem.

    Random search is often used as a control in evolutionary algorithm experiments:
    if your pet algorithm can't perform better than random search, then it's a sign
    that you've barked up the wrong tree!

    This implementation also allows you to pass in an operator pipeline, which will
    be applied to each individual.  The pipeline must have the following types
    of operators:

    - a selection operator, probably cyclic_selection since there will be only
        one individual from which to choose
    - clone operator to ensure we don't overwrite the previous individual
    - a pertubation operator, likely a simple mutation pipeline operator
    - evaluate operator so we know where the new individual is in the space
    - pool(size=1) to act as a pipeline sink to pull the new individuals through

    :param evaluations: how many evaluations to perform
    :param problem: the Problem instance to use for evaluating individuals
    :param representation: the Representation describing individuals
    :param pipeline: reproductive operator pipeline
    :param context: optional context for storing state as algorithm progresses
    :returns: the series of individuals that describe a random walk

    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> from leap_ec.binary_rep.ops import mutate_bitflip
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.representation import Representation
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.ops import evaluate, clone, cyclic_selection, pool
    >>> result = random_search(evaluations=5,
    ...                    problem=MaxOnes(),      # Solve a MaxOnes Boolean optimization problem
    ...
    ...                    representation=Representation(
    ...                        individual_cls=Individual,     # Use the standard Individual as the prototype for the population
    ...                        decoder=IdentityDecoder(),     # Genotype and phenotype are the same for this task
    ...                        initialize=create_binary_sequence(length=3)  # Initial genomes are random binary sequences
    ...                    ),
    ...                     pipeline=[cyclic_selection,
    ...                               clone,
    ...                               mutate_bitflip(expected_num_mutations=3),
    ...                               evaluate,
    ...                               pool(size=1)])
    >>> assert(len(result) == 5)

    The algorithm outputs a list containing all the generated individuals.
    """
    # Use the representation to sample a new individual to start us off
    individual = representation.create_individual(problem=problem)

    individual.evaluate()  # Figure out where they are in solution space

    trajectory = [individual]  # start with this guy as step 0 in random walk

    # Set up an evaluation counter that records the current generation to
    # context; start at 1 to account for individual we already created.
    evaluation_counter = util.inc_generation(start_generation=1)

    while evaluation_counter.generation() < evaluations:
        # Apply the provided reproductive operators to create a new individual.
        # This is also an opportunity for probes embedded in the pipeline to
        # report on pipeline behavior; e.g., how multiple pertubation operators
        # change the new individual as it progresses through the pipeline.
        individual = pipe([individual], *pipeline)

        trajectory.extend(individual)

        # Because the pipeline returned a list of one individual, we need to
        # unpack it because the pipe() is going to expect `individual` to not
        # be in a list.
        individual = individual[0]

        evaluation_counter()  # Increment to the next evaluation

    # Return the list of all individuals that were created
    return trajectory


##############################
# Function stop_at_generation()
##############################
def stop_at_generation(max_generation: int, context=context):
    """A stopping criterion function that checks the 'generation' count in the `context`
    object and returns True iff it is >= `max_generation`.

    The resulting function takes a `population` argument, which is ignored.

    For example:

    >>> from leap_ec import context
    >>> stop = stop_at_generation(100)

    If we set the generation field in the context object (this value will typically be
    updated by the algorithm as it runs) like so:

    >>> context['leap']['generation'] = 15

    Then we don't stop yet:

    >>> stop(population=[])
    False

    We do stop at the 100th generation:

    >>> context['leap']['generation'] = 100
    >>> stop([])
    True
    """
    assert (max_generation >= 0)
    assert (context is not None)

    def stop(population):
        return not (context['leap']['generation'] < max_generation)

    return stop
