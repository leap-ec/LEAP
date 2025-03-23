Multiobjective Optimization
===========================

LEAP supports multi-objective optimization via an implementation of [NSGA-II]_. There are two ways of using this
functionality -- using a single function, `leap_ec.mulitobjective.nsga2.generalized_nsga_2` , or by assembling a bespoke NSGA-II using pipeline
operators.  We will cover both approaches here.

Using `generalized_nsga_2`
--------------------------

`leap_ec.mulitobjective.nsga2.generalized_nsga_2` is similar to other LEAP
metaheuristic functions, such as `generational_ea`.  It has arguments for specifying
the maximum number of generations, population size, stopping criteria, problem
representation, and others.

Note that by default a faster rank sorting algorithm is used [Burlacu]_, but if it is
important to use the original NSGA-II rank sorting algorithm, then that can be provided
by specifying `leap_ec.mulitobjective.ops.fast_nondominated_sort` for the `rank_func`
argument.

Example
^^^^^^^

.. code-block:: Python
    :linenos:

    from leap_ec.representation import Representation
    from leap_ec.ops import random_selection, clone, evaluate, pool
    from leap_ec.real_rep.initializers import create_real_vector
    from leap_ec.real_rep.ops import mutate_gaussian
    from leap_ec.multiobjective.nsga2 import generalized_nsga_2
    from leap_ec.multiobjective.problems import SCHProblem
    pop_size = 10
    max_generations = 5
    final_pop = generalized_nsga_2(
        max_generations=max_generations, pop_size=pop_size,

        problem=SCHProblem(),

        representation=Representation(
            initialize=create_real_vector(bounds=[(-10, 10)])
        ),

        pipeline=[
            random_selection,
            clone,
            mutate_gaussian(std=0.5, expected_num_mutations=1),
            evaluate,
            pool(size=pop_size),
        ]
    )

The above code snippet shows how to set up NSGA-II for one of the benchmark
multiobjective problems, `SCHProblem`.  We specify the maximum number of generations,
the population size, representation, and give a reproduction pipeline.  The
representation is a simple single valued gene, that we see on line 15 is initialized
in the range of [-10,10].

The reproduction pipeline given on lines 18-24 is used to create the offspring for each generation. It is spliced
into another pipeline so that the offspring created via this pipeline are then passed to the rank sorting and
crowding distance functions.  Then truncation selection based on rank and crowding distance is used to return the
final set of offspring that then become the parents for the next generation.

Creating a tailored NSGA-II
---------------------------

However, it may be desirable to have fine-grained control over the NSGA-II implementation, maybe to
more conveniently perform some necessary ancillary calculations during a run.  In that case, the
lower-level NSGA-II operators can be directly used in a full LEAP pipeline, as shown below.

Example
^^^^^^^

.. code-block:: Python
    :linenos:

    # represenations have a convenience function for creating
    # initial random population
    parents = representation.create_population(int(config.ea.pop_size),
                                               problem=problem)

    generation_counter = util.inc_generation(context=context)

    # Scatter the initial parents to dask workers for evaluation
    parents = synchronous.eval_population(parents, client=client)

    context['std'] = np.array([0.001,  # start_lr
                               0.0001, # stop_lr
                               0.0625, # rcut
                               0.0625, # rcut smth
                               0.0625, # training batch
                               0.0625, # valid. batch
                               0.0625, # scale by worker
                               0.0625, # des activ func
                               0.0625, # fitting activ func
                               ])

    try:
        while generation_counter.generation() < max_generations:
            generation_counter()  # Increment to the next generation

            offspring = pipe(parents,
                             ops.random_selection,
                             ops.clone,
                             mutate_gaussian(
                                 std=context['std'],
                                 expected_num_mutations='isotropic', # zap all genes
                                 hard_bounds=DeepMDRepresentation.bounds),
                             eval_pool(client=client, size=len(parents)),
                             rank_ordinal_sort(parents=parents),
                             crowding_distance_calc,
                             ops.truncation_selection(size=len(parents),
                                                      key=lambda x: (-x.rank,
                                                                     x.distance)),
                             )

            parents = offspring  # Make offspring new parents for next generation

            context['std'] *= .85

The above code demonstrates how to use the NSGA operators, `rank_ordinal_sort` and `crowding_distance_calc`, in a
LEAP reproductive operator pipeline to do the rank sorting and crowding distance calculation on newly formed
offspring.  The truncation selection operator uses the rank and distances that are added as attributes to individuals
as they pass through the pipeline by those operators.

Also shown is how to use Dask to perform parallel fitness evaluations.  On line 9 the initial random population
is scattered to preassigned Dask workers for evaluation.  Line 33 performs a similar operation with newly
created offspring.

And, finally, this shows how to add some ancillary computation, in this case updating a vector of
standard deviations to be used with the Gaussian mutation operator.  The vector is assigned to the LEAP
global dictionary, `context`, on line 11, and is updated every generation on line 43.  The mutation operator, itself,
is on line 29.  Although a special pipeline operator could have been made to do this same update to enable use of `generalized_nsga_2` ,
it was cleaner to separate out this update outside the pipeline.


Representing multiple fitnesses
-------------------------------

Normally a fitness is a real-valued scalar, but in the case of multiple objectives, LEAP uses a numpy
array of floats for fitnesses, with each element of the array corresponding to one objective.  Be mindful to
*not* use a python tuple or list to hold fitnesses.

Another caveat if using `DistributedIndividual` is that class will assign NaNs as fitnesses if something should go
wrong while evaluating an individual's fitness.  E.g., if optimizing a neural network architecture and exception is
thrown during model training due to a hardware failure. This poses a problem for rank sorting since sorting floating
point values with NaNs leads to undefined behavior.  In which case it's advisable to create a`DistributedIndividual`
subclass that overrides this behavior and assigns, say, MAXINT or -MAXINT (as appropriate for maximizing or
minimizing objectives) for fitnesses where there was a problem in performing the fitness evaluation.

Asynchronous steady-state multiobjective optimization
-----------------------------------------------------

LEAP also supports a distributed asynchronous-steady state version of NSGA-II.  This is useful for HPC clusters
where it is desirable to have a large number of workers evaluating individuals in parallel.  Moreover, this allows for
minimizing worker idle time in that new offspring are allocated to workers that finished evaluating their previous
individuals.  This is in contrast to the traditional synchronous version of NSGA-II where all workers must finish evaluating their
individuals before the next generation can be created; within an HPC context this would mean that some workers would
be idle while waiting for others to finish, thus wasting computational resources. As with the other distributed support
in LEAP, this functionality is implemented using Dask, and so a Dask client must be provided.

The asynchronous steady-state version of NSGA-II is implemented in :py:func:`leap_ec.multiobjective.asynchronous.steady_state_nsga_2`.
An example of use is in the `examples/distributed/multiobjective_async_distributed.ipynb` notebook.


References
----------

.. [NSGA-II] Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
            "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
            transactions on evolutionary computation 6, no. 2 (2002): 182-197.

.. [Burlacu] Bogdan Burlacu. 2022. "Rank-based Non-dominated Sorting". arXiv.
      DOI:https://doi.org/10.48550/ARXIV.2203.13654