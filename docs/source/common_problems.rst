Common Problems
===============

Here we address common problems that may arise when using LEAP.

`min()` returns the worst individual for minimization problems
--------------------------------------------------------------

`min()` and `max()` works the opposite you may expect for minimization problems
because the `<` operator has been overriden to consider fitness scalars that are
numerically less than than another to be "better".  So `min()` takes into consideration
the problem semantics not the raw number values.

E.g., for a given minimization problem:

.. code-block:: python

    min(parents).fitness
    Out[2]: 66.49057507514954
    max(parents).fitness
    Out[3]: 59.87865996360779

The above shows that the value `59.87865996360779` is "better" than
`66.49057507514954` even though _numerically_ it is less than the other value.

It was important for LEAP to override the `<` operator for `Individual`s because
it uses native sort operations to find the "best" and "worst", and so minimization vs.
maximization semantics needed to be taken into account.

Missing pipeline operator arguments
-----------------------------------

If you see an error like this:

```
TypeError: mutate_binomial() missing 1 required positional argument: 'next_individual'
```

The corresponding code may look like this:

.. code-block:: python
    :emphasize-lines: 3

    int_ops.mutate_binomial(std=[context['leap']['std0'],
                                 context['leap']['std1']],
                            hard_bounds=[(1, 127), (0, 255)],
                            probability=context['leap']['mutation']),

In this case, the API for :func:`leap_ec.int_rep.ops.mutate_binomial` had changed
such that the argument `hard_bounds` had been shortened to `bounds`.  Renaming that
argument to `bounds` fixed this instance of the problem.

In general, if you see an error like this, you should check the API documentation and
ensure that all mandatory function arguments are getting passed into the pipeline
operator.
