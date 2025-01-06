This sub-package is for LEAP support for multiobjective optimization.

We presume that fitnesses, which are normally scalars, are vectors where 
each element corresponds to a single objective.  Moreover, we presume that 
numpy arrays are used to represent these vectors.

As a side-effect some operators will add state to individuals as they pass 
through a LEAP pipeline.

`fast_nondominated_sort()` adds these attributes:

* `dominates`, a list
* `dominated_by`, integer, count of others that dominate current individual
* `rank`, integer, valid values in regular expression `[1-9]+`

`crowding_distance_calc()` adds these:

* `distance`, float, the average distance between immediate neighbors

Alternatively, we could have used a sub-class to manage this additional 
state, but we felt that could unnecessarily complicate implementations, 
particularly in situations where a user had already defined their own 
`Individual` subclass for their own specialized needs.  This approach 
provides more flexibility. 
