#!/usr/bin/env python3
"""
    Defines the `Decoder` base class.

    `Decoders` are used to translate from genotypic to phenotypic space.  E.g.,
    binary strings may have to be decoded into corresponding integers or real
    values meaningful to a `Problem`.
"""
import abc


##############################
# Abstract Base Class Decoder
##############################
class Decoder(abc.ABC):
    """Decoders in LEAP implement how solutions to a problem are represented.
     Specifically, a :py:class:`~leap.Decoder` converts  an
     :py:class:`~leap.Individual`'s *genotype* (which is a format that
     can easily be manipulated by mutation and recombination operators) into
     a *phenotype* (which is a format that can be fed directly into a
     :py:class:`~leap.problem.Problem` object to obtain a fitness value).

    Genotypes and phenotypes can be of arbitrary type, from a simple list of
    numbers to a complex data structure. Choosing a good genotypic
    representation and genotype-to-phenotype mapping for a given problem
    domain is a critical part of evolutionary algorithm design: the
    :py:class:`~leap.Decoder` object that an algorithm uses can have a
    big impact on the effectiveness of your metaheuristics.

    In LEAP, a :py:class:`~leap.Decoder` is typically used by
    :py:class:`~leap.Individual` as an intermediate step in calculating
    its own fitness.

    For example, say that we want to use a binary-represented
    :py:class:`~leap.Individual` to solve a real-valued optimization
    problem, such as :py:class:`~leap.real_problems.SchwefelProblem`.  Here,
    the genotype is a vector of binary values, whereas the phenotype is its
    corresponding float vector.

    We can use a :py:class:`~leap.BinaryToIntDecoder` to express this
    mapping.  And when we initialize an individual, we give it all three
    pieces of this information:

    >>> from leap_ec.binary_rep.decoders import BinaryToRealDecoder
    >>> from leap_ec.individual import Individual
    >>> from leap_ec.real_rep.problems import SchwefelProblem
    >>> import numpy as np
    >>> genome = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    >>> decoder = BinaryToRealDecoder((4, -5.12, 5.12), (4, -5.12, 5.12))  # Every 4 bits map to a float on (-5.12, 5.12)
    >>> ind = Individual(genome, decoder=decoder, problem=SchwefelProblem())

    Now we can decode the individual to examine its phenotype:

    >>> ind.decode()
    array([-1.024     ,  2.38933333])

    This call is just a wrapper for the :py:class:`~leap_ec.Decoder`,
    which has the same output:

    >>> decoder.decode(genome)
    array([-1.024     ,  2.38933333])

    But now :py:class:`~leap.Individual` also has everything it needs to
    evaluate its own fitness:

    >>> ind.evaluate()
    np.float64(836.4453949...)

    Calling `evaluate()` also has the side effect of setting the fitness
    attribute:

    >>> ind.fitness
    np.float64(836.4453949...)

    """

    @abc.abstractmethod
    def decode(self, genome, *args, **kwargs):
        """
        :param genome: a genome you wish to convert
        :returns: the phenotype associated with that genome
        """
        pass


##############################
# Class IdentityDecoder
##############################
class IdentityDecoder(Decoder):
    """A decoder that maps a genome to itself.  This acts as a 'direct' or
    'phenotypic' encoding: Use this when your genotype and phenotype are the
    same thing. """

    def __init__(self):
        super().__init__()

    def decode(self, genome, *args, **kwargs):
        """:return: the input `genome`.

        For example:

        >>> import numpy as np
        >>> d = IdentityDecoder()
        >>> d.decode(np.array([0.5, 0.6, 0.7]))
        array([0.5, 0.6, 0.7])
        """
        return genome

    def __repr__(self):
        return type(self).__name__ + "()"

