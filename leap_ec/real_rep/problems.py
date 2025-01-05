#!/usr/bin/env python3
""" This module contains a variety of classic real-valued optimization
problems that frequently occur in research benchmarks.

It also contains helpers for translating, rotating, and visualizing them.
"""
import warnings

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import random

from leap_ec.problem import FitnessOffsetProblem, ScalarProblem
from leap_ec import Individual


##############################
# Class SpheroidProblem
##############################
class SpheroidProblem(ScalarProblem):
    """ Classic paraboloid function, known as the "sphere" or "spheroid"
    problem, because its equal-fitness contours form (hyper)spheres in n > 2.

    .. math::

       f(\\vec{x}) = \\sum_{i}^n x_i^2

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import SpheroidProblem, plot_2d_problem
       bounds = SpheroidProblem.bounds  # Contains traditional bounds
       plot_2d_problem(SpheroidProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds for a spheroid functions solution."""
    bounds = (-5.12, 5.12)

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, maximize=False):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [0.5, 0.8, 1.5]
        >>> SpheroidProblem().evaluate(phenome)
        3.14

        :param phenome: real-valued vector to be evaluated
        :return: it's fitness, `sum(phenome**2)`
        """
        if isinstance(phenome, np.ndarray):
            return np.sum(phenome ** 2)
        return sum([x ** 2 for x in phenome])

    def worse_than(self, first_fitness, second_fitness):
        """
        We minimize by default:

        >>> s = SpheroidProblem()
        >>> s.worse_than(100, 10)
        True

        >>> s = SpheroidProblem(maximize=True)
        >>> s.worse_than(100, 10)
        False
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(SpheroidProblem())
        'SpheroidProblem'
        """
        return SpheroidProblem.__name__


##############################
# Class RastriginProblem
##############################
class RastriginProblem(ScalarProblem):
    """ The classic Rastrigin problem.  The Rastrigin provides a real-valued
    fitness landscape with a quadratic global structure (like the
    :class:`~leap.SpheroidProblem`), plus a sinusoidal local
    structure with many local optima.

    .. math::

       f(\\vec{x}) = An + \\sum_{i=1}^n x_i^2 - A\\cos(2\\pi x_i)

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import RastriginProblem, plot_2d_problem
       bounds = RastriginProblem.bounds  # Contains traditional bounds
       plot_2d_problem(RastriginProblem(), xlim=bounds, ylim=bounds, granularity=0.025)


    """

    """ Standard bounds."""
    bounds = (-5.12, 5.12)

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, a=1.0, maximize=False):
        super().__init__(maximize)
        self.a = a

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [1.0/12, 0]
        >>> RastriginProblem().evaluate(phenome) # doctest: +ELLIPSIS
        0.1409190406...

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness
        """
        if isinstance(phenome, np.ndarray):
            return self.a * len(phenome) + \
                np.sum(phenome ** 2 - self.a * np.cos(2 * np.pi * phenome))
        return self.a * \
            len(phenome) + sum([x ** 2 - self.a *
                                np.cos(2 * np.pi * x) for x in phenome])

    def worse_than(self, first_fitness, second_fitness):
        """
        We minimize by default:

        >>> s = RastriginProblem()
        >>> s.worse_than(100, 10)
        True

        >>> s = RastriginProblem(maximize=True)
        >>> s.worse_than(100, 10)
        False
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(RastriginProblem())
        'RastriginProblem'
        """
        return RastriginProblem.__name__


##############################
# Class RosenbrockProblem
##############################
class RosenbrockProblem(ScalarProblem):
    """ The classic RosenbrockProblem problem, a.k.a. the "banana" or
    "valley" function.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{d-1} \\left[ 100 (x_{i + 1} - x_i^2)^2 + (x_i - 1)^2\\right]

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import RosenbrockProblem, plot_2d_problem
       bounds = RosenbrockProblem.bounds  # Contains traditional bounds
       plot_2d_problem(RosenbrockProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-2.048, 2.048)

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, maximize=False):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> phenome = [0.5, -0.2, 0.1]
        >>> RosenbrockProblem().evaluate(phenome)
        22.3

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness
        """
        if isinstance(phenome, np.ndarray):
            x_p = phenome[1:]
            x = phenome[:-1]
            return np.sum(100 * (x_p - x ** 2) ** 2 + (x - 1) ** 2)

        sum = 0
        for i, x in enumerate(phenome[0:-1]):
            x_p = phenome[i + 1]
            sum += 100 * (x_p - x ** 2) ** 2 + (x - 1) ** 2
        return sum

    def worse_than(self, first_fitness, second_fitness):
        """
        We minimize by default:

        >>> s = RosenbrockProblem()
        >>> s.worse_than(100, 10)
        True

        >>> s = RosenbrockProblem(maximize=True)
        >>> s.worse_than(100, 10)
        False
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(RosenbrockProblem())
        'RosenbrockProblem'
        """
        return RosenbrockProblem.__name__


##############################
# Class StepProblem
##############################
class StepProblem(ScalarProblem):
    """ The classic 'step' function—a function with a linear global
    structure, but with stair-like plateaus at the local level.

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} \\lfloor x_i \\rfloor

    where :math:`\\lfloor x \\rfloor` denotes the floor function.

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import StepProblem, plot_2d_problem
       bounds = StepProblem.bounds  # Contains traditional bounds
       plot_2d_problem(StepProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-5.12, 5.12)

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, maximize=True):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome:

        >>> import numpy as np
        >>> phenome = np.array([3.5, -3.8, 5.0])
        >>> StepProblem().evaluate(phenome)
        4.0

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness
        """
        return np.sum(np.floor(phenome))

    def worse_than(self, first_fitness, second_fitness):
        """
        We maximize by default:

        >>> s = StepProblem()
        >>> s.worse_than(100, 10)
        False

        >>> s = StepProblem(maximize=False)
        >>> s.worse_than(100, 10)
        True
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(StepProblem())
        'StepProblem'
        """
        return StepProblem.__name__


##############################
# Class NoisyQuarticProblem
##############################
class NoisyQuarticProblem(ScalarProblem):
    """ The classic 'quadratic quartic' function with Gaussian noise:

    .. math::

       f(\\mathbf{x}) = \\sum_{i=1}^{n} i x_i^4 + \\texttt{gauss}(0, 1)

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import NoisyQuarticProblem, plot_2d_problem
       bounds = NoisyQuarticProblem.bounds  # Contains traditional bounds
       plot_2d_problem(NoisyQuarticProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """ Standard bounds."""
    bounds = (-1.28, 1.28)

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, maximize=False):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome (the output varies, since the function has noise):

        >>> phenome = [3.5, -3.8, 5.0]
        >>> r = NoisyQuarticProblem().evaluate(phenome)
        >>> print(f'Result: {r}')
        Result: ...

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness
        """
        indices = np.arange(len(phenome)) + 1
        noise = np.random.normal(0, 1, len(phenome))
        return np.dot(indices, np.power(phenome, 4)) + np.sum(noise)

    def worse_than(self, first_fitness, second_fitness):
        """
        We minimize by default:

        >>> s = NoisyQuarticProblem()
        >>> s.worse_than(100, 10)
        True

        >>> s = NoisyQuarticProblem(maximize=True)
        >>> s.worse_than(100, 10)
        False
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(NoisyQuarticProblem())
        'NoisyQuarticProblem'
        """
        return NoisyQuarticProblem.__name__


##############################
# Class ShekelProblem
##############################
class ShekelProblem(ScalarProblem):
    """ The classic 'Shekel's foxholes' function.

    .. math::

       f(\\mathbf{x}) = \\frac{1}{\\frac{1}{K} + \\sum_{j=1}^{25} \\frac{1}{f_j(\\mathbf{x})}}


    where

    .. math::

       f_j(\\mathbf{x}) = c_j + \\sum_{i=1}^2 (x_i - a_{ij})^6

    and the points :math:`\\left\\{ (a_{1j}, a_{2j})\\right\\}_{j=1}^{25}` define the functions various optima, and are
    given by the following hardcoded matrix:

    .. math::

       \\left[a_{ij}\\right] = \\left[ \\begin{array}{lllllllllll}
                                        -32 & -16 & 0 & 16 & 32 & -32 & -16 & \\cdots & 0 & 16 & 32 \\\\
                                        -32 & -32 & -32 & -32 & -32 & -16 & -16 & \\cdots & 32 & 32 & 32
                                       \\end{array} \\right].

    :param int k: the value of :math:`K` in the fitness function.
    :param [int] c: list of values for the function's :math:`c_j` parameters.  Each `c[j]` approximately corresponds to
        the depth of the jth foxhole.
    :param maximize: the function is maximized if `True`, else minimized.
    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import ShekelProblem, plot_2d_problem
       bounds = ShekelProblem.bounds  # Contains traditional bounds
       plot_2d_problem(ShekelProblem(), xlim=bounds, ylim=bounds, granularity=0.9)

    """

    """ Standard bounds."""
    bounds = (-65.536, 65.536)

    points = np.array([[-32, -16, 0, 16, 32] * 5,
                       [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5])

    # TODO See if we get an error if we try to add a constructor that doesn't
    # set maximize

    def __init__(self, k=500, c=np.arange(1, 26), maximize=False):
        super().__init__(maximize)
        self.k = k
        self.c = c

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued list phenome (the output varies, since the function has noise).

        :param phenome: real-valued to be evaluated
        :returns: its fitness
        """
        assert (len(phenome) == 2)

        def f(j):
            return self.c[j] + (phenome[0] - self.points[0][j]
                                ) ** 6 + (phenome[1] - self.points[1][j]) ** 6

        return 1 / (1 / self.k + np.sum([1 / f(j) for j in range(25)]))

    def worse_than(self, first_fitness, second_fitness):
        """
        We minimize by default:

        >>> s = ShekelProblem()
        >>> s.worse_than(100, 10)
        True

        >>> s = ShekelProblem(maximize=True)
        >>> s.worse_than(100, 10)
        False
        """
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        """Returns the name of the class.

        >>> str(ShekelProblem())
        'ShekelProblem'
        """
        return ShekelProblem.__name__


##############################
# Class GriewankProblem
##############################
class GriewankProblem(ScalarProblem):
    """The classic Griewank problem.  Like the
    :class:`~leap.RastriginProblem` function, the Griewank has
    a quadratic global structure with many local optima that are distrib
    in a regular pattern.

    .. math::
        f(\\mathbf{x}) = \\sum_{i=1}^d \\frac{x_i^2}{4000} -
                         \\prod_{i=1}^d \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import GriewankProblem, plot_2d_problem
       bounds = GriewankProblem.bounds  # Contains traditional bounds
       plot_2d_problem(GriewankProblem(), xlim=bounds, ylim=bounds, granularity=10)


    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import GriewankProblem, plot_2d_problem
       bounds = [-50, 50]
       plot_2d_problem(GriewankProblem(), xlim=bounds, ylim=bounds, granularity=1)
    """

    bounds = [-600, 600]

    def __init__(self, maximize=False):
        super().__init__(maximize)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness.
        """
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        t1 = np.sum(np.power(phenome, 2) / 4000)
        i_vector = np.sqrt(np.arange(1, len(phenome) + 1))
        t2 = np.prod(np.cos(phenome / i_vector))
        return t1 - t2 + 1

    def __str__(self):
        """Returns the name of the class.

        >>> str(GriewankProblem())
        'GriewankProblem'
        """
        return GriewankProblem.__name__


##############################
# Class AckleyProblem
##############################
class AckleyProblem(ScalarProblem):
    """
    .. math::
        f(\\mathbf{x}) = -a \\exp \\left( -b \\sqrt {\\frac{1}{d} \\sum_{i=1}^d x_i^2} \\right)
                         - \\exp \\left( \\frac{1}{d} \\sum_{i=1}^d \\cos(cx_i) \\right)
                         + a + \\exp(1)


    :param float a: depth parameter for the bowl-shaped macrostructure
    :param float b: exponential scale parameter for the bowl
    :param float c: wavenumber (frequency) of the cosine pattern of local optima
    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import AckleyProblem, plot_2d_problem
       import math
       problem = AckleyProblem(a=20, b=0.2, c=2*math.pi)
       bounds = AckleyProblem.bounds  # Contains traditional bounds
       plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.25)
    """

    bounds = [-32.768, 32.768]

    def __init__(self, a=20, b=0.2, c=2 * np.pi, maximize=False):
        super().__init__(maximize)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness.
        """
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        d = len(phenome)
        t1 = -self.a * np.exp(-self.b * np.sqrt(1.0 /
                                                d * np.sum(np.power(phenome, 2))))
        t2 = np.exp(1.0 / d * np.sum(np.cos(self.c * phenome)))
        return t1 - t2 + self.a + np.e

    def __str__(self):
        """Returns the name of the class.

        >>> str(AckleyProblem())
        'AckleyProblem'
        """
        return AckleyProblem.__name__


##############################
# Class WeierstrassProblem
##############################
class WeierstrassProblem(ScalarProblem):
    """The Weierstrass function is famous for being the first discovered
    example of a function that is continuous, but not differentiable.  Built
    by adding the terms of a Fourier series, it has a jagged, self-similar
    structure:

    .. math::
        f(\\mathbf{x}) = \\sum_{i=1}^d \\left[ \\sum_{k=0}^{kmax} a^k \\cos\\left( 2\\pi b^k(x_i + 0.5)\\right)
                         - n \\sum_{k=0}^{kmax} a^k \\cos(\\pi b^k) \\right]

    When used in optimization benchmarks, it's typical to carry out the
    Fourier sum to `kmax=20` terms.

    :param int kmax: number of terms to carry the Fourier sum out to
    :param float a: amplitude parameter of the cosine terms
    :param float b: wavenumber (frequency) parameter of the cosine terms
    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import WeierstrassProblem, plot_2d_problem
       bounds = WeierstrassProblem.bounds  # Contains traditional bounds
       plot_2d_problem(WeierstrassProblem(), xlim=bounds, ylim=bounds, granularity=0.01)
    """
    bounds = [-0.5, 0.5]

    def __init__(self, kmax=20, a=0.5, b=3, maximize=False):
        super().__init__(maximize)
        self.kmax = kmax
        self.a = a
        self.b = b

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness.
        """
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        result = 0
        for x in phenome:
            t1 = 0
            for k in range(self.kmax):
                t1 += self.a ** k * \
                    np.cos(2 * np.pi * (self.b ** k) * (x + 0.5))
            result += t1

        t2 = 0
        for k in range(self.kmax):
            t2 += self.a ** k * np.cos(np.pi * (self.b ** k))

        result = result - len(phenome) * t2
        return result

    def __str__(self):
        """Returns the name of the class.

        >>> str(WeierstrassProblem())
        'WeierstrassProblem'
        """
        return WeierstrassProblem.__name__


##############################
# Class LangermannProblem
##############################
class LangermannProblem(ScalarProblem):
    """A popular multi-modal test function built by summing together
    :math:`m` terms.

    .. math::
        f(\\mathbf{x}) = -\\sum_{i=1}^m c_i \\exp\\left( -\\frac{1}{\\pi} \\sum_{j=1}^d(x_j - A_{ij})^2\\right)
                         \\cos\\left(\\pi\\sum_{j=1}^d(x_j - A_{ij})^2\\right)

    Langermann's function is parameterized by a vector :math:`c_i` of length
    :math:`m` and a matrix :math:`A_{ij}` of dimension :math:`m \\times d`.
    This class uses the traditional parameterization as the default,
    with :math:`m=5` and

    .. math::
        c = (1, 2, 5, 2, 3) \\\\
        A = \\left[ \\begin{array}{ll}
                        3 & 5\\\\
                        5 & 2\\\\
                        2 & 1\\\\
                        1 & 4\\\\
                        7 & 9
                    \\end{array} \\right].

    :param int m: total number of terms in the function's sum
    :param [float] c: amplitude coefficients for each term
    :param [[float]] a: offsets points for each term
    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import LangermannProblem, plot_2d_problem
       bounds = LangermannProblem.bounds  # Contains traditional bounds
       plot_2d_problem(LangermannProblem(), xlim=bounds, ylim=bounds, granularity=0.2)
    """

    bounds = [0, 10]

    default_a = ((3, 5),
                 (5, 2),
                 (2, 1),
                 (1, 4),
                 (7, 9))

    def __init__(self, m=5, c=(1, 2, 5, 2, 3), a=default_a, maximize=False):
        super().__init__(maximize)

        self.m = m
        self.c = np.array(c)
        self.a = np.array(a)

        if not np.isscalar(m):
            raise ValueError(
                f"Got value of {m} for 'm', but must be a scalar.")
        if len(self.c.shape) != 1:
            raise ValueError(f"Got a value of shape {self.c.shape} for 'c', "
                             f"but it must be one-dimensional with length {m}.")
        if len(self.a.shape) != 2 or self.a.shape[0] != m:
            raise ValueError(
                f"Got a value of shape {self.a.shape} for 'a', but must be a {m}xd matrix.")

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness.
        """
        assert (phenome is not None)
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        if len(phenome) != self.a.shape[1]:
            raise ValueError(
                f"Received an {len(phenome)}-dimensional phenome, but this is a {self.a.shape[1]}-dimensional Langerman function.")
        result = 0
        for i in range(self.m):
            result -= self.c[i] * np.exp(
                -1.0 / np.pi * np.sum((phenome - self.a[i]) ** 2)) \
                      * np.cos(np.pi * np.sum((phenome - self.a[i]) ** 2))
        return result

    def __str__(self):
        """Returns the name of the class.

        >>> str(LangermannProblem())
        'LangermannProblem'
        """
        return LangermannProblem.__name__


##############################
# Class LunacekProblem
##############################
class LunacekProblem(ScalarProblem):
    """
    Lunacek's function is also know as the "double Rastrigin" or
    "bi-Rastrigin" problem, because it overlays a
    :class:`~leap.RastriginProblem`-style cosine function
    across a *pair* of spheroid functions.

    This function was designed to model the double-funnel macrostructure that
    occurs in some difficult cases of the Lennard-Jones function (a famous
    function from molecular dynamics).

    .. math::
        f(\\mathbf{x}) = \\min \\left( \\left\\{ \\sum_{i=1}^N(x_i - \\mu_1)^2 \\right\\},
                                       \\left\\{ d \\cdot N + s \\cdot \\sum_{i=1}^N(x_i - \\mu_2)^2\\right\\} \\right)
                         + 10\\sum_{i=1}^N(1 - \\cos(2\\pi(x_i-\\mu_i))),

    where :math:`N` is the dimensionality of the solution vector, and the
    second sphere center parameter :math:`\\mu_2` is typically given by

    .. math::
        \\mu_2 = -\\sqrt{\\frac{\\mu_1^2 - d}{s}}

    and :math:`s` is by default a function on :math:`N`:

    .. math::
        s = 1 - \\frac{1}{2\\sqrt{N + 20} - 8.2}

    These respective defaults are used for :math:`\\mu_2` and :math:`s`
    whenever `mu_2` and `s` are set to `None`.

    Because of these complicated defaults, this class requires that you
    explicitly set the dimensionality of :math:`N` of the expected input
    solutions.  A warning will be thrown if an input solution is encountered
    that doesn't match the expected dimensionality.

    :param int N: dimensionality of the anticipated input solutions
    :param float d: base fitness value of the second spheroid
    :param float mu_1: offset of the first spheroid
    :param float mu_2: offset of the second spheroid (if `None`, this will be
        calculated automatically)
    :param float s: scale parameter for the second spheroid (if `None`,
        this will be calculated automatically)
    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import LunacekProblem, plot_2d_problem
       bounds = LunacekProblem.bounds  # Contains traditional bounds
       plot_2d_problem(LunacekProblem(N=2), xlim=bounds, ylim=bounds, granularity=0.1)
    """
    bounds = (-5, 5)

    def __init__(self, N, d=1.0, mu_1=2.5, mu_2=None, s=None, maximize=False):
        super().__init__(maximize)
        self.N = N
        self.d = d
        self.mu_1 = mu_1

        # s and mu_2 are automatically inferred if not given
        self.s = s if s is not None else 1 - 1.0 / (2 * np.sqrt(N + 20) - 8.2)
        self.mu_2 = mu_2 if mu_2 is not None else - \
            np.sqrt((mu_1**2 - d) / self.s)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: real-valued vector to be evaluated
        :returns: its fitness.
        """
        assert(phenome is not None)
        if len(phenome) != self.N:
            warnings.warn(
                f"Phenome has length {len(phenome)}, but this function expected {self.N}-dimensional input.")
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        sphere1 = np.sum((phenome - self.mu_1)**2)
        sphere2 = self.d * len(phenome) + self.s * \
            np.sum((phenome - self.mu_2)**2)
        sinusoid = 10 * np.sum(1 - np.cos(2 * np.pi * (phenome - self.mu_1)))
        return min(sphere1, sphere2) + sinusoid

    def __str__(self):
        """Returns the name of the class.

        >>> str(LunacekProblem(N=2))
        'LunacekProblem'
        """
        return LunacekProblem.__name__


##############################
# Class SchwefelProblem
##############################
class SchwefelProblem(ScalarProblem):
    """
    Schwefel's function is another traditional multimodal test function whose
    local optima are distributed in a slightly irregular way, and whose
    global optimum is out at the edge of the search space (with no gently
    sloping macrostructure to guide the algorithm toward it).

    Compare this to the :class:`~leap.RastriginProblem`
    function, whose global optimum lies at the center of a quadratic bowl
    with a regular grid of local optima.

    .. math::
        f(\\mathbf{x}) = \\sum_{i=1}^d\\left(-x_i \\cdot\\sin\\left(\\sqrt{|x_i|} \\right)\\right) + \\alpha \\cdot d

    :param float alpha: fitness offset (the default value ensures that the
        global optimum has zero fitness)

    :param bool maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import SchwefelProblem, plot_2d_problem
       bounds = SchwefelProblem.bounds  # Contains traditional bounds
       plot_2d_problem(SchwefelProblem(), xlim=bounds, ylim=bounds, granularity=10)
    """
    bounds = (-512, 512)

    def __init__(self, alpha=418.982887, maximize=False):
        super().__init__(maximize)
        self.alpha = alpha

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: phenome with a real-valued phenome to be evaluated
        :returns: its fitness.
        """
        assert(phenome is not None)
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))

        return np.sum(-phenome * np.sin(np.sqrt(np.abs(phenome)))
                      ) + self.alpha * len(phenome)

    def __str__(self):
        """Returns the name of the class.

        >>> str(SchwefelProblem())
        'SchwefelProblem'
        """
        return SchwefelProblem.__name__


##############################
# Class GaussianProblem
##############################
class GaussianProblem(ScalarProblem):
    """
    A multidimensional, isotropic Gaussian function, defined by

    .. math::
       A\\exp\\left( - \\sum_i^n \\left(\\frac{x_i}{w}\\right)^2 \\right)

    :param float width: the width parameter :math:`w`
    :param float height: the height parameter :math:`A`

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import GaussianProblem, plot_2d_problem
       bounds = GaussianProblem.bounds  # Some typical bounds
       problem = GaussianProblem(width=1, height=1)
       plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.1)
    """
    bounds = (-3, 3)

    def __init__(self, width=1, height=1, maximize=True):
        assert(width > 0)
        super().__init__(maximize)
        self.width = 1
        self.height = 1

    def evaluate(self, phenome):
        assert(phenome is not None)

        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))

        return self.height * np.exp(-np.sum(np.power(phenome/self.width, 2)))

    def __str__(self):
        """Returns the name of the class.

        >>> str(GaussianProblem())
        'GaussianProblem'
        """
        return GaussianProblem.__name__


##############################
# Class CosineFamilyProblem
##############################
class CosineFamilyProblem(ScalarProblem):
    """
    A configurable multi-modal function based on combinations of cosines,
    taken from the problem generators proposed by :cite:t:`RonkkonenEtAl2008generator`.

    .. math::

       f_{\\cos}(\\mathbf{x}) = \\frac{\\sum_{i=1}^n -\\cos((G_i - 1)2 \\pi x_i)
                                - \\alpha \\cdot \\cos((G_i - 1)2 \\pi L_i x_i)}{2n}

    where :math:`G_i` and :math:`L_i` are parameters that indicate the number
    of global and local optima, respectively, in the ith dimension.

    :param float alpha: parameter that controls the depth of the local optima.
    :param [int] global_optima_counts: list of integers indicating the number
        of global optima for each dimension.
    :param [int] local_optima_counts: list of integers indicated the number
        of local optima for each dimension.
    :param maximize: the function is maximized if `True`, else minimized.

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import CosineFamilyProblem, plot_2d_problem
       problem = CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])
       bounds = CosineFamilyProblem.bounds  # Contains traditional bounds
       plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.025)

    The number of optima can be varied independently by each dimension:

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import CosineFamilyProblem, plot_2d_problem
       problem = CosineFamilyProblem(alpha=3.0, global_optima_counts=[4, 2], local_optima_counts=[2, 2])
       bounds = CosineFamilyProblem.bounds  # Contains traditional bounds
       plot_2d_problem(problem, xlim=bounds, ylim=bounds, granularity=0.025)

    """

    bounds = (0, 1)

    def __init__(self, alpha, global_optima_counts,
                 local_optima_counts, maximize=False):
        super().__init__(maximize)
        self.alpha = alpha
        self.dimensions = len(global_optima_counts)
        assert (len(local_optima_counts) == self.dimensions)
        self.global_optima_counts = np.array(global_optima_counts)
        self.local_optima_counts = np.array(local_optima_counts)

    def evaluate(self, phenome):
        """
        Computes the function value from a real-valued phenome.

        :param phenome: phenome with a real-valued phenome vector to be evaluated
        :returns: its fitness.
        """
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        term1 = -np.cos((self.global_optima_counts - 1) * 2 * np.pi * phenome)
        term2 = - self.alpha * \
            np.cos((self.global_optima_counts - 1) * 2 *
                   np.pi * self.local_optima_counts * phenome)
        value = np.sum(term1 + term2) / (2 * self.dimensions)
        return value

    def __str__(self):
        """Returns the name of the class.

        >>> str(CosineFamilyProblem(alpha=0.5, global_optima_counts=[2, 2], local_optima_counts=[2, 2]))
        'CosineFamilyProblem'
        """
        return CosineFamilyProblem.__name__



##############################
# Class QuadraticFamilyProblem
##############################
class QuadraticFamilyProblem(ScalarProblem):
    """
    A configurable multi-modal function based on combinations of
    spheroids or parabaloids.  Taken from the problem generators
    proposed by :cite:t:`RonkkonenEtAl2008generator`.

    The function is given by

    .. math::

       f(\\mathbf{x}) = \\min_{i=1,2,\\dots,q} \\left( (\\mathbf{x} - \\mathbf{p}_i)^\\top \\mathbf{B}_i^{-1} (\\mathbf{x} - \\mathbf{p}_i) + v_i \\right)

    where the :math:`\\mathbf{p}_i` gives the center of each quadratic (i.e. the location
    of each local minimum), the :math:`v_i` give their fitness values, and the
    :math:`\\mathbf{B}_i^{-1}` are symmetric matrices.

    The easiest way to create one of these problems is to use the random generator:

    .. plot::
       :include-source:

        from leap_ec.real_rep.problems import QuadraticFamilyProblem, plot_2d_problem
        from matplotlib import pyplot as plt

        problem = QuadraticFamilyProblem.generate(dimensions=2, num_basins=30)
        plot_2d_problem(problem, xlim=(-10, 10), ylim=(-10, 10), granularity=0.5)
        plt.show()

    You can also specify the problem structure directly by providing two matrices for each
    parabaloid along with an offset vector (for translation) and a scalar offset (to define the minimum
    fitness value for the basin):

    .. plot::
       :include-source:

        from leap_ec.real_rep.problems import QuadraticFamilyProblem, plot_2d_problem, random_orthonormal_matrix
        from matplotlib import pyplot as plt
        import numpy as np

        # Define the parameters for each parabaloid

        diag1 = np.diag([2, 4])     # Diagonal matrix defining the widths (eigenvalues) of the basin for each dimension
        rot1 = np.identity(2)       # Rotation matrix, in this case the identity (no rotation)
        offset1 = np.array([-1, -1])  # Offset used to translate the basin location
        fitness1 = 0                # Fitness value of the local optimum

        diag2 = np.diag([5, 1])
        rot2 = random_orthonormal_matrix(dimensions=2)  # Apply a random rotation to the second basin
        offset2 = np.array([3, 4])
        fitness2 = 100.0

        # Build the problem
        problem = QuadraticFamilyProblem(
            diagonal_matrices = [ diag1, diag2 ],
            rotation_matrices = [ rot1, rot2 ],
            offset_vectors = [ offset1, offset2 ],
            fitness_offsets = [ fitness1, fitness2 ]
        )

        # Visualize
        plot_2d_problem(problem, xlim=(-10, 10), ylim=(-10, 10), granularity=0.5)
        plt.show()

    """
    def __init__(self, diagonal_matrices: list, rotation_matrices: list, offset_vectors: list, fitness_offsets: list, maximize=False):
        super().__init__(maximize)
        assert(diagonal_matrices is not None)
        assert(rotation_matrices is not None)
        assert(offset_vectors is not None)
        assert(fitness_offsets is not None)
        assert(len(diagonal_matrices) == len(rotation_matrices))
        assert(len(offset_vectors) == len(rotation_matrices))
        assert(len(fitness_offsets) == len(rotation_matrices))

        # Store all of the input parameters in case the user wants to see them
        self.diagonal_matrices = diagonal_matrices
        self.rotation_matrices = rotation_matrices
        self.offset_vectors = offset_vectors
        self.fitness_offsets = fitness_offsets

        parabaloids = [ ParabaloidProblem(d, r) for d, r in zip(diagonal_matrices, rotation_matrices) ]
        parabaloids = [ TranslatedProblem(p, o) for p, o in zip(parabaloids, offset_vectors) ]
        self.parabaloids = [ FitnessOffsetProblem(p, v) for p, v in zip(parabaloids, fitness_offsets) ]

    @property
    def num_basins(self):
        return len(self.diagonal_matrices)

    @property
    def dimensions(self):
        return len(self.offset_vectors[0])

    def evaluate(self, phenome):
        basin_values = [ p.evaluate(phenome) for p in self.parabaloids ]
        return np.min(basin_values)

    @classmethod
    def generate(self, dimensions: int, num_basins: int, num_global_optima: int = 1, width_bounds: tuple = (1, 5), offset_bounds: tuple = (-10, 10), fitness_offset_bounds: tuple = (10, 100)):
        """
        Convenient method to generate a QuadraticFamilyProblem by randomly sampling the matrices that define it.

        >>> problem = QuadraticFamilyProblem.generate(10, 20, num_global_optima = 2)
        >>> x = problem.evaluate(np.array([0.0, 0.5, 0.0, 0.6, 0.0, 0.7, 0.6, 0.8, 4.3, 0.2]))
        """
        assert(num_basins >= 0)
        assert(len(width_bounds) == 2)
        assert(width_bounds[1] >= width_bounds[0])
        assert(len(offset_bounds) == 2)
        assert(offset_bounds[1] >= offset_bounds[0])
        assert(len(fitness_offset_bounds) == 2)
        assert(num_global_optima <= num_basins)

        diagonal_matrices = [ np.diag(np.random.uniform(*width_bounds, dimensions)) for _ in range(num_basins) ]
        rotation_matrices = [ random_orthonormal_matrix(dimensions) for _ in range(num_basins) ]
        offset_vectors = [ np.random.uniform(*offset_bounds, dimensions) for _ in range(num_basins) ]
        fitness_offsets = np.append(np.zeros(num_global_optima), np.random.uniform(*fitness_offset_bounds, num_basins - num_global_optima))

        p = QuadraticFamilyProblem(diagonal_matrices, rotation_matrices, offset_vectors, fitness_offsets)
        p.bounds = (-2*width_bounds[1], 2*width_bounds[1])

        # Record a global optimum for convenience
        if num_global_optima > 0:
            p.global_optimum = offset_vectors[0]

        return p


##############################
# ParabaloidProblem
##############################
class ParabaloidProblem(ScalarProblem):
    """
    A generalization of the `SpheroidProblem` into parabaloids (including elliptic and hyperbolic parabaloids).

    We construct the parabaloid by combining a diagonal matrix (which defines an axis-aligned parabaloid)
    with an orthornormal rotation.  Together, these make up the eigenvalues and
    eigenbasis, respectively, of an arbitrary parabaloid:

    .. math::
       \\mathbf{A} = \\mathbf{R}^\\top \\mathbf{D} \\mathbf{R}

    We then compute fitness by interpretting :math:`A` as a quadratic form:

    .. math::
       f(x) = x^\\top \\mathbf{A} x

    When the eigenvalues are all positive, then the result is an elliptic parabaloid

    .. plot::
       :include-source:

       from leap_ec.real_rep .problems import ParabaloidProblem, plot_2d_problem
       from matplotlib import pyplot as plt
       import numpy as np

       p = ParabaloidProblem(diagonal_matrix=np.diag([1, 5]), rotation_matrix=np.identity(2))
       plot_2d_problem(p, xlim=(-10, 10), ylim=(-10, 10), granularity=0.5)
       plt.show()

    If one or more eigenvalues are negative, then a hyperbolic parabloid results,
    which has a saddle shape:

    .. plot::
       :include-source:

       from leap_ec.real_rep .problems import ParabaloidProblem, plot_2d_problem
       from matplotlib import pyplot as plt
       import numpy as np

       p = ParabaloidProblem(diagonal_matrix=np.diag([-3, 5]), rotation_matrix=np.identity(2))
       plot_2d_problem(p, xlim=(-10, 10), ylim=(-10, 10), granularity=0.5)
       plt.show()

    """
    def __init__(self, diagonal_matrix: np.ndarray, rotation_matrix: np.ndarray, maximize=False):
        super().__init__(maximize)
        assert(diagonal_matrix is not None)
        assert(rotation_matrix is not None)
        # Trick to check for diagonality: np.diagonal() extracts the diagonal, np.diag() builds a diagonal matrix from it.
        # This should equal the original matrix.
        assert(np.allclose(diagonal_matrix, np.diag(np.diagonal(diagonal_matrix)))), f"Expected a diagonal matrix, but received {diagonal_matrix}."
        R = rotation_matrix
        D = diagonal_matrix
        # Construct the matrix for the quadratic form
        self.matrix = R.T @ D @ R

    def evaluate(self, phenome):
        return phenome.T @ self.matrix @ phenome


##############################
# Class TranslatedProblem
##############################
class TranslatedProblem(ScalarProblem):
    """
    Takes an existing fitness function and translates it by applying a fixed
    offset vector.

    For example,

    .. plot::
       :include-source:

       from matplotlib import pyplot as plt
       from leap_ec.real_rep.problems import SpheroidProblem, TranslatedProblem, plot_2d_problem

       original_problem = SpheroidProblem()
       offset = [-1.0, -2.5]
       translated_problem = TranslatedProblem(original_problem, offset)

       fig = plt.figure(figsize=(12, 8))

       plt.subplot(221, projection='3d')
       bounds = SpheroidProblem.bounds  # Contains traditional bounds
       plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(222, projection='3d')
       plot_2d_problem(translated_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(223)
       plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(224)
       plot_2d_problem(translated_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)
    """
    def __init__(self, problem, offset, maximize=None):
        if maximize is None:
            maximize = problem.maximize
        super().__init__(maximize=maximize)
        assert (problem is not None)
        self.problem = problem
        self.offset = np.array(offset)
        if hasattr(problem, 'bounds'):
            self.bounds = problem.bounds

    @classmethod
    def random(cls, problem, offset_bounds, dimensions, maximize=None):
        """ Apply a random real-valued translation to a fitness function, sampled uniformly between min_offset and
        max_offset in every dimension.

        >>> from leap_ec.real_rep.problems import TranslatedProblem, RastriginProblem, plot_2d_problem

        >>> original_problem = RastriginProblem()
        >>> bounds = RastriginProblem.bounds  # Contains traditional bounds
        >>> translated_problem = TranslatedProblem.random(original_problem, bounds, 2)

        >>> plot_2d_problem(translated_problem, kind='contour', xlim=bounds, ylim=bounds)
        <matplotlib.contour...>
    
        .. plot::
           :include-source:

           from leap_ec.real_rep.problems import TranslatedProblem, RastriginProblem, plot_2d_problem

           original_problem = RastriginProblem()
           bounds = RastriginProblem.bounds  # Contains traditional bounds
           translated_problem = TranslatedProblem.random(original_problem, bounds, 2)

           plot_2d_problem(translated_problem, kind='contour', xlim=bounds, ylim=bounds)
        """
        min_offset, max_offset = offset_bounds
        offset = np.random.uniform(min_offset, max_offset, dimensions)
        return cls(problem, offset, maximize=maximize)

    def evaluate(self, phenome):
        """
        Evaluate the fitness of a point after translating the fitness function.

        Translation can be used in higher than two dimensions:

        >>> import numpy as np
        >>> offset = [-1.0, -1.0, 1.0, 1.0, -5.0]
        >>> t_sphere = TranslatedProblem(SpheroidProblem(), offset)
        >>> genome = np.array([0.5, 2.0, 3.0, 8.5, -0.6])
        >>> t_sphere.evaluate(genome)
        90.86
        """
        assert (len(phenome) == len(self.offset)), \
            f"Tried to evalute a {len(phenome)}-D genome in a " \
            f"{len(self.offset)}-D fitness function. "
        # Substract the offset so that we are moving the origin *to* the offset.
        # This way we can think of it as offsetting the fitness function,
        # rather than the input points.
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        new_phenome = phenome - self.offset
        return self.problem.evaluate(new_phenome)

    def __str__(self):
        """Returns the name of this class, followed by the `__str__ of the wrapped class
        in parentheses.

        >>> str(TranslatedProblem(problem=SpheroidProblem(), offset=[5, 5, 5]))
        'TranslatedProblem(SpheroidProblem)'
        """
        return f"{TranslatedProblem.__name__}({str(self.problem)})"


################################
# Class ScaledProblem
################################
class ScaledProblem(ScalarProblem):
    """ Scale the search space of a fitness function up or down."""

    def __init__(self, problem, new_bounds, maximize=None):
        if maximize is None:
            maximize = problem.maximize
        super().__init__(maximize=maximize)
        self.problem = problem
        if not hasattr(problem, 'bounds'):
            raise ValueError(f"Problem {problem} has no 'bounds' attribute.  "
                             f"The original bounds must be defined before "
                             f"we can scale them with this method.")
        self.old_bounds = problem.bounds
        self.bounds = new_bounds

    def evaluate(self, phenome):
        if not isinstance(phenome, np.ndarray):
            raise ValueError(("Expected phenome to be a numpy array. "
                              f"Got {type(phenome)}."))
        transformed_phenome = self.old_bounds[0] + (
                    phenome - self.bounds[0]) / (
                                          self.bounds[1] - self.bounds[0]) \
                              * (self.old_bounds[1] - self.old_bounds[0])
        assert (len(transformed_phenome) == len(phenome))
        return self.problem.evaluate(transformed_phenome)

    def __str__(self):
        """Returns the name of this class, followed by the `__str__ of the wrapped class
        in parentheses.

        >>> str(ScaledProblem(problem=SpheroidProblem(), new_bounds=[[0, 1], [0, 1]]))
        'ScaledProblem(SpheroidProblem)'
        """
        return f"{ScaledProblem.__name__}({str(self.problem)})"


################################
# Function random_orthonormal_matrix()
################################
def random_orthonormal_matrix(dimensions: int):
    """Generate a random orthornomal matrix using the Gramm-Schmidt
    process.

    Orthonormal matrices represent rotations (and flips) of a space.

    The defining property of an orthonormal matrix is that its
    transpose is its inverse:

    >>> Q = random_orthonormal_matrix(10)
    >>> np.allclose( Q.dot(Q.T), np.identity(10) )
    True

    """
    matrix = np.random.normal(size=[dimensions, dimensions])
    for i, row in enumerate(matrix):
        previous_rows = matrix[0:i, :]
        row = row - \
            sum([np.dot(row, prev) * prev for prev in previous_rows])
        matrix[i, :] = row / np.linalg.norm(row)

    # Any vector in the resulting matrix will be of unit length
    assert (
            round(
                np.linalg.norm(
                    matrix[0]),
                5) == 1.0), f"A column in the transformation matrix has a " \
                            f"norm of {np.linalg.norm(matrix[0])}, " \
                            f"but it should always be approximately 1.0. "
    # Any pair of vectors will be linearly independent
    assert (abs(round(np.dot(matrix[0], matrix[1]), 5)) ==
            0.0), f"A pair of columns in the transformation matrix has " \
                    f"dot product of {round(np.dot(matrix[0], matrix[1]),5)},"\
                    f" but it should always be approximately 0.0. "
    # The matrix's transpose will be its inverse
    assert(np.allclose(matrix.dot(matrix.T), np.identity(len(matrix)))), \
        f"Any orthornormal matrix should satisfy Q^(-1) = Q^T, but we got " \
        f"QQ^T = {matrix.dot(matrix.T)} instead of {np.identity(len(matrix))}."

    return matrix


################################
# Class MatrixTransformedProblem
################################
class MatrixTransformedProblem(ScalarProblem):
    """ Apply a linear transformation to a fitness function.

    :param matrix: an nxn matrix, where n is the genome length.

    :returns: a function that first applies -matrix to the input,
        then applies fun to the transformed input.

    For example, here we manually construct a 2x2 rotation matrix and apply
    it to the :class:`leap.RosenbrockProblem` function:

    .. plot::
       :include-source:

       from matplotlib import pyplot as plt
       from leap_ec.real_rep.problems import RosenbrockProblem, MatrixTransformedProblem, plot_2d_problem

       original_problem = RosenbrockProblem()
       theta = np.pi/2
       matrix = [[np.cos(theta), -np.sin(theta)],\
                 [np.sin(theta), np.cos(theta)]]

       transformed_problem = MatrixTransformedProblem(original_problem, matrix)

       fig = plt.figure(figsize=(12, 8))

       plt.subplot(221, projection='3d')
       bounds = RosenbrockProblem.bounds  # Contains traditional bounds
       plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(222, projection='3d')
       plot_2d_problem(transformed_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(223)
       plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

       plt.subplot(224)
       plot_2d_problem(transformed_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

    """
    def __init__(self, problem, matrix, maximize=None):
        if maximize is None:
            maximize = problem.maximize
        super().__init__(maximize=maximize)
        assert (problem is not None)
        assert (len(matrix) == len(matrix[0]))
        self.matrix = np.array(matrix)
        self.problem = problem
        if hasattr(problem, 'bounds'):
            self.bounds = problem.bounds

    @classmethod
    def random_orthonormal(cls, problem, dimensions, maximize=None):
        """Create a :class:`~leap.MatrixTransformedProblem` that performs a random rotation and/or inversion of the
        function.

        We accomplish this by generating a random orthonormal basis for R^n and plugging the resulting matrix into
        :class:`~leap.MatrixTransformedProblem`.

        The classic algorithm we use here is based on the Gramm-Schmidt process: we first generate a set of random vectors, and
        then convert them into an orthonormal basis.  This approach is described in Hansen and Ostermeier's original CMA-ES paper:

        "Completely derandomized self-adaptation in evolution strategies." *Evolutionary Computation* 9.2 (2001): 159-195.

        :param problem: the original :class:`~leap.ScalarProblem` to apply the transform to.
        :param int dimensions: the number of elements each vector should have.
        :param bool maximize: whether to maximize or minimize the resulting fitness function.  Defaults to whatever setting the
            underlying problem uses.

        .. plot::
           :include-source:

           from matplotlib import pyplot as plt
           from leap_ec.real_rep.problems import CosineFamilyProblem, MatrixTransformedProblem, plot_2d_problem

           original_problem = CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 3], local_optima_counts=[2, 3])

           transformed_problem = MatrixTransformedProblem.random_orthonormal(original_problem, 2)

           fig = plt.figure(figsize=(12, 8))

           plt.subplot(221, projection='3d')
           bounds = original_problem.bounds
           plot_2d_problem(original_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(222, projection='3d')
           plot_2d_problem(transformed_problem, xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(223)
           plot_2d_problem(original_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)

           plt.subplot(224)
           plot_2d_problem(transformed_problem, kind='contour', xlim=bounds, ylim=bounds, ax=plt.gca(), granularity=0.025)
        """
        matrix = random_orthonormal_matrix(dimensions)
        return cls(problem, matrix, maximize)

    def evaluate(self, phenome):
        """
        Evaluated the fitness of a point on the transformed fitness landscape.

        For example, consider a sphere function whose global optimum is
        situated at (0, 1):

        >>> import numpy as np
        >>> s = TranslatedProblem(SpheroidProblem(), offset=[0, 1])
        >>> round(s.evaluate(np.array([0, 1])), 5)
        0

        Now let's take a rotation matrix that transforms the space by pi/2
        radians:

        >>> import numpy as np
        >>> theta = np.pi/2
        >>> matrix = [[np.cos(theta), -np.sin(theta)],\
                      [np.sin(theta), np.cos(theta)]]
        >>> r = MatrixTransformedProblem(s, matrix)

        The rotation has moved the new global optimum to (1, 0)

        >>> round(r.evaluate(np.array([1, 0])), 5)
        0.0

        The point (0, 1) lies at a distance of sqrt(2) from the new optimum,
        and has a fitness of 2:

        >>> round(r.evaluate(np.array([0, 1])), 5)
        2.0
        """
        assert (len(phenome) == len(
            self.matrix)), f"Tried to evalute a {len(phenome)}-D genome in a " \
                           f"{len(self.matrix)}-D fitness function. "
        new_point = np.matmul(self.matrix, phenome)
        return self.problem.evaluate(new_point)

    def __str__(self):
        """Returns the name of this class, followed by the `__str__ of the wrapped class
        in parentheses.

        >>> str(MatrixTransformedProblem.random_orthonormal(problem=SpheroidProblem(), dimensions=10))
        'MatrixTransformedProblem(SpheroidProblem)'
        """
        return f"{MatrixTransformedProblem.__name__}({str(self.problem)})"


##############################
# Function plot_2d_problem
##############################
def plot_2d_problem(problem, xlim=None, ylim=None, kind='surface',
                    ax=None, granularity=None, title=None, pad=None, **kwargs):
    """
    Convenience function for plotting a :class:`~leap.problem.Problem` that
    accepts 2-D real-valued phenomes and produces a 1-D scalar fitness output.

    :param ~leap.problem.Problem fun: The :class:`~leap.problem.Problem` to
        plot.

    :param xlim: Bounds of the horizontal axes.  If None, uses `problem.bounds`.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.  If None, uses `problem.bounds`.
    :type ylim: (float, float)
    :param kind: The kind of plot to create: 'surface' or 'contour'
    :type kind: str
    :param pad: An array of extra gene values, used to fill in the hidden
        dimensions with contants while drawing fitness contours.
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param float granularity: Spacing of the grid to sample points along. If
        none is given, then the granularity will default to 1/50th of the range
        of the function's `bounds` attribute.
    :param kwargs: additional keyword arguments to pass along to plot_surface()


    The difference between this and :meth:`plot_2d_function` is that this
    takes a :class:`~leap.problem.Problem` object (instead of a raw function).

    If no axes are specified, a new figure is created for the plot:

    .. plot::
       :include-source:

       from leap_ec.real_rep.problems import CosineFamilyProblem, plot_2d_problem
       problem = CosineFamilyProblem(alpha=1.0, global_optima_counts=[2, 2], local_optima_counts=[2, 2])
       plot_2d_problem(problem, xlim=(0, 1), ylim=(0, 1), granularity=0.025);

    You can also specify axes explicitly (ex. by using `ax=plt.gca()`.  When
    plotting surfaces, you  must configure your axes to use
    `projection='3d'`.  Contour plots don't need 3D axes:

    .. plot::
       :include-source:

       from matplotlib import pyplot as plt
       from leap_ec.real_rep.problems import RastriginProblem, plot_2d_problem

       fig = plt.figure(figsize=(12, 4))
       bounds=RastriginProblem.bounds  # Contains default bounds

       plt.subplot(121, projection='3d')
       plot_2d_problem(RastriginProblem(), ax=plt.gca(), xlim=bounds, ylim=bounds)

       plt.subplot(122)
       plot_2d_problem(RastriginProblem(), ax=plt.gca(), kind='contour', xlim=bounds, ylim=bounds)

    """

    def call(phenome):
        return problem.evaluate(phenome)

    if xlim is None:
        xlim = problem.bounds
    if ylim is None:
        ylim = problem.bounds

    if granularity is None:
        if hasattr(problem, 'bounds'):
            granularity = (problem.bounds[1] - problem.bounds[0]) / 50.
        else:
            raise ValueError(f"Problem {problem} has no 'bounds' attribute, "
                             f"so we couldn't set the granularity " +
                             "automatically.  You'll need to specify the "
                             "granularity to plot the problem.")

    if kind == 'surface':
        return plot_2d_function(call, xlim, ylim, granularity, ax, title, pad, **kwargs)
    elif kind == 'contour':
        return plot_2d_contour(call, xlim, ylim, granularity, ax, title, pad, **kwargs)
    else:
        raise ValueError(f'Unrecognized plot kind: "{kind}".')


##############################
# Function plot_2d_function
##############################
def plot_2d_function(fun, xlim, ylim, granularity=0.1, ax=None, title=None, pad=None, **kwargs):
    """
    Convenience method for plotting a function that accepts 2-D real-valued
    imputs and produces a 1-D scalar output.

    :param function fun: The function to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will be created).
    :param float granularity: Spacing of the grid to sample points along.
    :param pad: An array of extra gene values, used to fill in the hidden
        dimensions with contants while drawing fitness contours.
    :param kwargs: additional keyword arguments to pass along to plot_surface() or contour()

    The difference between this and :meth:`plot_2d_problem` is that this
    takes a raw function (instead of a :class:`~leap.problem.Problem` object).

    .. plot::
       :include-source:

       import numpy as np
       from scipy import linalg

       from leap_ec.real_rep.problems import plot_2d_function

       def sinc_hd(phenome):
           r = linalg.norm(phenome)
           return np.sin(r)/r

       plot_2d_function(sinc_hd, xlim=(-10, 10), ylim=(-10, 10), granularity=0.2)
    """
    assert(len(xlim) == 2)
    assert(len(ylim) == 2)
    if pad is None:
        pad = np.array([])
    assert(isinstance(pad, np.ndarray)), f"Expected pad to be a numpy array.  Got {type(pad)}."

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    @np.vectorize
    def v_fun(x, y):
        return fun(np.append(np.array([x, y]), pad))

    x = np.arange(xlim[0], xlim[1], granularity)
    y = np.arange(ylim[0], ylim[1], granularity)
    xx, yy = np.meshgrid(x, y)

    if title:
        ax.set_title(title)

    return ax.plot_surface(xx, yy, v_fun(xx, yy), **kwargs)


##############################
# Function plot_2d_contour
##############################
def plot_2d_contour(fun, xlim, ylim, granularity, ax=None, title=None, pad=None):
    """
    Convenience method for plotting contours for a function that accepts 2-D
    real-valued inputs and produces a 1-D scalar output.

    :param function fun: The function to plot.
    :param xlim: Bounds of the horizontal axes.
    :type xlim: (float, float)
    :param ylim: Bounds of the vertical axis.
    :type ylim: (float, float)
    :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
        be created).
    :param float granularity: Spacing of the grid to sample points along.
    :param pad: An array of extra gene values, used to fill in the hidden
        dimensions with contants while drawing fitness contours.

    The difference between this and :meth:`plot_2d_problem` is that this
    takes a raw function (instead of a :class:`~leap.problem.Problem` object).

    .. plot::
       :include-source:

       import numpy as np
       from scipy import linalg

       from leap_ec.real_rep.problems import plot_2d_contour

       def sinc_hd(phenome):
           r = linalg.norm(phenome)
           return np.sin(r)/r

       plot_2d_contour(sinc_hd, xlim=(-10, 10), ylim=(-10, 10), granularity=0.2)


    """
    assert (len(xlim) == 2)
    assert (len(ylim) == 2)
    if pad is None:
        pad = np.array([])
    assert(isinstance(pad, np.ndarray)), f"Expected pad to be a numpy array.  Got {type(pad)}."

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    @np.vectorize
    def v_fun(x, y):
        return fun(np.append(np.array([x, y]), pad))

    x = np.arange(xlim[0], xlim[1], granularity)
    y = np.arange(ylim[0], ylim[1], granularity)
    xx, yy = np.meshgrid(x, y)

    if title:
        ax.set_title(title)

    return ax.contour(xx, yy, v_fun(xx, yy))
