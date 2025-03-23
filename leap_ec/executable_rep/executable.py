"""This module provides executable object representations.  An `Executable` in
LEAP represents problem solutions as functions, agent controllers, etc.

A LEAP `Executable` is a kind of phenotype, typically constructed when we use a
:class:`~leap_ec.core.Decoder` to convert a 
genotypic representation of the object into an executable phenotype.

Executable are also just callable functors, so you can use them in your code 
like any other function.
"""

import abc
from enum import Enum
import time

import numpy as np

from leap_ec.decoder import Decoder


##############################
# Abstract Class Executable
##############################
class Executable(abc.ABC):
    @abc.abstractmethod
    def __call__(self, input):
        pass


##############################
# Class RandomExecutable
##############################
class RandomExecutable(Executable):
    """
    A trivial `Executable` phenotype that samples a random value from its 
    output space.

    :param input_space: space of possible inputs (ignored)
    :param output_space: the space of possible actions to sample from,
        satisfying the `Space` interface used by OpenAI Gym
    """

    def __init__(self, input_space, output_space):
        assert (output_space is not None)
        assert (hasattr(output_space, 'sample'))
        self.input_space = input_space
        self.output_space = output_space

    def __call__(self, input_):
        """
        Return a random output.

        :param input: ignored
        :return: a randomly selection action from the output space

        For example, if we use a space from OpenAI Gym that defines a 2-D box of continuous values:

        >>> from gymnasium import spaces
        >>> import numpy as np
        >>> output_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float32)

        Then this method will sample a random 2-D point in that box:

        >>> b = RandomExecutable(None, output_space)
        >>> b('whatever')
        array([..., ...], dtype=float32)
        """
        return self.output_space.sample()


##############################
# Class KeyboardExecutable
##############################
class KeyboardExecutable(Executable):
    """
    A non-autonomous `Executable` phenotype that allows users to control an 
    agent via the keyboard.

    :param input_space: space of possible inputs (ignored)
    :param output_space: the space of possible actions to sample from,
        satisfying the `Space` interface used by OpenAI Gym
    :param keymap: `dict` mapping keys to elements of the output space
    """

    def __init__(self, input_space, output_space,
                 keymap=lambda x: int(x - ord('0')) if x in range(ord('0'), ord(
                     '9')) else 0):
        assert (output_space is not None)
        if np.prod(output_space.shape) > 1:
            raise ValueError(
                "This environment requires an Executable with {0} ".format(
                    np.prod(output_space.shape)) +
                "outputs, but {0} can only produce 1 output at a time.".format(
                    KeyboardExecutable.__name__))
        assert (keymap is not None)
        self.output_space = output_space
        self.keymap = keymap
        self.action = 0

    def key_press(self, key, mod):
        """You'll need to assign this function to your environment's
        key_press handler. """
        self.action = self.keymap(key)

    def key_release(self, key, mod):
        """You'll need to assign this function to your environment's
        key_release handler. """
        if self.keymap(key) == self.action:
            self.action = 0

    def __call__(self, input_):
        time.sleep(0.05)
        return self.action


##############################
# Class ArgmaxExecutable
##############################
class ArgmaxExecutable(Executable):
    """Wraps another `Executable` with logic that returns the
    index of the highest output.
    
    For example, we can use this to convert the class selection 
    distribution output by a softmax layer to an integer representing
    the index of the most likely class:

    >>> executable = lambda x: [ x[0] ^ x[1], x[0] & x[1], x[0] + x[1] ]
    >>> wrapped = ArgmaxExecutable(executable)

    >>> executable([1, 1])
    [0, 1, 2]

    >>> wrapped([1, 1])
    np.int64(2)
    """
    def __init__(self, wrapped_executable):
        assert(wrapped_executable is not None)
        self.wrapped_executable = wrapped_executable

    def __call__(self, input_):
        assert(input_ is not None)
        #print(f"I: {input_}")
        value = self.wrapped_executable(input_)
        #print(f"V: {value}")
        converted = np.argmax(value)
        #print(f"C: {converted}")
        return converted

    def __getattr__(self, attr):
        """If somebody tries to access an attribute we don't have, pass the
        request on to the wrapped object."""
        # Use the regular __getattr__ for wrapped executable, to avoid an infinite loop
        if attr == 'wrapped_executable':
            return super().__getattr__(attr)
        return getattr(self.wrapped_executable, attr)


########################
# Class WrapperDecoder
########################
class WrapperDecoder(Decoder):
    """A decoder that takes an executable object output by the wrapped
    `Decoder`, and then wrapps that `Executable` with an additional decorator
    function.
    
    For example, if we have a `Decoder` that produces `Executable` objects
    whose output is governed by a softmax layer (i.e. a distribution),
    we can use this class to decorate them with an `ArgmaxExecutable` to 
    transform their output into an integer.
    """
    def __init__(self, wrapped_decoder, decorator):
        assert(wrapped_decoder is not None)
        assert(decorator is not None)
        self.wrapped_decoder = wrapped_decoder
        self.decorator = decorator

    def decode(self, genome, *args, **kwargs):
        value = self.wrapped_decoder.decode(genome)
        converted = self.decorator(value)
        return converted
