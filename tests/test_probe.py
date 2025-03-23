"""
    Unit tests for the various pipeline probes.

    Note that this does *NOT* use python3 unittest.  pytest?
"""
import io

import pytest

import leap_ec.ops as ops
from leap_ec import data
from leap_ec.global_vars import context
from leap_ec.probe import *


##############################
# Fixtures
##############################
@pytest.fixture
def test_pop_with_attributes():
    """A fixture that returns a test population with `foo`, `bar`, and `baz` attributes assigned.

    This allows us to test probes that record information from arbitrary attributes on individuals
    (rather than, say, just their genomes and fitness).
    """
    # Set up a population
    pop = data.test_population

    # Assign distinct alues to an attribute on each individual
    attrs = [('foo', ['GREEN', 15, 'BLUE', 72.81]),
             ('bar', ['Colorless', 'green', 'ideas', 'sleep']),
             ('baz', [['a', 'b', 'c'], [1, 2, 3], [None, None, None], [0.1, 0.2, 0.3]])]
    for attr, vals in attrs:
        for (ind, val) in zip(pop, vals):
            ind.__dict__[attr] = val
    
    return pop


##############################
# Tests for AttributesCSVProbe
##############################
def test_AttributesCSVProbe_1():
    """When recording the attribute 'my_value' and leaving other arguments at their default,
    running CSVProbe on a population of individuals with just the attribute 'my_value' should
    produce the correct CSV-formatted output."""
    # Set up a population
    pop = data.test_population

    # Assign distinct values to an attribute on each individual
    attr = 'my_value'
    vals = ['GREEN', 15, 'BLUE', 72.81]
    for (ind, val) in zip(pop, vals):
        ind.__dict__[attr] = val

    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(['my_value'], stream)

    # Set the generation in the context
    context['leap']['generation'] = 10

    # Execute
    probe(pop)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,my_value\n" + \
               "10,GREEN\n" + \
               "10,15\n" + \
               "10,BLUE\n" + \
               "10,72.81\n"
    assert (result == expected)


def test_AttributesCSVProbe_2(test_pop_with_attributes):
    """When recording the attribute 'my_value' and leaving other arguments at their default,
    running CSVProbe on a population of individuals with several attributes should
    produce CSV-formatted output that only records 'my_value'."""
    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(['foo', 'bar'], stream)

    # Set the generation in the context
    context['leap']['generation'] = 10

    # Execute
    probe(test_pop_with_attributes)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,foo,bar\n" + \
               "10,GREEN,Colorless\n" + \
               "10,15,green\n" + \
               "10,BLUE,ideas\n" + \
               "10,72.81,sleep\n"
    assert (result == expected)


def test_AttributesCSVProbe_3(test_pop_with_attributes):
    """Changing the order of the attributes list changes the order of the columns."""
    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    # Passing params in reverse order from the other test above
    probe = AttributesCSVProbe(['bar', 'foo'], stream)

    # Set the generation in the context
    context['leap']['generation'] = 10

    # Execute
    probe(test_pop_with_attributes)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,bar,foo\n" + \
               "10,Colorless,GREEN\n" + \
               "10,green,15\n" + \
               "10,ideas,BLUE\n" + \
               "10,sleep,72.81\n"
    assert (result == expected)


def test_AttributesCSVProbe_4(test_pop_with_attributes):
    """Providng an attribute that contains list data should work flawlessly.."""
    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(['bar', 'baz'], stream)

    # Set the generation in the context
    context['leap']['generation'] = 10

    # Execute
    probe(test_pop_with_attributes)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,bar,baz\n" + \
               "10,Colorless,\"['a', 'b', 'c']\"\n" + \
               "10,green,\"[1, 2, 3]\"\n" + \
               "10,ideas,\"[None, None, None]\"\n" + \
               "10,sleep,\"[0.1, 0.2, 0.3]\"\n"
    assert (result == expected)


def test_AttributesCSVProbe_5(test_pop_with_attributes):
    """The `job` attribute should print as a column, even if its value is 0."""
    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(['foo', 'bar'], stream, job=0)

    # Set the generation in the context
    context['leap']['generation'] = 10

    # Execute
    probe(test_pop_with_attributes)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "job,step,foo,bar\n" + \
               "0,10,GREEN,Colorless\n" + \
               "0,10,15,green\n" + \
               "0,10,BLUE,ideas\n" + \
               "0,10,72.81,sleep\n"
    assert (result == expected)