import networkx as nx

from leap_ec.island.ops import migrate
from leap_ec import Individual, context
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.ops import naive_cyclic_selection


##############################
# Tests for migrate()
##############################
def test_migrate1():
    """When using deterministic selection operators, we should
    see the first individual in pop0 migrate by replacing the first
    individual in pop1, then the second replace the second, etc.,
    whenever the immigrant's fitness is higher than the contestant.
    """
    # Set up two populations with known fitness values
    pop0 = [ Individual(f"A{i}", problem=MaxOnes()) for i in range(5) ]
    fitnesses0 = [ 100, 10, 100, 10, 100 ]
    pop1 = [ Individual(f"B{i}", problem=MaxOnes()) for i in range(5) ]
    fitnesses1 = [ 10, 100, 10, 100, 10 ]
    for ind0, f0, ind1, f1 in zip(pop0, fitnesses0, pop1, fitnesses1):
        ind0.fitness = f0
        ind1.fitness = f1

    # Create the operator
    op = migrate(topology=nx.complete_graph(2),
                                    emigrant_selector=naive_cyclic_selection,
                                    replacement_selector=naive_cyclic_selection,
                                    migration_gap=50)

    # Generation 0
    context['leap']['generation'] = 0

    context['leap']['current_subpopulation'] = 0
    pop0 = op(pop0)
    assert(pop1[0].genome == 'B0'), "pop1 should not yet be modified"

    context['leap']['current_subpopulation'] = 1
    pop1 = op(pop1)
    assert(pop1[0].genome == 'A0'), "The first element of pop1 should be replaced by the first element of pop0"
    assert(pop1[0].fitness == pop0[0].fitness), "The immigrant should have the same fitness as the sponsor it was copied from."


def test_migrate2():
    """If the population contains multilpe references to the same object,
    only one of them should be removed during replacement.

    We don't really expect people to use populations this way, but
    added this test to avoid any surprises.
    """
    # Set up two populations

    # pop0 has just one individual in it
    pop0 = [ Individual(f"A", problem=MaxOnes()) ]
    pop0[0].fitness = 100

    # pop1 has 5 references to the same individual
    ind = Individual(f"B", problem=MaxOnes())
    pop1 = [ ind for i in range(5) ]
    for x in pop1:
        x.fitness = 10
    assert(len(pop1) == 5)

    # Create the operator
    op = migrate(topology=nx.complete_graph(2),
                                    emigrant_selector=naive_cyclic_selection,
                                    replacement_selector=naive_cyclic_selection,
                                    migration_gap=50)

    # Generation 0
    context['leap']['generation'] = 0

    context['leap']['current_subpopulation'] = 0
    pop0 = op(pop0)  # This call will choose an emigrant from pop0

    context['leap']['current_subpopulation'] = 1
    pop1 = op(pop1)
    assert(len(pop1) == 5), f"The population's size shouldnt' change after migration, but got {len(pop1)} instead of 5."
    #assert(pop1[0].genome == 'A'), "The first element of pop1 should be replaced by the first element of pop0"
