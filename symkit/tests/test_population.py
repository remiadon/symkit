from sympy import symbols

from ..population import tree_distances


def test_distances():
    a, b, c = symbols("a, b, c")
    exprs = {a * b, a + b, a * c - b}
    distances = tree_distances(exprs)
    assert distances.mean(axis=0).idxmax() == a * c - b
