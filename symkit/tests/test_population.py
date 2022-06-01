import pandas as pd
import pytest
from sklearn.utils import check_random_state
from sympy import symbols

from ..operators import add, cos, mul, pdiv, sub
from ..population import get_next_generation, tree_distances


def test_distances():
    a, b, c = symbols("a, b, c")
    exprs = {a * b, a + b, a * c - b}
    distances = tree_distances(exprs)
    assert distances.mean(axis=0).idxmax() == a * c - b


def test_population_class():
    from sympy import cos
    from sympy.abc import x, y

    fitness = pd.Series(
        {x ** y: 2.0, x + y: 3.0, x - y: 1.0, x + (2 * y): 0.5, cos(x): 4.0,}
    )

    fns = [add, sub, cos]

    new_gen = get_next_generation(
        fitness,
        fns,
        [x, y],
        check_random_state(10),
        crossover_ratio=0.5,
        subtree_ratio=0.3,
        p_float=0.0,
    )

    assert (
        cos(x) in new_gen
    ), "cos(x) should be selected by simple reproduction (best score)"
    assert cos(y) in new_gen, "cos(y) should result from a subtree mutation of cos(x)"
    # assert x + cos(x) in new_gen, "x + cos(x) should result from a crossover"  FIXME
