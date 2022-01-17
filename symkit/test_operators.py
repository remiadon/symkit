import numpy as np
from sympy import symbols
from sympy.abc import x, y

from .operators import pdiv

a, b, c, d = symbols("a b c d")


def test_pdiv():
    assert pdiv(a, 3) == a / 3
    assert pdiv(a, 0) == 1
    assert pdiv(pdiv(a, b), pdiv(c, d)) == pdiv(a * d, b * c)
    assert pdiv(3, pdiv(a, b)) == pdiv(3 * b, a)
    assert pdiv(pdiv(a, b), 3) == pdiv(a, b) / 3
    assert pdiv(pdiv(a, b), c) == pdiv(a, c * b)
    assert pdiv(pdiv(a, b), a) == pdiv(1, b)
    assert pdiv(pdiv(a, b), b) == pdiv(a, b ** 2)

    assert not np.isinf(pdiv(a, b).vectorized_fn([2, 3], [0.001, 4])).any()
