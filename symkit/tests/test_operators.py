import numpy as np
import pytest
from sympy import symbols
from sympy.abc import x, y

from ..operators import pdiv

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
    assert pdiv(4 * a, a) == 4

    assert not np.isinf(pdiv(a, b).vectorized_fn([2, 3], [0.001, 4])).any()


@pytest.mark.parametrize("shape", (-1, (4, 1)))
def test_pdiv_vect(shape):
    x1 = np.array([3, np.inf, 10, 5]).reshape(shape)
    x2 = np.array([2, 1, 0.00001, -0.01]).reshape(shape)
    op = pdiv(a, b)
    np.testing.assert_almost_equal(
        op.vectorized_fn(x1, x2), [3 / 2, np.inf, 1.0, -500.0]
    )

    res = op.vectorized_fn(x1, 0.000001)
    assert len(res.shape) == 1
    np.testing.assert_array_almost_equal(op.vectorized_fn(x1, 0.00001), [1.0] * len(x1))

    assert len(op.vectorized_fn(0.0001, x2).shape) == 1
    assert op.vectorized_fn(0.0001, x2).shape[0] == len(x2)

    assert len(op.vectorized_fn(x1, 3).shape) == 1
    assert op.vectorized_fn(x1, 3).shape[0] == len(x1)
