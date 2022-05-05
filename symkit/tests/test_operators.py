import numpy as np
import polars as pl
import pytest
from sympy import symbols
from sympy.abc import x, y

from ..operators import pdiv

a, b, c, d = symbols("a b c d")


@pytest.mark.parametrize("dtype", (pl.Float32, pl.Float64))
def test_pdiv(dtype):
    assert pdiv(a, 3) == a / 3
    assert pdiv(a, 0) == 1
    assert pdiv(pdiv(a, b), pdiv(c, d)) == pdiv(a * d, b * c)
    assert pdiv(3, pdiv(a, b)) == pdiv(3 * b, a)
    assert pdiv(pdiv(a, b), 3) == pdiv(a, b) / 3
    assert pdiv(pdiv(a, b), c) == pdiv(a, c * b)
    assert pdiv(pdiv(a, b), a) == pdiv(1, b)
    assert pdiv(pdiv(a, b), b) == pdiv(a, b ** 2)
    assert pdiv(4 * a, a) == 4

    pl_pdiv = pdiv(a, b).polars(pl.col("a"), pl.col("b"))
    df = pl.DataFrame(
        [pl.Series("a", [2, 3], dtype=dtype), pl.Series("b", [0.001, 4], dtype=dtype)]
    )

    assert df.select(pl_pdiv.is_nan().sum()).select_at_idx(0)[0] == 0
