import numbers
import re
from functools import lru_cache

import polars as pl
import sympy as sp
from sympy import lambdify, symbols


def symbol_generator():
    i = 0
    while True:  # TODO : set a limit
        yield symbols(f"__symkit_{i}")
        i += 1


@lru_cache()
def camel_to_lower(s: str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


@lru_cache()
def _sympy_to_polars(expr: sp.Expr) -> pl.Expr:
    if expr.is_Integer:
        return int(expr)
    if expr.is_Number:
        return pl.lit(float(expr))
    if expr.is_Symbol:
        return pl.col(str(expr))
    args = list()
    for arg in expr.args:
        ev = _sympy_to_polars(arg)
        args.append(ev)
    pl_fn = getattr(expr, "polars", None)
    if pl_fn is not None:
        return pl_fn(*args)
    _t = camel_to_lower(str(type(expr)))
    pl_fn = getattr(pl.Expr, _t, None)
    if pl_fn is None:
        syms = symbols(f"X0:{len(args)}")
        _expr = type(expr)(*syms)
        pl_fn = lambdify(syms, _expr)
    return pl_fn(*args)


def sympy_to_polars(expr: sp.Expr) -> pl.Expr:
    pl_expr = _sympy_to_polars(expr)
    if isinstance(pl_expr, numbers.Number):
        pl_expr = pl.lit(pl_expr)
    return pl_expr
