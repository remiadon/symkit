import numbers
import operator
import re
from functools import lru_cache, reduce
from multiprocessing.sharedctypes import Value

import polars as pl
import sympy as sp
from sympy import symbols

__acc_start_vals = dict(mul=1, add=0)


def symbol_generator():
    i = 0
    while True:  # TODO : set a limit
        yield symbols(f"__symkit_{i}")
        i += 1


@lru_cache()
def camel_to_lower(s: str):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


sympy_to_str = lru_cache()(str)  # hash(expr) is fast, but str(expr) is slow


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
    _t = str(type(expr)).rsplit(".", 1)[1].replace("'>", "")
    _t = camel_to_lower(_t)
    pl_fn = getattr(pl.Expr, _t, None)
    if pl_fn is None:
        op = getattr(operator, _t, None)
        pl_fn = lambda *args: reduce(op, args, __acc_start_vals[_t])
        if pl_fn is None:
            raise ValueError(f"could not retrive operator for {_t}")
    return pl_fn(*args)


def sympy_to_polars(expr: sp.Expr) -> pl.Expr:
    pl_expr = _sympy_to_polars(expr)
    if isinstance(pl_expr, numbers.Number):
        pl_expr = pl.lit(pl_expr)
    return pl_expr
