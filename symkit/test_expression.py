import pytest
from sympy.core.symbol import Symbol 
from sympy.parsing.sympy_parser import parse_expr
from .expression import random_expr, complexity, mate
import sympy
from operator import itemgetter

from .expression import random

@pytest.mark.parametrize("size", (2, 5, 10))
def test_random_expression(size):
    expr = random_expr(size=size, evaluate=False)
    assert expr.count(Symbol) <= size

@pytest.mark.parametrize("expr1,expr2,result,operator", [
    ("X0 * 2", "X6", "X0 * 2 + X6", sympy.core.add.Add),
    ("X0 * 3 + X2", "(X3 - 1) * (X4 + 2)", "X0 * 3 + (X3 - 1) * (X4 + 2)", sympy.core.add.Add),
    ("X0 * 2", "X6", "X0 * X6", sympy.core.add.Mul),
    ("X0 * 3 + X2", "(X3 - 1) * (X4 + 2)", "(3*X0 + X2)*(X4 + 2)", sympy.core.add.Mul),
])
def test_mate(expr1, expr2, result, operator, monkeypatch):
    monkeypatch.setattr(random, "choice", itemgetter(1))
    expr1 = parse_expr(expr1)
    expr2 = parse_expr(expr2)
    child = mate(expr1, expr2, op=operator)
    assert child == parse_expr(result)
