from operator import itemgetter

import pytest
import sympy
from sympy.core.symbol import Symbol
from sympy.parsing.sympy_parser import parse_expr

from .expression import complexity, crossover, hoist_mutation, random, random_expr


@pytest.mark.parametrize("size", (2, 5, 10))
def test_random_expression(size):
    expr = random_expr(size=size, evaluate=False)
    assert expr.count(Symbol) <= size


@pytest.mark.parametrize(
    "expr1,expr2,result",
    [
        ("X0 * 2", "X6", "X0 * X6"),
        ("X0 * 3 + X2", "(X3 - X1) * (X4 + 2)", "3 * X0 + (-X1 + X3) * (X4 + 2)"),
    ],
)
def test_crossover(expr1, expr2, result, monkeypatch):
    monkeypatch.setattr(random, "choices", lambda e, **kwargs: [e[1]])
    expr1 = parse_expr(expr1)
    expr2 = parse_expr(expr2)
    child = crossover(expr1, expr2)
    assert child == parse_expr(result)


@pytest.mark.parametrize(
    "expr, complexity_mesure",
    [
        ("X0 * 2", 3),
        ("Piecewise((height/weight, 2*Abs(weight) > 0.001), (1.0, True))", 5),
    ],
)
def test_complexity(expr, complexity_mesure):
    expr = parse_expr(expr)
    assert complexity(expr) == complexity_mesure


@pytest.mark.parametrize("expr", ["sin(X ** 2) - 2 + cos(Y)", "Abs(X // 2) + Y % 3",])
def test_hoist_mutation(expr):
    expr = parse_expr(expr)
    hoisted = hoist_mutation(expr)
    assert complexity(hoisted) < complexity(expr)
    assert hoisted.find(Symbol).issubset(expr.find(Symbol))
