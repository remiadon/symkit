import pytest
from sklearn.utils import check_random_state
from sympy import symbols
from sympy.core.symbol import Symbol
from sympy.parsing.sympy_parser import parse_expr

from ..expression import (
    complexity,
    crossover,
    get_subtree,
    hoist_mutation,
    random_expr_full,
    random_expr_grow,
    tree_distance,
    tree_hash,
)
from ..operators import add2, cos1, div2, mul2, sin1, sub2

syms = symbols("X:10")


def test_getsubtree():
    rs = check_random_state(12)
    X0, X1, *_ = syms
    expr = X0 * (X1 + 1) ** 2
    subs = [get_subtree(expr, rs, start=0) for _ in range(6)]
    assert subs == [X0 * (X1 + 1) ** 2, 1, (X1 + 1) ** 2, X1 + 1, X0 * (X1 + 1) ** 2, 2]
    subs = [get_subtree(expr, rs, start=1) for _ in range(4)]
    assert subs == [2, (X1 + 1) ** 2, 2, (X1 + 1) ** 2]


@pytest.mark.parametrize("size", (2, 5, 10, 20))
def test_random_expression_full(size):
    random_state = check_random_state(12)
    ops = [add2, sub2, mul2, div2, sin1, cos1]
    expr = random_expr_full(ops, syms, size, random_state, p_float=0.0)
    assert complexity(expr) <= size


@pytest.mark.parametrize("size", (2, 5, 10, 20))
def test_random_expression_grow(size):
    random_state = check_random_state(None)
    ops = [add2, sub2, mul2, div2, sin1, cos1]
    expr = random_expr_grow(ops, syms, size, random_state, p_float=0.0)
    assert complexity(expr) == size
    assert expr.count(Symbol) <= size


@pytest.mark.parametrize(
    "expr1,expr2,result",
    [
        ("X0 * 2", "X6", "X6"),
        ("X6", "X0 * X2", "X2 * X6"),  # non commutative
        ("X0 * 3 + X2", "(X3 - X1) * (X4 + 2)", "(3 * X0 + X2) * (X4 + 2)"),
        ("Mul(a, b, c)", "(a / (b - 4)) * (c + d)", "a * (c + d) / (a * b * c + b)"),
    ],
)
def test_crossover(expr1, expr2, result):
    random_state = check_random_state(2)
    expr1 = parse_expr(expr1)
    expr2 = parse_expr(expr2)
    child = crossover(expr1, expr2, random_state=random_state)
    assert child == parse_expr(result)


@pytest.mark.parametrize(
    "expr, complexity_mesure",
    [
        ("X0 * 2", 3),
        ("Piecewise((height/weight, 2*Abs(weight) > 0.001), (1.0, True))", 14),
        ("X0 * 5 - X1", 5),  # make sure `-X1` does not account for (+ -1 * ...)
    ],
)
def test_complexity(expr, complexity_mesure):
    expr = parse_expr(expr)
    assert complexity(expr) == complexity_mesure


@pytest.mark.parametrize("expr", ["sin(X ** 2) - 2 + cos(Y)", "Abs(X // 2) + Y % 3",])
def test_hoist_mutation(expr):
    expr = parse_expr(expr)
    hoisted = hoist_mutation(expr, random_state=check_random_state(2))
    assert complexity(hoisted) < complexity(expr)
    assert hoisted.find(Symbol).issubset(expr.find(Symbol))


@pytest.mark.parametrize("expr", ["sin(X ** 2) - 2 + cos(Y)", "Abs(X // 2) + Y % 3"])
def test_tree_hash(expr):
    expr = parse_expr(expr)
    hashes = tree_hash(expr)
    assert len(hashes) == complexity(expr)


def test_tree_distance():
    a, b, c = parse_expr("X0 + 1"), parse_expr("X0 + 2"), parse_expr("X0 * 3")
    assert tree_distance(a, b) < tree_distance(b, c)
    d = parse_expr("(X0 + 1) * 2")
    assert tree_distance(a, b) < tree_distance(a, d)
    assert tree_distance(a, d) < tree_distance(a, c)
