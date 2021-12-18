import random
from math import ceil, floor

import numpy as np
import sympy
from sympy import Abs, Function, Piecewise, S, Symbol, preorder_traversal, symbols
from sympy.abc import x, y
from sympy.printing.lambdarepr import NumExprPrinter
from sympy.utilities.lambdify import lambdastr

# TODO : replace by sympy.Add, sympy.Sub and sympy.Mul
add2 = x + y
sub2 = x - y
mul2 = x * y
div2 = Piecewise((x / y, Abs(y) > 0.001), (1.0, True))


class pdiv(Function):
    @classmethod
    def eval(cls, x, y):
        _len = (
            len(x)
            if hasattr(x, "__len__")
            else len(y)
            if hasattr(y, "__len__")
            else None
        )
        if _len is not None:
            x, y = np.array(x, dtype=float), np.array(y, dtype=float)
            oks = np.abs(y) > 0.001
            ones = np.ones(_len, dtype=float)
            return np.divide(x, y, out=ones, where=oks)
        if y.is_real:
            if Abs(y) > 0.001:
                return x / y
            else:
                return S.One


div2 = pdiv(x, y)

DEFAULT_OPS = [
    add2,
    sub2,
    mul2,
    div2,
]


def get_symbols(expr):
    return sorted(expr.atoms(Symbol), key=str)


def is_function(expr):
    from sympy import Function
    from sympy.core.operations import AssocOp

    return isinstance(expr, (Function, AssocOp))


def get_subtree(expr, start=0):
    picks = list(preorder_traversal(expr))[start:]
    if not picks:
        return S.Zero
    probs = [0.9 if is_function(_) else 0.1 for _ in picks]
    return random.choices(picks, weights=probs, k=1)[0]


def random_expr(ops=DEFAULT_OPS, syms=symbols("X:10"), size=10, evaluate=True):
    if size <= 1:
        return random.choice(syms)
    op = random.choice(ops)
    left_size, right_size = floor(size / 2), ceil(size / 2)
    left = random_expr(ops, syms, size=left_size)
    right = random_expr(ops, syms, size=right_size)
    return op.subs([(x, left), (y, right)], evaluate=evaluate)


def hoist_mutation(expr):
    to_replace = get_subtree(expr, start=1)  # start at 1 to avoid taking the root
    sub = get_subtree(to_replace, start=1)
    return expr.subs(to_replace, sub, evaluate=True)


def subtree_mutation(expr, ops, syms=None, size=2, evaluate=True):
    if not expr or not expr.args:
        return random_expr(ops, syms, size=size, evaluate=evaluate)
    to_replace = random.choice(expr.args)
    if syms is None:
        syms = get_symbols(expr)
    new_expr = random_expr(ops, syms, size=size)
    return expr.subs(to_replace, new_expr, evaluate=evaluate)


def crossover(expr1, expr2, evaluate=False):
    """
    split_point is the point where we split a nformulae
    a value of 0.5(default) will take the left part of expr1,
    and join it with left part of expr2
    """
    to_replace = get_subtree(expr1)
    return expr1.subs(to_replace, expr2, evaluate=evaluate)


def arity(expr):
    return len(expr.find(lambda e: e.is_symbol))


def complexity(expr, complexity_map={Piecewise: lambda e: e.args[0].args[0]}):
    for op_type, accessor in complexity_map.items():  # TODO avoid substitution
        founds = expr.find(op_type)
        expr = expr.subs(zip(founds, map(accessor, founds)))
    return sum((1 for _ in preorder_traversal(expr)))
