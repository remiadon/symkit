from operator import sub
from sympy import Expr, symbols
from sympy import Symbol
from sympy.abc import R, x, y
import sympy
from sympy import preorder_traversal, postorder_traversal

from math import ceil, floor

import random

add2 = x + y
sub2 = x - y
mul2 = x * y

DEFAULT_OPS = [
    add2,
    sub2,
    mul2,
]

def get_symbols(expr):
    return sorted(expr.atoms(Symbol), key=str)

def random_expr(ops=DEFAULT_OPS, syms=symbols("X:10"), size=10, evaluate=True):
    if size <= 1:
        return random.choice(syms)
    op = random.choice(ops)  #TODO : get_arity
    left_size, right_size = floor(size / 2), ceil(size / 2)
    left = random_expr(ops, syms, size=left_size)
    right = random_expr(ops, syms, size=right_size)
    return op.subs([(x, left), (y, right)], evaluate=evaluate)

def mutate(expr, ops, syms=None, size=2, evaluate=True):
    to_replace = random.choice(expr.args)
    if syms is None:
        syms = get_symbols(expr)
    new_expr = random_expr(ops, syms, size=size)
    return expr.subs(to_replace, new_expr, evaluate=evaluate)

def mate(expr1, expr2, split_point=0.5, op=sympy.core.add.Add, evaluate=True):
    """
    split_point is the point where we split a formulae
    a value of 0.5(default) will take the left part of expr1, 
    and join it with left part of expr2
    """
    def prepare(expr):
        if isinstance(expr, op):
            return random.choice(expr.args)
        return expr

    left = prepare(expr1)
    right = prepare(expr2)
    return op(left, right, evaluate=evaluate)

def arity(expr):
    return len(expr.find(lambda e: e.is_symbol))


def internal_nodes(expr):
    return [_ for _ in preorder_traversal(expr) if not isinstance(_, (Symbol, float))]

def complexity(expr):
    return sum((1 for _ in preorder_traversal(expr)))
