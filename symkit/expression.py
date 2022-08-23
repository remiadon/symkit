from functools import lru_cache
from typing import List

import numpy as np
import sympy as sp
from sklearn.utils import check_random_state
from sympy import Function, S, Symbol, postorder_traversal, preorder_traversal, sympify
from sympy.core.operations import AssocOp


def is_op(expr: sp.Expr):
    return expr.free_symbols != {expr}


def arity(expr: sp.Expr):
    return expr.count(Symbol)


def get_subtree(expr, random_state, start=0):
    picks = list(preorder_traversal(expr))[start:]
    if not picks:
        return S.Zero
    probs = np.array([0.9 if is_op(_) else 0.1 for _ in picks])
    probs /= probs.sum()
    return random_state.choice(picks, p=probs)


def random_expr_full(
    operators: List[sp.Expr],
    symbols: List[sp.Symbol],
    size: int,
    random_state,
    p_float: float = 0.2,
):
    ops = list({_ for _ in operators if arity(_) < size})
    if size <= 1 or not ops:
        if random_state.choice([1, 0], p=[p_float, 1 - p_float]):
            return sympify(random_state.random_sample())
        else:
            return random_state.choice(symbols)

    op = random_state.choice(ops)
    orig_syms = list(op.free_symbols)
    subs = list()
    d, remained = divmod(size - 1, arity(op))
    for orig_sym in orig_syms:
        _size = d + remained if orig_sym == orig_syms[-1] else d
        sub_expr = random_expr_full(ops, symbols, _size, random_state, p_float)
        subs.append((orig_sym, sub_expr))
    return op.subs(subs)


def random_expr_grow(
    operators: List[sp.Function],
    symbols: List[sp.Symbol],
    size: int,
    random_state,
    p_float: float = 0.2,
):
    expr = random_state.choice(symbols)  # TODO : add float with p_float
    ops = list({_ for _ in operators if arity(_) < size})
    while ops:
        p = np.array([arity(_) ** 2 for _ in ops])
        p = p / p.sum()
        op = random_state.choice(ops, p=p)
        _syms = random_state.choice(symbols, size=arity(op) - 1)
        args = [expr] + _syms.tolist()
        expr = type(op)(
            *args
        )  # TODO : avoid that in order to passed bases containing fixed arguments
        ops = [_ for _ in ops if arity(_) <= size - complexity(expr)]
    return expr


def hoist_mutation(expr, random_state):
    to_replace = get_subtree(
        expr, start=1, random_state=random_state
    )  # start at 1 to avoid taking the root
    sub = get_subtree(to_replace, start=1, random_state=random_state)
    return expr.subs(to_replace, sub)


def subtree_mutation(
    expr: sp.Expr,
    functions: List[sp.Function],
    symbols: List[sp.Symbol],
    random_state,
    p_float=0.2,
):
    if not expr or not expr.args:
        return random_state.choice(symbols)
    to_replace = random_state.choice(expr.args)
    size = complexity(to_replace)
    if random_state.choice([0, 1]):
        random_meth = random_expr_full
    else:
        random_meth = random_expr_grow
    new_expr = random_meth(
        functions, symbols, size=size, random_state=random_state, p_float=p_float
    )
    return expr.subs(to_replace, new_expr)


def crossover(donor, receiver, random_state):
    """
    """
    to_replace = get_subtree(receiver, start=1, random_state=random_state)
    replace = get_subtree(donor, start=0, random_state=random_state)
    return receiver.subs(to_replace, replace)


def complexity(expr):
    ctr = 0
    for _ in preorder_traversal(expr):
        if _.is_Mul and _.args[0] == -1:  # eg. -X1 is X1 * -1
            ctr -= 1
            continue
        ctr += 1
    return ctr


@lru_cache()
def tree_hash(expr: sp.Expr):
    hashes = list()
    for node in postorder_traversal(expr):
        to_hash = type(node) if node.args else node
        hashes.append(hash(to_hash))
    return sorted(hashes)


def tree_distance(expr1: sp.Expr, expr2: sp.Expr):
    H1, H2 = tree_hash(expr1), tree_hash(expr2)
    i = j = count = 0
    while i < len(H1) and j < len(H2):
        if H1[i] == H2[j]:
            count += 1
            i += 1
            j += 1
        elif H1[i] < H2[j]:
            i += 1
        else:
            j += 1
    return 1 - ((2 * count) / (len(H1) + len(H2)))
