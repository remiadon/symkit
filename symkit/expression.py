import numpy as np
from sklearn.utils import check_random_state
from sympy import S, Symbol, preorder_traversal

_STATE = check_random_state(23)


def is_function(expr):
    from sympy import Function
    from sympy.core.operations import AssocOp

    return isinstance(expr, (Function, AssocOp))


def get_subtree(expr, start=0, random_state=_STATE):
    picks = list(preorder_traversal(expr))[start:]
    if not picks:
        return S.Zero
    probs = np.array([0.9 if is_function(_) else 0.1 for _ in picks])
    probs /= probs.sum()
    return random_state.choice(picks, p=probs)


def random_expr_full(ops, syms, size, random_state):
    ops = list({_ for _ in ops if _.count(Symbol) < size})
    if size <= 1 or not ops:
        return random_state.choice(syms)
    op = random_state.choice(ops)
    orig_syms = list(op.free_symbols)
    subs = list()
    d, remained = divmod(size - 1, len(orig_syms))
    for orig_sym in orig_syms:
        _size = d + remained if orig_sym == orig_syms[-1] else d
        sub_expr = random_expr_full(ops, syms, _size, random_state)
        subs.append((orig_sym, sub_expr))
    return op.subs(subs)


"""
def random_expr_full(ops, syms, size, random_state):
    ops = list({_ for _ in ops if _.count(Symbol) < size})
    if not ops:
        return random_state.choice(syms)
    expr = random_state.choice(ops)
    _size = size - complexity(expr)
    _syms = expr.find(Symbol)
    while _size > 0 and _syms:
        ops = list({_ for _ in ops if _.count(Symbol) <= _size // len(_syms)})
        if not ops:
            break
        sub_exprs = random_state.choice(ops, size=len(_syms))
        expr = expr.subs(zip(_syms, sub_exprs))
        _size = size - complexity(expr)  # costly but handle simplifications
        _syms = expr.find(Symbol)

    replace = random_state.choice(syms, size=expr.count(Symbol))
    expr = expr.subs(zip(_syms, replace))
    return expr
"""


def random_expr_grow(ops, syms, size, random_state):
    expr = random_state.choice(syms)
    _size = 1
    ops = [_ for _ in ops if _.count(Symbol) <= size - _size]
    while _size <= size and ops:
        p = np.array([_.count(Symbol) ** 2 for _ in ops])
        p = p / p.sum()
        op = random_state.choice(ops, p=p)
        _syms = random_state.choice(syms, size=op.count(Symbol) - 1)
        args = [expr] + _syms.tolist()
        expr = type(op)(*args)
        _size = complexity(expr)  # expansive but only way to control simplifications
        ops = [_ for _ in ops if _.count(Symbol) <= size - _size]
    return expr


def hoist_mutation(expr, **kwargs):
    to_replace = get_subtree(
        expr, start=1, **kwargs
    )  # start at 1 to avoid taking the root
    sub = get_subtree(to_replace, start=1, **kwargs)
    return expr.subs(to_replace, sub)


def subtree_mutation(expr, ops, syms, random_state=_STATE):
    if not expr or not expr.args:
        return random_state.choice(syms)
    to_replace = random_state.choice(expr.args)
    size = complexity(to_replace)
    if random_state.choice([0, 1]):
        random_meth = random_expr_full
    else:
        random_meth = random_expr_grow
    new_expr = random_meth(ops, syms, size=size, random_state=random_state)
    return expr.subs(to_replace, new_expr)


def crossover(donor, receiver, random_state):
    """
    """
    to_replace = get_subtree(receiver, start=1, random_state=random_state)
    replace = get_subtree(donor, start=0, random_state=random_state)
    return receiver.subs(to_replace, replace)


def arity(expr):  # TODO : see sympy.function.arity
    return len(expr.find(lambda e: e.is_symbol))


def complexity(expr):
    ctr = 0
    for _ in preorder_traversal(expr):
        if is_function(_) and _.args[0] == -1:
            ctr -= 1
            continue
        ctr += 1
    return ctr
