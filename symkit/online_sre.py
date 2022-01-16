from functools import partial

import numpy as np
import pandas as pd
from river import base, metrics
from sklearn.utils import check_random_state
from sympy import S, Symbol, symbols

from symkit.expression import hoist_mutation, random_expr_full, random_expr_grow
from symkit.operators import add2, div2, mul2, sub2

from .expression import complexity, get_subtree


def init_population(
    random_state, population_size, operators, init_size, syms, population=set()
):
    low, high = init_size
    n_iter = 0
    while len(population) < population_size // 2 and n_iter < 1000:
        size = random_state.randint(low, high)
        expr = random_expr_full(operators, syms, size, random_state)
        population.add(expr)
        n_iter += 1
    n_iter = 0
    while len(population) < population_size and n_iter < 1000:
        size = random_state.randint(low, high)
        expr = random_expr_grow(operators, syms, size, random_state)
        population.add(expr)
        n_iter += 1
    return {_: metrics.MAE() for _ in population}


def subtree_mutation(expr, ops, syms, size, random_state):
    from sympy import preorder_traversal

    if not expr or not expr.args:
        return random_state.choice(syms)
    # m = max((complexity(c) - size for c in preorder_traversal(expr)))
    # cands = [c for c in preorder_traversal(expr) if complexity(c) == m]
    cands = [c for c in preorder_traversal(expr) if complexity(c) == size]
    if not cands:
        print()
        to_replace = get_subtree(expr, start=3, random_state=random_state)
    else:
        to_replace = random_state.choice(cands)
    if random_state.choice([0, 1]):
        random_meth = random_expr_full
    else:
        random_meth = random_expr_grow
    new_expr = random_meth(ops, syms, size=size, random_state=random_state)
    return expr.subs(to_replace, new_expr)


class OnlineSymbolicRegression(base.Regressor):
    # try:  # symengine is faster
    #    from symengine import Symbol
    # except:
    from sympy import Symbol

    def __init__(
        self,
        population_size: int = 20,
        operators=[add2, sub2, div2, mul2],
        random_state=None,
        init_size=(3, 8),
        random_mutation_size=1,
        sample_weight_inc: float = 0.1,
    ):
        if sample_weight_inc > 0.5 or sample_weight_inc < 0.0:
            raise ValueError("sample_weight_inc should be in [0.0, 0.5]")
        self.population_size = population_size
        # self._supervised = True
        self.operators = operators
        self.random_state = check_random_state(random_state)
        self._hof = None
        self.init_size = init_size
        self.random_mutation_size = random_mutation_size
        self._hoist = partial(hoist_mutation, random_state=self.random_state)
        self.sample_weight_inc = sample_weight_inc
        self.mutate_every = int(1 / sample_weight_inc)
        self._subtree = partial(
            subtree_mutation,
            random_state=self.random_state,
            ops=self.operators,
            size=random_mutation_size,
        )
        self.expression = S.Zero
        self._history = list()
        self._sample_weight = 1

    def learn_one(self, x: dict, y):
        x = {k: v for k, v in x.items() if isinstance(v, (int, float))}
        syms = [self.Symbol(_) for _ in x]
        if self._hof is None:  # not fitted
            hof = init_population(
                self.random_state,
                self.population_size,
                self.operators,
                self.init_size,
                syms,
            )
        else:
            hof = self._hof

        evals = dict()
        for expr in hof:
            val = expr.xreplace(x)
            try:
                evals[expr] = float(val)
            except:
                evals[expr] = 10_000  # really high value

        for expr, s in hof.items():
            # give more weight to lastly seen observations
            x = evals[expr]
            s.update(x, y, sample_weight=self._sample_weight)
        self._sample_weight += self.sample_weight_inc

        if len(self) % self.mutate_every == 0:  # mutation every `self.mutate_every`
            float_hof = {_: v.get() for _, v in hof.items()}
            s_hof = sorted(float_hof, key=float_hof.__getitem__)
            expression = s_hof[0]
            payload = dict(
                best_fitness=float_hof[expression],
                complexity_mean=np.mean([complexity(_) for _ in hof]),
            )
            self._history.append(payload)

            # subtree mutations
            # take strongest expressions, apply slight mutation on them
            # to see if something comes up
            # allocate their `parent` score as the score
            split_point = len(s_hof) // 2  # take half of it
            reproduction = s_hof[:split_point]
            subtree_d = {self._subtree(e, syms=syms): e for e in reproduction}
            hof = {
                **{_: hof[_] for _ in reproduction},
                **{_: hof[v] for _, v in subtree_d.items()},
            }

            diff = self.population_size - len(hof)
            if diff > 0:
                hof.update(
                    init_population(
                        self.random_state,
                        diff,
                        self.operators,
                        self.init_size,
                        syms,
                        population=set(hof),
                    )
                )

            nans = {e for e, s in float_hof.items() if s in (np.nan, np.inf, -np.inf)}
            if nans:
                for _ in nans:
                    del hof[_]
                hof.update(
                    init_population(
                        self.random_state,
                        len(nans),
                        self.operators,
                        self.init_size,
                        syms,
                        population=set(hof),
                    )
                )

            self.expression = expression
            self._hof = hof

        return self

    @property
    def hall_of_fame(self):
        hof = {k: v.get() for k, v in self._hof.items()}
        return pd.Series(hof, dtype=float, name="MAE")

    def predict_one(self, x: dict):
        x = {Symbol(k): v for k, v in x.items() if isinstance(v, (int, float))}
        return self.expression.xreplace(x)

    def __len__(self):
        return int((self._sample_weight - 1) / self.sample_weight_inc)
