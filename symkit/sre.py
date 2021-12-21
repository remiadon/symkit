import random
from functools import partial
from typing import AnyStr, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state

# from joblib.parallel import delayed
from sympy import Expr, symbols
from sympy.core.symbol import Symbol

from .expression import (
    add2,
    complexity,
    crossover,
    div2,
    get_symbols,
    hoist_mutation,
    mul2,
    pdiv,
    random_expr,
    sub2,
    subtree_mutation,
)


# TODO : joblib.cache this function
def _execute(expr: Expr, X: pd.DataFrame, modules="numpy"):
    from sympy import lambdify, nan, oo, zoo

    if expr.has(oo, -oo, zoo, nan):
        return np.zeros(len(X), dtype=float)  # TODO log this
    if expr.is_number:
        return np.repeat(float(expr), len(X)).astype(float)

    syms = [str(_) for _ in get_symbols(expr)]
    args = X[syms].values.T
    fn = lambdify(syms, expr, modules)
    res = fn(*args)

    if np.isnan(res).any():  # FIXME this happened with expr=X9**(-X10)
        return np.zeros(len(X), dtype=float)

    if ((res == np.inf) | (res == -np.inf)).any():  # TODO this also happens with **
        return np.zeros(len(X), dtype=float)

    return res


def score(y, y_pred, expr):
    return r2_score(y, y_pred) - complexity(expr) / 100


def pick(size_getter, target_size, pool, limit=1_000, **kwargs):
    n_try = 0
    while size_getter() < target_size:
        choices = random.choices(pool, **kwargs)
        if len(choices) == 1:
            yield choices[0]
        else:
            yield choices
        n_try += 1
        if n_try > limit:
            break


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators: List[Expr] = [add2, sub2, mul2, div2],
        population_size: int = None,
        init_size: int = None,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        selection_ratio=0.5,
        ratios=dict(
            reproduction=0.4, crossover=0.2, subtree_mutation=0.2, hoist_mutation=0.2
        ),
        memory=joblib.Memory(location=None),
        modules: AnyStr = [
            "numpy",
            dict(pdiv=pdiv),
        ],  # TODO : protected division with Piecewise
        random_state=12,
        elimination=False,  # TODO make this work
    ):
        self.population_size = population_size
        self.init_size = init_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        self.selection_ratio = selection_ratio
        assert isinstance(ratios, dict) and sum(ratios.values()) == 1.0
        self.ratios = ratios
        self.memory = memory
        self.random_state = random_state
        self.elimination = elimination
        self.modules = modules

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(X, y)  # TODO : check parameter setting for `check_X_y`
            X = pd.DataFrame(data=X, columns=map(str, symbols(f"X:{X.shape[1]}")))

        # FIXME we should not mutate data, see `X.loc[:, ...] = ...` in self.evaluate_population
        self._execute = self.memory.cache(
            partial(_execute, modules=self.modules), ignore=["X"]
        )

        history = list()
        syms = symbols(X.columns.tolist())
        init_size = self.init_size or max(len(syms), len(self.operators))
        population_size = self.population_size or len(self.operators) ** 2

        reproduction_size = int(population_size * self.ratios["reproduction"])
        subtree_mutation_size = int(population_size * self.ratios["subtree_mutation"])
        hoist_mutation_size = int(population_size * self.ratios["hoist_mutation"])
        crossover_size = int(population_size * self.ratios["crossover"])

        # intiliaze to random
        # use a set for deduplication
        population = set()
        while len(population) < population_size:
            expr = random_expr(
                ops=self.operators, syms=syms, size=init_size, random_state=random_state
            )
            population.add(expr)

        max_fit = -np.inf
        n_iter_no_change = 0

        for _ in range(self.n_iter):
            fitness_vect = self.evaluate_population(population, X, y)
            _max_fit = fitness_vect.max()
            if _max_fit <= max_fit:
                n_iter_no_change += 1
            else:
                max_fit = _max_fit
            if n_iter_no_change >= self.n_iter_no_change:
                break
            history_payload = dict(
                fitness_mean=fitness_vect.mean(),
                fitness_max=fitness_vect.max(),
                expr_complexity_mean=np.mean(fitness_vect.index.map(complexity)),
            )
            history.append(history_payload)

            # REPRODUCTIONS
            reproduction = fitness_vect.nlargest(reproduction_size)
            population = set(reproduction.index)  # start with those who reproduce

            # HOIST MUTATIONS
            target_size = reproduction_size + hoist_mutation_size
            for expr in pick(population.__len__, target_size, reproduction.index):
                hoisted = hoist_mutation(expr, random_state=random_state)
                population.add(hoisted)

            # SUBTREE MUTATIONS
            target_size += subtree_mutation_size
            for expr in pick(population.__len__, target_size, reproduction.index):
                n_symbols = expr.count(Symbol)
                mutation_size = np.ceil(np.log(n_symbols))
                subtree_mutant = subtree_mutation(
                    expr,
                    self.operators,
                    syms,
                    size=mutation_size,
                    random_state=random_state,
                )
                population.add(subtree_mutant)

            # CROSSOVERS
            target_size += crossover_size
            for expr1, expr2 in pick(
                population.__len__, target_size, reproduction.index, k=2
            ):
                child = crossover(expr1, expr2, random_state=random_state)
                population.add(child)

        self.symbols_ = syms
        self.history_ = history
        self.hall_of_fame_ = self.evaluate_population(population, X, y)
        self.expression_ = self.hall_of_fame_.idxmax()
        return self

    def _get_tags(self):
        return {
            **super()._get_tags(),
            **dict(
                allow_nan=False,
                binary_only=True,
                requires_fit=True,
                poor_score=True,  # FIXME
                _xfail_checks=dict(
                    check_dtype_object="objects operators will be defined to handle categorical data",
                    check_estimators_data_not_an_array="",
                    check_regressor_data_not_an_array="",
                    check_supervised_y_2d="",
                    check_regressors_int="",
                    check_fit_idempotent="",
                    check_fit2d_1sample="",
                    check_no_attributes_set_in_init="need to set self.execute_ at init time",
                ),
                X_types=["2darray"],  # TODO : add more dtypes, this should be doable
            ),
        }

    def evaluate_population(self, population: List[Expr], X: pd.DataFrame, y):
        """
        apply a population of sympy expression onto the input data `X`

        This version tries to identify subexpressions, in order to mutualise
        pre-computed results.
        """
        # TODO
        # 1. cache calls to _execute

        # TODO parallel execution
        from sympy import cse

        def infinite_gen():  # TODO : apply cse and then hashing on subexpr as its identifier, to optimise caching
            i = 0
            while True:
                yield symbols(f"__symkit_{i}")
                i += 1

        if self.elimination:
            common_exprs, exprs = cse(population, order="none", symbols=infinite_gen())
            # iteratively, because cse uses freshly discovered symbols to expression new ones
            for (sym, expr,) in common_exprs:
                X.loc[:, str(sym)] = self._execute(expr, X)
            preds = {
                orig_expr: self._execute(dest_expr, X)
                for orig_expr, dest_expr in zip(population, exprs)
            }
        else:
            preds = {expr: self._execute(expr, X) for expr in population}
        fitness = {expr: score(y, y_pred, expr) for expr, y_pred in preds.items()}
        return pd.Series(fitness)

    def predict(self, X, expr=None):
        if expr is None:
            expr = getattr(self, "expression_", None)
        if expr is None:
            raise NotFittedError()
        if not isinstance(X, pd.DataFrame):
            X = check_array(X)
            syms = getattr(self, "symbols_") or symbols(f"X:{X.shape[1]}")
            cols = map(str, syms)
            X = pd.DataFrame(data=X, columns=cols)
        y_pred = _execute(expr, X, modules=self.modules)  # no cache here
        return y_pred

    # TODO predict_proba from (mean) sigmoid from self.expression | self.hall_of_fame

    def score(self, X, y, expr=None):
        if expr is None:
            expr = getattr(self, "expression_", None)
        if expr is None:
            raise NotFittedError()
        y_pred = self.predict(X, expr=expr)
        return r2_score(y, y_pred)
