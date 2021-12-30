import random
from functools import partial
from typing import AnyStr, List

import joblib
import numexpr  # just to early raise an error, before calling lambdify
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state

# from joblib.parallel import delayed
from sympy import Expr, Function, symbols
from sympy.core.symbol import Symbol

from .core import symbol_generator
from .expression import (
    add2,
    complexity,
    crossover,
    div2,
    hoist_mutation,
    mul2,
    random_expr,
    sub2,
    subtree_mutation,
)


def _execute(expr: Expr, X: dict, symbol_gen):
    from sympy import lambdify, nan, oo, zoo

    n_samples = max(map(len, X.values()))
    if expr.has(oo, -oo, zoo, nan):
        return np.zeros(n_samples, dtype=float)  # TODO log this
    if expr.is_number:
        return np.repeat(float(expr), n_samples).astype(float)

    if expr.is_Function:  # user defined function, bypass numexpr
        # recursive calls to still rely on numexpr in subexpressions
        _args = list()
        for _ in expr.args:
            _args.append(_execute(_, X, symbol_gen))
        return expr._numpy_(*_args)  # eg. protected div

    syms = sorted(expr.free_symbols, key=str)
    _args = [X[_] for _ in syms]

    user_defined = expr.find(Function)
    if user_defined:
        _args = list(_args)
        replace_dict = dict()
        for _ in user_defined:
            _args.append(_execute(_, X, symbol_gen))
            var = next(symbol_gen)
            replace_dict[_] = var
            syms.append(var)
        expr = expr.subs(replace_dict)

    if expr.is_Pow:  # FIXME : this makes numexpr fail, but not numpy
        fn = lambdify(syms, expr, "numpy")
    else:
        fn = lambdify(syms, expr, "numexpr")

    res = fn(*_args)

    nans = np.isnan(res) | (res == np.inf) | (res == -np.inf)
    if nans.any():  # FIXME this happened with expr=X9**(-X10)
        res[nans] = 10_000_000  # arbitrary high value

    return res


def score(y, y_pred, expr):
    return r2_score(y, y_pred)  # - complexity(expr) / 100


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators: List[Expr] = [add2, sub2, mul2, div2],
        population_size: int = None,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        selection_ratio=0.5,
        ratios=dict(
            reproduction=0.4, crossover=0.2, subtree_mutation=0.2, hoist_mutation=0.2
        ),
        random_state=12,
    ):
        self.population_size = population_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        self.selection_ratio = selection_ratio
        assert isinstance(ratios, dict) and sum(ratios.values()) == 1.0
        self.ratios = ratios
        self.random_state = random_state

    def _check_input(self, X, y=None, return_symbols=False):
        if not isinstance(X, pd.DataFrame):
            if y is not None:
                X, y = check_X_y(X, y)  # TODO : check parameter setting for `check_X_y`
            else:
                X = check_array(X)
            syms = symbols(f"X:{X.shape[1]}")
            values = X
        else:
            syms = X.columns.map(Symbol)
            values = X.values
        X = dict(zip(syms, values.T))
        if return_symbols:
            return X, y, syms
        return X, y

    def fit(self, X, y):
        X, y, syms = self._check_input(X, y, return_symbols=True)
        random_state = check_random_state(self.random_state)

        history = list()
        population_size = self.population_size or len(self.operators) ** 2
        init_size = 2

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
            for expr in random_state.choice(
                reproduction.index, size=hoist_mutation_size, replace=False
            ):
                hoisted = hoist_mutation(expr, random_state=random_state)
                population.add(hoisted)

            # SUBTREE MUTATIONS
            for expr in random_state.choice(
                reproduction.index, size=subtree_mutation_size, replace=False
            ):
                n_symbols = expr.count(Symbol)
                mutation_size = np.ceil(np.log(n_symbols)) if n_symbols else init_size
                subtree_mutant = subtree_mutation(
                    expr,
                    self.operators,
                    syms,
                    size=mutation_size,  # TODO : auto set size inside subtree_mutation function
                    random_state=random_state,
                )
                population.add(subtree_mutant)

            # CROSSOVERS
            for expr1, expr2 in random_state.choice(
                reproduction.index, size=(crossover_size, 2)
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
        sym_gen = symbol_generator()
        preds = {expr: _execute(expr, X, sym_gen) for expr in population}
        fitness = {expr: score(y, y_pred, expr) for expr, y_pred in preds.items()}
        return pd.Series(fitness)

    def predict(self, X):
        expr = getattr(self, "expression_", None)
        if expr is None:
            raise NotFittedError()
        X, _ = self._check_input(X, y=None)
        y_pred = _execute(expr, X, symbol_gen=symbol_generator())
        return y_pred

    # TODO predict_proba from (mean) sigmoid from self.expression | self.hall_of_fame

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
