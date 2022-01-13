from typing import AnyStr, List

import numexpr  # just to early raise an error, before calling lambdify
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state

# from joblib.parallel import delayed
from sympy import Expr, symbols
from sympy.core.symbol import Symbol

from .core import symbol_generator
from .expression import (
    complexity,
    crossover,
    hoist_mutation,
    random_expr_full,
    random_expr_grow,
    subtree_mutation,
)
from .operators import UserDefinedFunction, add2, div2, mul2, sub2


def _execute(expr: Expr, X: dict, symbol_gen):
    from sympy import lambdify, nan, oo, zoo

    n_samples = max(map(len, X.values()))

    if expr.has(oo, -oo, zoo, nan):
        return np.zeros(n_samples, dtype=float)  # TODO log this
    if expr.is_number:
        return np.repeat(float(expr), n_samples).astype(float)

    if isinstance(expr, UserDefinedFunction):  # user defined function, bypass numexpr
        # recursive calls to still rely on numexpr in subexpressions
        _args = list()
        for _ in expr.args:
            _args.append(_execute(_, X, symbol_gen))
        return expr.vectorized_fn(*_args)  # eg. protected div

    syms = sorted(expr.free_symbols, key=str)
    _args = [X[_] for _ in syms]

    user_defined = expr.find(UserDefinedFunction)
    if user_defined:
        _args = list(_args)
        replace_dict = dict()
        for _ in user_defined:
            _args.append(_execute(_, X, symbol_gen))
            var = next(symbol_gen)
            replace_dict[_] = var
            syms.append(var)
        expr = expr.subs(replace_dict)

    if expr.is_Pow:  # some powers makes numexpr fail, but not numpy
        fn = lambdify(syms, expr, "numpy")
    else:
        fn = lambdify(syms, expr, "numexpr")

    try:
        res = fn(*_args)
    except Exception as e:
        print(f"could not execute {expr} because of {e}")
        return np.ones(n_samples, dtype=float) * 10_000  # arbitraty high value

    nans = np.isnan(res) | (res == np.inf) | (res == -np.inf)
    if nans.any():  # FIXME this happened with expr=X9**(-X10)
        res[nans] = 10_000_000  # arbitrary high value

    return res


def score(y, y_pred, expr):
    return r2_score(y, y_pred) - complexity(expr) / 1000


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators: List[Expr] = [add2, sub2, mul2, div2],
        population_size: int = 100,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        init_size=(2, 8),
        ratios=dict(
            crossover=0.4, subtree_mutation=0.3, hoist_mutation=0.2, reproduction=0.1,
        ),
        random_state=12,
    ):
        assert population_size
        self.population_size = population_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        assert isinstance(init_size, tuple) and len(init_size) == 2
        self.init_size = init_size
        ratios_sum = sum(ratios.values())
        assert isinstance(ratios, dict) and abs(ratios_sum - 1.0) < 0.001
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
        population = self.prefit(syms, random_state)
        return self._fit(X, y, population, syms, random_state)

    # def partial_fit(self, X, y):
    #    X, y, syms = self._check_input(X, y, return_symbols=True)
    #    if not hasattr(self, "symbols_"):
    #        raise NotFittedError()
    #    if not syms == self.symbols_:  # TODO : only keey sub population with same symbols
    #        raise ValueError()
    #    random_state = check_random_state(self.random_state)
    #    population = self.hall_of_fame_.index.tolist()  # start from pre-learned set of expressions
    #    return self._fit(X, y, population, syms, random_state)

    def prefit(self, syms, random_state):
        # intiliaze to random
        # use a set for deduplication
        population = set()
        low, high = self.init_size
        n_iter = 0
        n_grow = self.population_size // 2  # TODO : pass ratio as param
        while len(population) < n_grow and n_iter < 1000:
            size = random_state.randint(low, high)
            expr = random_expr_grow(self.operators, syms, size, random_state)
            population.add(expr)
            n_iter += 1

        n_iter = 0
        while len(population) < self.population_size and n_iter < 1000:
            size = random_state.randint(low, high)
            expr = random_expr_full(self.operators, syms, size, random_state)
            population.add(expr)
            n_iter += 1

        return population

    def _fit(self, X, y, population, syms, random_state):
        history = list()

        selection_size = int(max(self.ratios.values()) * self.population_size)
        reproduction_size = int(self.population_size * self.ratios["reproduction"])
        subtree_mutation_size = int(
            self.population_size * self.ratios["subtree_mutation"]
        )
        hoist_mutation_size = int(selection_size * self.ratios["hoist_mutation"])
        crossover_size = int(self.population_size * self.ratios["crossover"])

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

            # SELECTION
            index = fitness_vect.nlargest(selection_size).index

            # REPRODUCTION
            reproduction = fitness_vect.nlargest(reproduction_size)
            # start with those who reproduce, unmodified
            population = set(reproduction.index)

            # HOIST MUTATIONS
            for expr in random_state.choice(index, size=hoist_mutation_size):
                hoisted = hoist_mutation(expr, random_state=random_state)
                population.add(hoisted)

            # SUBTREE MUTATIONS
            for expr in random_state.choice(index, size=subtree_mutation_size):
                subtree_mutant = subtree_mutation(
                    expr, self.operators, syms, random_state=random_state,
                )
                population.add(subtree_mutant)

            # CROSSOVERS
            for expr1, expr2 in random_state.choice(index, size=(crossover_size, 2)):
                # child = crossover(expr1, expr2, random_state=random_state)  # FIXME
                child = expr1 + expr2
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
        # fitness = {expr: score(y, y_pred, expr) for expr, y_pred in preds.items()}
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
