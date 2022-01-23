from typing import List, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state
from symengine import lambdify

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


def _execute_postprocess(res, size=None):
    res = np.asfarray(res).reshape(-1)
    if res.shape[0] != size:
        return np.repeat(res[0], size)
    nans = np.isnan(res) | (res == np.inf) | (res == -np.inf)
    if nans.any():  # FIXME this happened with expr=X9**(-X10)
        res[nans] = 10_000_000  # FIXME set arbitrary high value ?
    return res


def _execute_udf(expr: Expr, X: Mapping, symbol_gen):
    from sympy import nan, oo, zoo

    n_samples = max(map(len, X.values()))

    if expr.is_Number:
        return np.repeat(float(expr), n_samples)  # TODO : return a scalar

    if expr.is_Symbol:
        return X[expr]

    if expr.has(oo, -oo, zoo, nan):
        return np.zeros(n_samples, dtype=float)  # TODO log this

    if isinstance(expr, UserDefinedFunction):
        # recursive calls to still rely on lambdify in subexpressions
        args = list()
        for _ in expr.args:
            args.append(_execute_udf(_, X, symbol_gen))
        return expr.vectorized_fn(*args)  # eg. protected div

    syms = list(expr.free_symbols)

    user_defined = expr.find(UserDefinedFunction)
    if user_defined:
        replace_dict = dict()
        for _ in user_defined:
            res = _execute_udf(_, X, symbol_gen)
            var = next(symbol_gen)
            X[var] = res
            replace_dict[_] = var
        expr = expr.subs(replace_dict)

    syms, args = zip(*X.items())

    fn = lambdify(syms, [expr], order="F")

    res = fn(args)

    return res.reshape(-1)  # make sure this is a vector


def score(y, y_pred, expr):
    return r2_score(y, y_pred) - complexity(expr) / 1000


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators: List[Expr] = (add2, sub2, mul2, div2),
        population_size: int = 100,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        init_size=(2, 8),
        crossover_ratio=0.4,
        subtree_mutation_ratio=0.3,
        hoist_mutation_ratio=0.2,
        random_state=12,
    ):
        assert population_size
        self.population_size = population_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        assert isinstance(init_size, tuple) and len(init_size) == 2
        self.init_size = init_size
        assert (
            sum((crossover_ratio, subtree_mutation_ratio, hoist_mutation_ratio)) < 0.9
        )
        self.crossover_ratio = crossover_ratio
        self.subtree_mutation_ratio = subtree_mutation_ratio
        self.hoist_mutation_ratio = hoist_mutation_ratio
        self.random_state = random_state

    def _check_input(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            if y is not None:
                X, y = check_X_y(X, y)  # TODO : check parameter setting for `check_X_y`
            else:
                X = check_array(X)
            syms = symbols(f"X:{X.shape[1]}")
            X = pd.DataFrame(X, columns=list(syms))
        else:
            X = X.copy(deep=False)  # do not modify the original df
            X.columns = X.columns.map(Symbol)
        return X, y

    def fit(self, X, y):
        X, y = self._check_input(X, y)
        random_state = check_random_state(self.random_state)
        syms = X.columns.tolist()
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

        pop_size = self.population_size
        subtree_mutation_size = int(pop_size * self.subtree_mutation_ratio)
        hoist_mutation_size = int(pop_size * self.hoist_mutation_ratio)
        crossover_size = int(pop_size * self.crossover_ratio)
        reproduction_size = (
            pop_size - subtree_mutation_size - hoist_mutation_size - crossover_size
        )
        selection_size = max(subtree_mutation_size, hoist_mutation_size, crossover_size)

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
                fitness_median=fitness_vect.median(),
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
        """  # TODO : define this function outside of this class
        from .operators import UserDefinedFunction

        sym_gen = symbol_generator()
        udfs = {
            expr
            for expr in population
            if expr.has(UserDefinedFunction) or expr.is_Number
        }
        pures = list(set(population) - udfs)

        # execute user defined functions in a purely iterative way, to avoid risks
        preds = dict()
        _X = dict(zip(X.columns.tolist(), X.values.T))
        for expr in udfs:  # for loop instead of dictcomp for easier debugging
            ex = _execute_udf(expr, _X, sym_gen)
            preds[expr] = ex

        if pures:
            pure_syms = list(set.union(*(_.free_symbols for _ in pures)))
            pure_fn = lambdify(pure_syms, pures)
            _X = X[pure_syms].values
            pure_preds = pure_fn(_X)
            pure_preds = dict(zip(pures, pure_preds))
            preds.update(pure_preds)

        preds = {
            expr: _execute_postprocess(pred, size=X.shape[0])
            for expr, pred in preds.items()
        }
        fitness = {expr: score(y, y_pred, expr) for expr, y_pred in preds.items()}
        return pd.Series(fitness)

    def predict(self, X):
        expr = getattr(self, "expression_", None)
        if expr is None:
            raise NotFittedError()
        X, _ = self._check_input(X, y=None)
        if expr.has(UserDefinedFunction) or expr.is_Number:
            _X = dict(zip(X.columns.tolist(), X.values.T))
            y_pred = _execute_udf(expr, _X, symbol_gen=symbol_generator())
        else:
            syms = list(expr.free_symbols)
            fn = lambdify(syms, [expr])
            y_pred = fn(X[syms])
        y_pred = _execute_postprocess(y_pred, size=X.shape[0])
        return y_pred

    # TODO predict_proba from (mean) sigmoid from self.expression | self.hall_of_fame

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
