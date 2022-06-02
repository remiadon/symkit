from typing import FrozenSet, Iterable, List

import numpy as np
import pandas as pd
import polars as pl

# from joblib.parallel import delayed
import sympy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state

from .core import sympy_to_polars
from .metrics import pl_r2_score
from .operators import add2, div2, mul2, sub2
from .population import get_next_generation, populate


def evaluate_population(population: Iterable[sp.Expr], X: pl.DataFrame, y):
    """
    1. converts all sympy expressions within `population` to polars.Expr instances
    2. alias all resulting polars.Expr as their indices
    3. call X.compute() of all expressions to get predictions
    4. score those predictions given `y`
    """
    pl_expressions = list()
    for idx, expr in enumerate(population):
        pl_expr = sympy_to_polars(expr)
        pl_expressions.append(pl_expr.alias(str(idx)))

    preds = X.select(pl_expressions)
    fitness = pl_r2_score(preds, y)
    res = fitness.to_pandas().T.loc[:, 0]
    res.index = list(population)
    return res


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        bases: List[sp.Expr] = (add2, sub2, mul2, div2),
        population_size: int = 100,
        n_iter: int = 20,
        n_iter_no_change: int = None,
        init_size=(2, 6),
        crossover_ratio=0.4,
        subtree_mutation_ratio=0.3,
        p_float=0.0,
        random_state=12,
    ):
        assert population_size
        self.population_size = population_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.bases = bases
        assert isinstance(init_size, tuple) and len(init_size) == 2
        self.init_size = init_size
        assert sum((crossover_ratio, subtree_mutation_ratio)) < 0.9
        self.crossover_ratio = crossover_ratio
        self.subtree_mutation_ratio = subtree_mutation_ratio
        # self.hoist_mutation_ratio = hoist_mutation_ratio
        self.random_state = random_state
        self.p_float = p_float
        self.init_size = init_size

    def _check_input(self, X, y=None):
        if not isinstance(X, pl.DataFrame):
            if y is not None:
                X, y = check_X_y(X, y)  # TODO : check parameter setting for `check_X_y`
            else:
                X = check_array(X)
            if isinstance(X, pd.DataFrame):
                X = pl.from_pandas(X)
            else:
                X = pl.DataFrame(X, columns=["X" + str(i) for i in range(X.shape[1])])
        y = pl.Series(y)
        return X, y

    def fit(self, X, y):
        X, y = self._check_input(X, y)
        self.random_state = check_random_state(self.random_state)

        self.symbols_ = sp.symbols(X.columns)

        first_generation = populate(
            self.bases,
            self.symbols_,
            random_state=self.random_state,
            population_size=self.population_size,
            expression_size_bounds=self.init_size,
            p_float=self.p_float,
        )
        return self._fit(X, y, first_generation)

    def _fit(self, X, y, population):
        history = list()

        max_fit = -np.inf  # FIXME works with r2_score
        n_iter_no_change = 0
        if self.n_iter_no_change is None:
            _n_iter_no_change = max(3, self.n_iter // 4)
        else:
            _n_iter_no_change = self.n_iter_no_change

        for _ in range(self.n_iter):
            fitness = evaluate_population(population, X, y)
            _max_fit = fitness.max()
            history_payload = dict(
                fitness_median=fitness.median(),
                fitness_max=fitness.max(),
                # expr_complexity_mean=np.mean(complexities),
            )
            history.append(history_payload)
            if _max_fit < max_fit:
                n_iter_no_change += 1
            else:
                max_fit = _max_fit
            if n_iter_no_change >= _n_iter_no_change:
                break

            population: FrozenSet[sp.Expr] = get_next_generation(
                fitness,
                self.bases,
                self.symbols_,
                self.random_state,
                self.crossover_ratio,
                self.subtree_mutation_ratio,
                size=self.population_size,
                p_float=self.p_float,
            )

        self.history_ = history
        self.hall_of_fame_ = fitness
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

    def predict(self, X):
        expr = getattr(self, "expression_", None)
        if expr is None:
            raise NotFittedError()
        X, _ = self._check_input(X, y=None)
        pl_fn = sympy_to_polars(expr)
        res = X.select(pl_fn).to_numpy().reshape(-1)
        if len(res) < X.shape[0]:
            res = res.repeat(X.shape[0] // len(res))
        return res

    # TODO predict_proba from (mean) sigmoid from self.expression | self.hall_of_fame

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
