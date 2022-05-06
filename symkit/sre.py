from typing import List, Mapping

import numpy as np
import pandas as pd
import polars as pl

# from joblib.parallel import delayed
import sympy as sp
from river.utils import Skyline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_random_state

from .core import sympy_to_polars, sympy_to_str
from .expression import (
    complexity,
    crossover,
    random_expr_full,
    random_expr_grow,
    subtree_mutation,
)
from .metrics import pl_r2_score
from .operators import add2, div2, mul2, sub2
from .population import Population


def evaluate_population(population: List[sp.Expr], X: pl.DataFrame, y):
    """
    1. converts all sympy expressions within `population` to polars.Expr instances
    2. alias all resulting polars.Expr as their original sympy equivalent
    3. call X.compute() of all expressions to get predictions
    4. score those predictions given `y`
    """
    pl_expressions = list()
    for expr in population:
        pl_expr = sympy_to_polars(expr)
        pl_expressions.append(pl_expr.alias(str(expr)))

    preds = X.select(pl_expressions)
    fitness = pl_r2_score(preds, y)
    res = fitness.to_pandas().T.loc[:, 0]
    res.index = list(population)
    return res


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        operators: List[sp.Expr] = (add2, sub2, mul2, div2),
        population_size: int = 100,
        n_iter: int = 20,
        n_iter_no_change: int = None,
        init_size=(2, 8),
        crossover_ratio=0.4,
        subtree_mutation_ratio=0.3,
        random_state=12,
    ):
        assert population_size
        self.population_size = population_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        assert isinstance(init_size, tuple) and len(init_size) == 2
        self.init_size = init_size
        assert sum((crossover_ratio, subtree_mutation_ratio)) < 0.9
        self.crossover_ratio = crossover_ratio
        self.subtree_mutation_ratio = subtree_mutation_ratio
        # self.hoist_mutation_ratio = hoist_mutation_ratio
        self.random_state = random_state

    def _check_input(self, X, y=None):
        if not isinstance(X, pl.DataFrame):
            if y is not None:
                X, y = check_X_y(X, y)  # TODO : check parameter setting for `check_X_y`
            else:
                X = check_array(X)
            X = pl.DataFrame(X, columns=["X" + str(i) for i in range(X.shape[1])])
        y = pl.Series(y)
        return X, y

    def fit(self, X, y):
        X, y = self._check_input(X, y)
        random_state = check_random_state(self.random_state)
        syms = sp.symbols(X.columns)
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

    def prefit(self, syms, random_state) -> Population:
        # intiliaze to random
        # use a set for deduplication
        population = Population()
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

        # paretoset kept up to date
        pareto_archive = Skyline(maximize=["score"], minimize=["complexity"])

        # keep evaluation in memory, this is costless and can greatly improve runtime
        evals = dict()

        pop_size = self.population_size
        subtree_mutation_size = int(pop_size * self.subtree_mutation_ratio)
        crossover_size = int(pop_size * self.crossover_ratio)
        reproduction_size = pop_size - subtree_mutation_size - crossover_size
        selection_size = max(subtree_mutation_size, crossover_size)

        max_fit = np.inf  # FIXME works with r2_score
        n_iter_no_change = 0
        if self.n_iter_no_change is None:
            _n_iter_no_change = max(3, self.n_iter // 4)
        else:
            _n_iter_no_change = self.n_iter_no_change

        for _ in range(self.n_iter):
            to_eval = population - evals.keys()  # avoid recomputing
            fitness_vect = evaluate_population(to_eval, X, y)
            fitness_vect = pd.Series(
                {**fitness_vect, **{k: v for k, v in evals.items() if k in population}}
            )
            complexities = fitness_vect.index.map(complexity).astype(float)
            evals.update(fitness_vect)
            _max_fit = fitness_vect.max()
            history_payload = dict(
                fitness_median=fitness_vect.median(),
                fitness_max=_max_fit,
                expr_complexity_mean=np.mean(complexities),
            )
            history.append(history_payload)
            if _max_fit < max_fit:
                n_iter_no_change += 1
            else:
                max_fit = _max_fit
            if n_iter_no_change >= _n_iter_no_change:
                break

            # SELECTION
            index = fitness_vect.nlargest(selection_size).index

            # REPRODUCTION
            reproduction = fitness_vect.nlargest(reproduction_size)
            # start with those who reproduce, unmodified
            population = Population(reproduction.index)

            # SUBTREE MUTATIONS
            ctr = 0  # use a counter to limit the tries on new expression creation
            while (
                len(population) < len(reproduction) + subtree_mutation_size
                and ctr < 1000
            ):
                expr = random_state.choice(index)
                subtree_mutant = subtree_mutation(
                    expr, self.operators, syms, random_state=random_state,
                )
                population.add(subtree_mutant)
                ctr += 1

            # CROSSOVERS
            # 1. update the pareto frontier
            for _ in zip(fitness_vect.index, fitness_vect.values, complexities.values):
                payload = dict(zip(("expr", "score", "complexity"), _))
                pareto_archive.update(payload)

            # 2. use pareto exprs for breeding
            ctr = 0
            while len(population) < pop_size and ctr < 1000:
                pareto_expr = random_state.choice(pareto_archive)["expr"]
                expr = random_state.choice(index)
                # child = pareto_expr + expr
                child = crossover(
                    donor=pareto_expr, receiver=expr, random_state=random_state
                )
                population.add(child)
                ctr += 1

        self.symbols_ = syms
        self.history_ = history
        self.hall_of_fame_ = pd.Series(evals).nlargest(self.population_size)
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
        return X.select(pl_fn).to_pandas()

    # TODO predict_proba from (mean) sigmoid from self.expression | self.hall_of_fame

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
