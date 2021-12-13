import math
from typing import List

import joblib
import numpy as np
import pandas as pd

# from joblib.parallel import delayed
from sympy import Expr, symbols
from sympy.core.symbol import Symbol
from sympy.utilities.lambdify import lambdify

from .expression import add2, complexity, get_symbols, mul2, mutate, random_expr, sub2


def r2_score(y_pred, y_true):
    correlation_matrix = np.corrcoef(y_pred, y_true)
    correlation = correlation_matrix[0, 1]
    return correlation ** 2


# TODO : joblib.cache this function
def _execute(expr: Expr, X: pd.DataFrame, engine="numexpr"):
    from sympy import Float, lambdify, simplify

    expr = simplify(expr)  # TODO : remove ?
    if isinstance(expr, (float, Float)):  # rare case of oversimplification
        return np.repeat(expr, len(X)).astype(float)

    common_syms = [str(_) for _ in get_symbols(expr)]
    cols = X.columns.intersection(common_syms)
    args = [X[col] for col in cols]
    fn = lambdify(cols, expr, engine)  # TODO : allow
    res = fn(*args)
    return res


class SymbolicRegression:
    def __init__(
        self,
        operators: List[Expr] = [add2, sub2, mul2],
        population_size: int = 100,
        init_size: int = None,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        selection_ratio: float = 0.5,
        memory=joblib.Memory(
            location="~/symkit_data/",
            backend="local",
            mmap_mode=None,  # no mmap --> reload a result vector entirely
            bytes_limit=2 ** 22,  # 4 MegaBytes
        ),
        engine="numexpr",
    ):
        self.population_size = population_size
        self.init_size = init_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        self.selection_ratio = 0.5
        self.expression = None
        self.hall_of_fame = list()
        self.fitness_history = list()
        self.selection_ratio = selection_ratio
        self.memory = memory
        self.engine = engine

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` should be a pandas dataframe")

        syms = symbols(X.columns)
        init_size = self.init_size or len(syms)
        population = [
            random_expr(ops=self.ops, syms=syms, size=init_size)
            for _ in self.population_size
        ]
        parent_size = math.ceil(len(population) * self.selection_ratio)

        max_fit = 0
        n_iter_no_change = 0

        for _ in self.n_iter:
            fitness_vect = self.evaluate_population(population, X, y)
            _max_fit = fitness_vect.max()
            if _max_fit <= max_fit:
                n_iter_no_change += 1
            else:
                max_fit = _max_fit
            if n_iter_no_change >= self.n_iter_no_change:
                break
            history = dict(
                fitness_mean=fitness_vect.mean(),
                fitness_max=fitness_vect.max(),
                expr_complexity=fitness_vect.index.map(complexity).mean(),
            )
            self.fitness_history.append(history)
            parents = fitness_vect.nlargest(parent_size)  # pd.Series
            # TODO : mating
            population = [
                mutate(
                    expr=p, ops=self.operators, size=1, evaluate=True
                )  # TODO size = ?
                for p in parents.index
            ]

        fitness_vect = self.evaluate_population(population, X)
        self.hall_of_fame = fitness_vect.nlargest(parent_size)
        self.expression = self.hall_of_fame.argmax()

    def evaluate_population(self, population: List[Expr], X, y):
        """
        apply a population of sympy expression onto the input data `X`

        This version tries to identify subexpressions, in order to mutualise
        pre-computed results.
        """
        # TODO
        # 1. extract common subexpr via sympy.cse
        # 2. compute those subexpressions
        # 3. cache calls to _execute

        # TODO parallel execution
        results = [_execute(expr, X, engine=self.engine) for expr in population]
        scores = {
            expr: r2_score(result, y) for expr, result in zip(population, results)
        }
        fitness = {
            expr: self.expr_fitness(expr, score) for expr, score in scores.items()
        }
        fitness_vect = pd.Series(fitness)
        return fitness_vect

    def score(self, X, y):
        if not self.expression:
            raise ValueError()
        # TODO : prelambdify
        syms = sorted(map(str, self.expression.find(Symbol)))
        syms = sorted(syms, key=X.columns.tolist().index)
        fn = lambdify(syms, self.expression, self.engine)
        args = [X[sym] for sym in syms]
        y_pred = fn(*args)
        return r2_score(y_pred, y)

    def fitness(self, X, y):
        score = self.score(X, y)
        return self.expr_fitness(self.expr_fitness, score)

    def expr_fitness(self, expression, score):
        # complexity is the number of nodes (internal + leaves)
        # in the formulae
        # this fitness method add complexity as an additional term in the score
        assert 0.0 <= score <= 1.0
        return max(score - (complexity(expression) / 1e3), 0.0)
