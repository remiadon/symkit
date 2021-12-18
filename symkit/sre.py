import random
from typing import AnyStr, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
from sklearn.metrics import mean_squared_error, r2_score

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
def _execute(expr: Expr, X: pd.DataFrame, engine="numpy"):
    from sympy import Float, Integer, lambdify, simplify

    expr = simplify(expr)  # TODO : remove ?
    if expr.is_number:
        return np.repeat(float(expr), len(X)).astype(float)

    common_syms = [str(_) for _ in get_symbols(expr)]
    cols = X.columns.intersection(common_syms)
    args = [X[col].values for col in cols]
    modules = [engine, dict(pdiv=pdiv)]
    fn = lambdify(cols, expr, modules)
    res = fn(*args)
    return res


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


class SymbolicRegression:
    def __init__(
        self,
        operators: List[Expr] = [add2, sub2, mul2, div2],
        population_size: int = None,
        init_size: int = None,
        n_iter: int = 100,
        n_iter_no_change: int = 10,
        ratios=dict(
            reproduction=0.4, crossover=0.2, subtree_mutation=0.2, hoist_mutation=0.2
        ),
        memory=joblib.Memory(
            location="~/symkit_data/",
            backend="local",
            mmap_mode=None,  # no mmap --> reload a result vector entirely
            bytes_limit=2 ** 22,  # 4 MegaBytes
        ),
        engine: AnyStr = "numpy",
    ):
        self.population_size = population_size
        self.init_size = init_size
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.operators = operators
        self.selection_ratio = 0.5
        self.expression = None
        self.hall_of_fame = set()
        self.history = list()
        assert isinstance(ratios, dict) and sum(ratios.values()) == 1.0
        self.ratios = ratios
        self.memory = memory
        self.engine = engine

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` should be a pandas dataframe")

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
            population.add(random_expr(ops=self.operators, syms=syms, size=init_size))

        # parent_size = math.ceil(len(population) * self.selection_ratio)

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
            self.history.append(history_payload)

            # REPRODUCTIONS
            reproduction = fitness_vect.nlargest(reproduction_size)
            population = set(reproduction.index)  # start with those who reproduce

            # HOIST MUTATIONS
            target_size = reproduction_size + hoist_mutation_size
            for expr in pick(population.__len__, target_size, reproduction.index):
                hoisted = hoist_mutation(expr)
                population.add(hoisted)

            # SUBTREE MUTATIONS
            target_size += subtree_mutation_size
            for expr in pick(population.__len__, target_size, reproduction.index):
                n_symbols = expr.count(Symbol)
                mutation_size = np.ceil(np.log(n_symbols))
                subtree_mutant = subtree_mutation(
                    expr, self.operators, syms, size=mutation_size
                )
                population.add(subtree_mutant)

            # CROSSOVERS
            target_size += crossover_size
            for expr1, expr2 in pick(
                population.__len__, target_size, reproduction.index, k=2
            ):
                child = crossover(expr1, expr2)
                population.add(child)

        self.hall_of_fame = self.evaluate_population(population, X, y)
        self.expression = self.hall_of_fame.idxmax()
        return self

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
        fitness = {expr: self.score(X, y, expr=expr) for expr in population}
        fitness_vect = pd.Series(fitness)
        return fitness_vect

    def score(self, X, y, expr=None):
        if expr is None:
            expr = self.expression
        if complexity(expr) > 10:
            return -np.inf
        if expr is None:
            raise ValueError()
        y_pred = _execute(expr, X, engine=self.engine)
        return -mean_squared_error(y_pred, y, squared=False)

    def fitness(self, X, y, expr=None):
        expr = expr or self.expression
        return self.score(X, y, expr=expr) - (complexity(expr) / 100)
