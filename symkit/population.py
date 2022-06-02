"""
A helper class to represent a population of sympy expressions
"""
from collections import defaultdict
from typing import FrozenSet, List, NewType

import pandas as pd
import sympy as sp
from river.utils import Skyline

from .expression import (
    complexity,
    crossover,
    random_expr_full,
    random_expr_grow,
    subtree_mutation,
    tree_distance,
)
from .operators import pdiv

Population = NewType("Population", FrozenSet[sp.Expr])  # TODO frozenset


def tree_distances(population: Population) -> pd.DataFrame:
    distances = defaultdict(list)
    cache = dict()
    index = list(population)
    for element in index:
        universe = population - {element}
        for cand in universe:
            key = frozenset((element, cand))
            dist = cache.get(key)
            if dist is None:
                dist = tree_distance(element, cand)
            cache[key] = dist
            distances[element].append(dist)
    return pd.DataFrame(distances)


def add_in(population: Population, element: sp.Expr) -> None:
    powers = element.find(sp.Pow)
    for power in powers:
        if not power.args[1].is_Number:
            print(
                f"""
                {element} not added in population,
                it contains {power} for which the exponent should be an int
            """
            )
        return
    if element.count(pdiv) > 2:  # FIXME, simplify complex exprs with many pdiv
        return
    population.add(element)


def populate(
    operators: List[sp.Function],
    symbols: List[sp.Symbol],
    random_state,
    population_size=100,
    expression_size_bounds=(2, 8),
    p_float=0.2,
) -> Population:
    n_grow = population_size // 2
    pop = set()
    min_size, max_size = expression_size_bounds

    n_iter = 0
    while len(pop) < n_grow and n_iter < 1000:
        size = random_state.randint(min_size, max_size)
        expr = random_expr_grow(operators, symbols, size, random_state, p_float=p_float)
        add_in(pop, expr)
        n_iter += 1

    n_iter = 0
    while len(pop) < population_size and n_iter < 1000:
        size = random_state.randint(min_size, max_size)
        expr = random_expr_full(operators, symbols, size, random_state, p_float=p_float)
        add_in(pop, expr)
        n_iter += 1

    return frozenset(pop)


def get_next_generation(
    fitness: pd.Series,
    bases: List[sp.Expr],
    symbols: List[sp.Symbol],
    random_state,
    crossover_ratio=0.4,
    subtree_ratio=0.3,
    size=None,
    compute_tree_dists=False,
    p_float=0.2,
) -> Population:
    if size is None:
        size = len(fitness)
    assert subtree_ratio <= crossover_ratio <= 0.5, "ratios should be lower than 0.5"
    reproduction_len = max(int((1.0 - subtree_ratio - crossover_ratio) * size), 1)
    subtree_len = int(subtree_ratio * size)

    bests = fitness.nlargest(reproduction_len).index
    new_gen = set(bests)

    ctr = 0
    while ctr < 1000 and len(new_gen) < reproduction_len + subtree_len:
        expr = random_state.choice(bests)
        mutant = subtree_mutation(
            expr, bases, symbols, random_state=random_state, p_float=p_float
        )
        add_in(new_gen, mutant)
        ctr += 1

    ctr = 0
    pareto_df = pd.DataFrame(
        dict(
            expr=fitness.index, score=fitness, complexity=fitness.index.map(complexity)
        )
    )
    to_maximize = ["score"]
    if compute_tree_dists:
        distances = (
            tree_distances(set(fitness.index)).mean().round(1)
        )  # round(1) --> max of 10 buckets
        pareto_df.loc[:, "treedist_mean"] = distances
        to_maximize.append("treedist_mean")
    pareto = Skyline(minimize=["complexity"], maximize=to_maximize)
    for d in pareto_df.to_dict("records"):
        pareto.update(d)

    pareto_exprs = [_["expr"] for _ in pareto]
    receivers = bests.difference(pareto_exprs)
    if receivers.empty:
        receivers = bests
    while ctr < 1000 and len(new_gen) < size:
        pareto_expr = random_state.choice(pareto_exprs)
        receiver = random_state.choice(receivers)
        child = crossover(
            donor=pareto_expr, receiver=receiver, random_state=random_state
        )
        add_in(new_gen, child)
        ctr += 1

    return frozenset(new_gen)
