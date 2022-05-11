"""
A helper class to represent a population of sympy expressions
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import sympy as sp

from .expression import tree_distance


class Population(set):
    def __init__(self, s=(), population_n_iter_limit=None):
        for element in s:
            self.add(element)

    @classmethod
    def _wrap_methods(cls, names):
        def wrap_method_closure(name):
            def inner(self, *args):
                result = getattr(super(cls, self), name)(*args)
                return result

            inner.fn_name = name
            setattr(cls, name, inner)

        for name in names:
            wrap_method_closure(name)

    def add(self, element: sp.Expr):
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
        # if element.has(sp.Rational):
        #    print(f"avoiding {element} because it contains a Rational")
        super().add(element)


def tree_distances(population: Population):
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


Population._wrap_methods(
    [
        "__ror__",
        "difference_update",
        "__isub__",
        "symmetric_difference",
        "__rsub__",
        "__and__",
        "__rand__",
        "intersection",
        "difference",
        "__iand__",
        "union",
        "__ixor__",
        "symmetric_difference_update",
        "__or__",
        "copy",
        "__rxor__",
        "intersection_update",
        "__xor__",
        "__ior__",
        "__sub__",
    ]
)
