import pandas as pd
import polars as pl
import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

from ..expression import complexity
from ..sre import SymbolicRegression, evaluate_population


@pytest.fixture
def body_mass_index():
    X = pl.DataFrame(
        dict(
            height=[1.78, 1.65, 1.83, 1.70],  # in meters
            weight=[70, 75, 86, 72],  # in kilograms
        )
    )
    y = X["weight"] / (
        X["height"] ** 2
    )  # beware, in polars `height` is a dataframe attribute
    return X, y


@pytest.fixture
def body_mass_candidate_expressions():
    return [
        "height * weight + 2",
        "height - 3",
        "(height * weight) ** 2",
        "weight / (height ** 2)",  # True formulae
    ]


@pytest.mark.parametrize(
    "expr,score",
    [
        ("height * weight + 2", -3138.02),
        ("height - 3", -179.96),
        ("(height * weight) ** 2", -85065064.57),
        ("weight / (height ** 2)", 1.0),  # body mass index forumlae
    ],
)
def test_score(expr, score, body_mass_index):
    # test scoring on data corresponding to the body mass index formulae
    sre = SymbolicRegression()
    X, y = body_mass_index
    sre.expression_ = parse_expr(expr)
    sc = sre.score(X, y)
    assert sc == pytest.approx(score, 0.05)


def test_evaluate_population(body_mass_index, body_mass_candidate_expressions):
    population = [parse_expr(e) for e in body_mass_candidate_expressions]
    X, y = body_mass_index
    fitness = evaluate_population(population, X, y)
    expected = pd.Series([-3138.02, -179.96, -85065064.57, 1.0], index=population)
    expected = expected[fitness.index]
    pd.testing.assert_series_equal(fitness, expected, atol=0.1, check_names=False)
    evaluated_syms = set.union(*fitness.index.map(lambda e: e.find(Symbol)))
    orig_syms = set.union(*(_.find(Symbol) for _ in population))
    assert evaluated_syms == orig_syms


@pytest.mark.parametrize("n_iter", [5, 10])
@pytest.mark.parametrize("population_size", [100])
def test_fit(body_mass_index, n_iter, population_size):
    from ..operators import add, mul, pdiv, sub

    X, y = body_mass_index
    sre = SymbolicRegression(
        n_iter=n_iter,
        population_size=population_size,
        random_state=12,
        operators=[add, sub, pdiv, mul],
        init_size=(2, 6),
        p_float=0.0,
    )
    sre.fit(X, y)
    assert sre.expression_
    assert sre.score(X, y) == pytest.approx(1.0, 0.1)
    assert complexity(sre.expression_) < 12
    assert sre.hall_of_fame_.shape[0] == population_size
