import pandas as pd
import pytest
from sympy.parsing.sympy_parser import parse_expr

from .sre import SymbolicRegression, r2_score


@pytest.fixture
def body_mass_index():
    X = pd.DataFrame(
        dict(
            height=[1.78, 1.65, 1.83, 1.70],  # in meters
            weight=[70, 75, 86, 72],  # in kilograms
        )
    )
    y = X.weight / (X.height ** 2)
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
        ("height * weight + 2", 0.026),
        ("height - 3", 0.26),
        ("(height * weight) ** 2", 0.027),
        ("weight / (height ** 2)", 1.0),  # body mass index forumlae
    ],
)
def test_score(expr, score, body_mass_index):
    # test scoring on data corresponding to the body mass index formulae

    sre = SymbolicRegression(memory=None)
    X, y = body_mass_index
    sre.expression = parse_expr(expr)
    sc = sre.score(X, y)
    assert sc == pytest.approx(score, 0.05)


def test_evaluate_population(body_mass_index):
    population = [
        "height * weight + 2",
        "height - 3",
        "(height * weight) ** 2",
        "weight / (height ** 2)",
    ]
    population = [parse_expr(e) for e in population]
    sre = SymbolicRegression(memory=None)
    X, y = body_mass_index
    fitness = sre.evaluate_population(population, X, y)
    pd.testing.assert_series_equal(
        fitness, pd.Series([0.0, 0.25, 0.01, 1.0], index=population), atol=0.1,
    )
