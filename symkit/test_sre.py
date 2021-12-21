import pandas as pd
import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr

from .expression import complexity
from .sre import SymbolicRegression, _execute


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
def mock_joblib(monkeypatch):
    import joblib

    class MemoryPatch:
        def __init__(self, *args, **kwargs):
            pass

        def cache(self, fn):
            return fn

    monkeypatch.setattr(joblib, "Memory", MemoryPatch)
    yield


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


@pytest.mark.parametrize("elimination", [True, False])
def test_evaluate_population(
    body_mass_index, body_mass_candidate_expressions, elimination
):
    population = [parse_expr(e) for e in body_mass_candidate_expressions]
    sre = SymbolicRegression(elimination=elimination)
    sre._execute = _execute  # sklearn prohibits setting attributes at init time
    X, y = body_mass_index
    fitness = sre.evaluate_population(population, X, y)
    pd.testing.assert_series_equal(
        fitness,
        pd.Series([-3138.02, -179.96, -85065064.57, 1.0], index=population),
        atol=0.1,
    )
    evaluated_syms = set.union(*fitness.index.map(lambda e: e.find(Symbol)))
    orig_syms = set.union(*(_.find(Symbol) for _ in population))
    assert evaluated_syms == orig_syms


def test_fit(body_mass_index):
    X, y = body_mass_index
    sre = SymbolicRegression(n_iter=10, population_size=100)
    sre.fit(X, y)
    assert sre.expression_
    assert sre.hall_of_fame_[sre.expression_] == pytest.approx(1.0, 0.1)
    assert complexity(sre.expression_) < 10
