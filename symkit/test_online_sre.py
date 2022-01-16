import itertools

import pandas as pd
import pytest

from .online_sre import OnlineSymbolicRegression


@pytest.fixture
def data():
    from river import datasets

    return datasets.Bikes()


def test_learn_one(data):
    sre = OnlineSymbolicRegression(init_size=(2, 8), population_size=10)
    for X, y in itertools.islice(data, 1000):
        sre.learn_one(X, y)
    assert isinstance(sre.hall_of_fame, pd.Series)
    assert sre.expression is not None
    assert len(sre) == 999  # first one is for instantiation
    assert not sre.hall_of_fame.hasnans
