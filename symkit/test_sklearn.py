from sklearn.utils.estimator_checks import parametrize_with_checks

from .sre import SymbolicRegression


@parametrize_with_checks([SymbolicRegression()])
def test_estimator_check(estimator, check):
    check(estimator)
