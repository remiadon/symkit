# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from symkit.sre import SymbolicRegression
from symkit.expression import complexity
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class BostonSuite:
    """"""
    params = [10, 50, 100]
    param_names = ["population_size"]
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    def setup(self, population_size):
        pass
        
    def time_fit(self, population_size):
        sre = SymbolicRegression(population_size=population_size)
        sre.fit(self.X_train, self.y_train)

    def mem_fit(self, population_size):
        sre = SymbolicRegression(population_size=population_size)
        sre.fit(self.X_train, self.y_train)

    def track_r2_score(self, population_size):
        sre = SymbolicRegression(population_size=population_size)
        sre.fit(self.X_train, self.y_train)
        y_pred = sre.predict(self.X_test)
        return r2_score(y_pred, self.y_test)

    def track_complexity(self, population_size):
        sre = SymbolicRegression(population_size=population_size)
        sre.fit(self.X_train, self.y_train)
        return complexity(sre.expression_)
