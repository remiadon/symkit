"""
symykit is an attempy to bridge the gap between two awesome python packages : scikit-learn and Sympy

The goal is to treat Machine Learning problems with Genetic Programming methodologies,
and converge toward a model that is:
 - explainable : it consits of an arithmetic expression. If you can understand a set of operators, you can probably understand a composition of some of its elements.
 - statistically efficient : arithmetic expressions can contain non-linear relationships, and get close to an optimal solution for you problem
 - runtime efficient : genetic algorithms are know to explore an enormous search space. Symkit will "reduce" it by applying optimizations on the expressions (via SymPy)
 - memory efficient : with different caching strategies
"""

__version__ = "0.0.1"
