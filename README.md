## Symbolic Regression 101
### About simple ML models
In Machine Learning we often borrow models from an extensive literature.
In the case of the Linear Regression we have a **fixed structure**, and the training phase (calling `fit`) consists in finding the coefficients to the equation
In Python we would end up with a decision
```
    X = # get data to predict on
    W = [1.0000000000000002, 1.9999999999999991]
    b = 3.0000000000000018

    predictions =  b + sum(X[i] * wi for i, wi in enumerate(W))
```
In this scenario we learned reuse the coefficients (learned) but some structural aspect will never change
 1) we **multiply** every input feature by its corresponding coefficient
 2) we aggregate resulting values using a `sum` function
 3) we `add` the bias 

Of course we have access to a variety of more complex, non-linear models out there, but they are usually harder to explain ...

### Where does Symbolic Regression stand then ?
[Symbolic Regression](https://en.wikipedia.org/wiki/Symbolic_regression#:~:text=Symbolic%20regression%20(SR)%20is%20a,regression%20to%20represent%20a%20function.) aims at learning not only the (potential) coefficients but first and foremost the structure of the model.
The resulting relaxed model consists in polynomial, build by composition from 
 - our input variables, eg. `Age` and `Height`
 - a set of operator at hand, eg. addition, substraction, multiplication and division

Symbolic Regression runs `Genetic Programming` in order to produce the N bests polynomials for your data.

### What pros can you expect compared to other ML methods ?
- built-in `explainability` along with non-linearity : if you understand all the operators you provide the algorithm with, you understand all potential polynomials derived from these operators
- built-in feature selection & model-size reduction : when passing existing traits to a new offspring, we take the polynomial complexity in consideration (eq. to the model size in a [Minimum Description Length](https://en.wikipedia.org/wiki/Minimum_description_length) setting). We then build a `skyline` based on both performance and complexity, and only consider models on this skyline. This process efficiently avoids "bloating" (complex individuals will not mate on the next round), controls overfitting (the simplest the polynomial, the more likely it does generalize to new data) and embeds explainability in the learning process
- easier transfer learning : it's fairly easy to extract the list of learned polynomials from a Symbolic Regression model and instantiates a new one with this list, to solve a new but similar problem

### cons ?
 - Genetic Algorithms usually come with a slow learning process. This has been mitigated through the abusive use of [polars]([url](https://docs.pola.rs/py-polars/html/reference/))

## TODO
 - add unary operators like sin and cos [X]
 - try an online version with RiverML [ ]
 - pareto front optimisation          [X]
   - multi-objective optimisation (eg. return vs risk ) ?   [ ]
 - SymbolicRegression for symbolic regression on timeseries  [ ]
 - scikit-learn tree structure into a sympy expression (that can be translated into a polars expression later on)
