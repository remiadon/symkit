import polars as pl
from sympy import Abs, Add, Function
from sympy import Mul as mul
from sympy import S, cos, sin
from sympy.abc import x, y

pl_abs = (
    lambda x: pl.when(x >= 0).then(x).otherwise(-x)
)  # .abs is not defined in polars expression


class UserDefinedFunction(Function):
    def polars(self, *args):
        raise NotImplementedError(
            f"You must implement the polars equivalent for this function"
        )


class pdiv(UserDefinedFunction):
    @classmethod
    def eval(cls, x, y):
        if y.is_real:
            if Abs(y) > 0.001:
                return x / y
            else:
                return S.One
        if x == y:
            return S.One
        if x == S.Zero:
            return x
        if y.is_Mul and x in y.args:  # eg. pdiv(a, a * b) -> pdiv(1, b)
            return pdiv(1, y / x)
        if x.is_Mul and y in x.args:  # eg. pdiv(4 * a, a) --> 4
            return x / y

        if isinstance(x, pdiv) and isinstance(y, pdiv):
            a, b = x.args
            c, d = y.args
            return pdiv(a * d, b * c)
        if isinstance(x, pdiv):
            a, b = x.args
            return pdiv(a, y * b)
        if isinstance(y, pdiv):
            a, b = y.args
            return pdiv(x * b, a)

    def __mul__(self, other):
        if isinstance(other, pdiv):  # pdiv(1, X1) * pdiv(X1, X2) = pdiv(1, X2**2)
            return pdiv(self.args[0] * other.args[0], self.args[1] * other.args[1])
        return super().__mul__(other)

    def polars(self, x: pl.Expr, y: pl.Expr):
        return pl.when(pl_abs(y) > 0.001).then(x / y).otherwise(pl.lit(1))


class plog(UserDefinedFunction):
    @classmethod
    def eval(cls, x):
        from sympy import log

        _x = Abs(x)
        if _x.is_real:
            if _x > 0.001:
                return log(_x)
            else:
                return S.Zero
        if x.is_Pow:  # log(M ** k) = k * log(M)
            #  see https://github.com/sympy/sympy/issues/13781
            return x.args[1] * plog(x.args[0])

    def polars(self, x: pl.Expr):
        return pl.when(pl_abs(x) > 0.001).then(x.log()).otherwise(0)


add2 = x + y
sub2 = x - y
div2 = pdiv(x, y)
mul2 = x * y
cos1 = cos(x)
sin1 = sin(x)
