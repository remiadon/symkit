import numpy as np
from sympy import Abs, Function, S, cos, sin
from sympy.abc import x, y


class UserDefinedFunction(Function):
    def vectorized_fn(self, *args):
        raise NotImplementedError(f"You must implement `{self.__name__}`")


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

    def vectorized_fn(self, x, y):
        _len = getattr(x, "__len__", None) or getattr(y, "__len__", None)
        oks = np.abs(y) > 0.001
        ones = np.ones(_len(), dtype=float) if _len else None
        return np.divide(x, y, out=ones, where=oks)


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
            return x.args[1] * log(x.args[0])

    def vectorized_fn(self, x):
        _len = getattr(x, "__len__", None)
        oks = np.abs(x) > 0.001
        zeros = np.zeros(_len(), dtype=float) if _len else None
        return np.log(x, out=zeros, where=oks)


add2 = x + y
sub2 = x - y
mul2 = x * y
div2 = pdiv(x, y)
cos1 = cos(x)
sin1 = sin(x)  # TODO useless ...
