#!/usr/bin/python3

import abc
import math
import numpy as np
import numpy.linalg as LA
import random
import scipy.optimize


class TargetFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p):
        pass

    @abc.abstractmethod
    def grad(self, p):
        pass

    @abc.abstractstaticmethod
    def quad(self, p):
        pass

    def __rmul__(self, c):
        class MulFunction(TargetFunction):
            def __init__(self, f, c):
                self.f = f
                self.c = c

            def __call__(self, p):
                return self.c * self.f(p)

            def grad(self, p):
                return self.c * self.f.grad(p)

            def quad(self, p):
                return self.c * self.f.quad(p)
        return MulFunction(self, c)

    def __add__(self, other):
        class SumFunction(TargetFunction):
            def __init__(self, l, r):
                self.l = l
                self.r = r

            def __call__(self, p):
                return self.l(p) + self.r(p)

            def grad(self, p):
                return self.l.grad(p) + self.r.grad(p)

            def quad(self, p):
                return self.l.quad(p) + self.r.quad(p)
        return SumFunction(self, other)


class ConstantFunction(TargetFunction):
    def __init__(self, c, n):
        self.c = c
        self.n = n

    def __call__(self):
        return self.c

    def grad(self, p):
        return np.array([0] * self.n)

    def quad(self, p):
        return np.array([[0] * self.n for i in range(self.n)])


class Restriction:
    def __init__(self, a, b):
        """<a, x> + b <= 0
        """
        self.a = a
        self.b = b

    def evaluate(self, x):
        return np.matmul(self.a, x) + b >= 0

    def log_function(self):
        class LogFunction(TargetFunction):
            def __init__(self, r):
                self.r = r

            def __call__(self, p):
                return -math.log(-(np.matmul(self.r.a, p) + self.r.b))

            def grad(self, p):
                return np.array([-self.r.a[i] / (np.matmul(self.r.a, p) + self.r.b) for i in range(len(p))])

            def quad(self, p):
                n = len(self.r.a)
                return np.array([[self.r.a[i] * self.r.a[j] / (np.matmul(self.r.a, p) + self.r.b) ** 2 for j in range(n)] for i in range(n)])
        return LogFunction(self)


class NoInnerPoint(Exception):
    def __str__(self):
        return "No inner point of restrictons found"


def try_inner_point_of_restrictions(restrictions, v):
    n = len(restrictions[0].a)
    res = scipy.optimize.linprog(
        np.array(restrictions[0].a),
        A_ub = np.array([r.a for r in restrictions]),
        b_ub = np.array([-r.b - v for r in restrictions])
    )
    if res.success:
        return res.x
    else:
        raise NoInnerPoint()


def inner_point_of_restrictions(restrictions):
    l, r = 0, 1
    for i in range(100):
        mid = (l + r) / 2
        f = True
        try:
            try_inner_point_of_restrictions(restrictions, mid)
        except NoInnerPoint:
            f = False
        if f:
            l = mid
        else:
            r = mid
    if l == 0:
        raise NoInnerPoint()
    return try_inner_point_of_restrictions(restrictions, l)


class InnerPointMethod:
    def __init__(self, function, restrictions, t, a):
        self.f = function
        self.t = t
        self.a = a
        self.x = inner_point_of_restrictions(restrictions)
        self.rs = restrictions

    def __F(self):
        n = len(self.rs[0].a)
        return self.t * self.f + sum(map(lambda g: g.log_function(), self.rs), ConstantFunction(0, n))

    def next(self):
        F = self.__F()
        xn = self.x - np.matmul(LA.inv(F.quad(self.x)), F.grad(self.x))
        tn = self.a * self.t
        self.x = xn
        self.t = tn


class QuadraticFunction(TargetFunction):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, x):
        return np.matmul(np.matmul(np.transpose(x), self.A), x) + np.matmul(self.b, x) + self.c

    def grad(self, p):
        ret = np.array([0.0] * len(p))
        for i in range(len(p)):
            for j in range(len(p)):
                ret[i] += 2 * self.A[i][j] * p[j]
            ret[i] += self.b[i]
        return ret

    def quad(self, p):
        return 2 * self.A


def solve(function, restrictons):
    steps = 100
    a = 1.1
    t = 1
    method = InnerPointMethod(function, restrictons, t, a)
    for i in range(steps):
        print("x={} t={} target value={}".format(method.x, method.t, function(method.x)))
        method.next()


def main():
    """
        4(x - 1/2)**2 + 9(x - 1/3)**2 + 36(x + y - 5/6)**2
    """
    A = np.array([
        [40, 36],
        [36, 45]
    ])
    b = np.array([-64, -66])
    c = 27
    restrictions = [
        Restriction(np.array([-1, 0]), 0),
        Restriction(np.array([0, -1]), 0),
        Restriction(np.array([1, 0]), -1),
        Restriction(np.array([0, 1]), -1)
    ]
    solve(QuadraticFunction(A, b, c), restrictions)


if __name__ == "__main__":
    main()
