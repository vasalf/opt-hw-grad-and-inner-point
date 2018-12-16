#!/usr/bin/python3

import abc
import math
import numpy as np
from numpy import linalg as LA


class TargetFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, p):
        pass

    @abc.abstractmethod
    def m(self):
        pass

    @abc.abstractmethod
    def M(self):
        pass

    @abc.abstractmethod
    def grad(self, x):
        pass


class TargetRegion:
    def __init__(self, lbound, rbound):
        self.lbound = lbound
        self.rbound = rbound

    def contains(self, p):
        for i in range(len(p)):
            if self.lbound[i] > p[i]:
                return False
            if self.rbound[i] < p[i]:
                return False
        return True

    def closest_inner_point_to(self, p):
        return np.array([min(self.rbound[i], max(self.lbound[i], p[i])) for i in range(len(p))])

    def inner_point(self):
        return np.array([(self.lbound[i] + self.rbound[i]) / 2 for i in range(len(self.lbound))])


def solve_quadratic(a, b, c):
    """Returns list of real solutions of ax**2 + bx + c == 0
    """
    assert a != 0
    d = b * b - 4 * a * c
    if d < 0:
        return []
    elif d == 0:
        return -b / (2 * a)
    else:
        return [
            (-b - math.sqrt(d)) / (2 * a),
            (-b + math.sqrt(d)) / (2 * a)
        ]


class OptimalProjectiveGradientDescent:
    def __init__(self, f, region, a):
        self.f = f
        self.region = region
        self.x = region.inner_point()
        self.y = self.x
        self.a = a

    def __next_a(self):
        a = 1
        b = self.a * self.a - self.f.m() / self.f.M()
        c = -self.a * self.a
        vs = [x for x in solve_quadratic(a, b, c) if 0 < x and x < 1]
        return vs[0]

    def next(self):
        xn = self.region.closest_inner_point_to(self.y - (1/self.f.M()) * self.f.grad(self.y))
        an = self.__next_a();
        b = self.a * (1 - self.a) / (self.a * self.a + an)
        yn = xn + b * (xn - self.x)
        self.x = xn
        self.a = an
        self.y = yn


class Task1TargetFunction(TargetFunction):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, p):
        return np.matmul(np.transpose(p), np.matmul(self.A, p)) + np.matmul(self.b, p) + self.c

    def grad(self, p):
        ret = np.array([0] * len(p))
        for i in range(len(p)):
            for j in range(len(p)):
                ret[i] += 2 * self.A[i][j] * p[j]
            ret[i] += self.b[i]
        return ret

    def m(self):
        return min(LA.eigvals(2 * self.A))

    def M(self):
        return max(LA.eigvals(2 * self.A))


def solve(function):
    steps = 1000
    a = 1 / (function.m() + function.M())
    n = len(function.b)
    region = TargetRegion(np.array([0] * n), np.array([1] * n))
    desc = OptimalProjectiveGradientDescent(function, region, a)
    for i in range(steps):
        print("x={} y={} a={} target value={}".format(desc.x, desc.y, desc.a, function(desc.x)))
        desc.next()


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
    solve(Task1TargetFunction(A, b, c))

if __name__ == "__main__":
    main()
