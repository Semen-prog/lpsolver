#!/usr/bin/env python

import numpy as np
import scipy as sp

def print_matrix(n, m, M):
    nz = []
    for i in range(n):
        for j in range(m):
            print(M[i, j], end = ' ')
        print()

def print_vector(V):
    print(*V)

def generate_problem(n, m, d):
    assert m > n
    rng = np.random.default_rng()
    rvs = sp.stats.uniform(-5, 5).rvs

    A = sp.sparse.random(
        n,
        m,
        density=d,
        random_state=rng,
        data_rvs=rvs,
        dtype=np.longdouble,
        format="csc",
    )

    # регуляризация
    # A += 10 * sp.sparse.diags([1] * min(n, m), shape=(n, m), dtype=np.longdouble)
    y = 2 * np.random.rand(n).astype(np.longdouble) - 1

    x = np.abs(np.random.rand(m), dtype=np.longdouble) + 1
    s = np.abs(np.random.rand(m), dtype=np.longdouble) + 1

    b = A @ x
    c = s + A.T @ y

    print(n, m)
    print_matrix(n, m, A)
    print_vector(b)
    print_vector(c)
    print_vector(x)
    print_vector(y)
    print_vector(s)
    #return Problem(A, b, c), (x, y, s)

import sys
n = int(sys.argv[1])
m = int(sys.argv[2])

generate_problem(n, m, 1 / n)