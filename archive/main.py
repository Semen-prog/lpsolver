# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import time
from copy import copy
import scipy.stats
from scipy import sparse
from scipy.sparse.linalg import splu
np.bool = np.bool_

#np.seterr(all="warn")

# Генератор задач
# m - число переменных
# n - число уравнений


def generate_problem(n, m, d):
    assert m > n
    rng = np.random.default_rng()
    rvs = sp.stats.uniform(-5, 5).rvs
    # rvs = lambda k: sp.stats.bernoulli.rvs(1, size=k)

    # тут можно настраивать плотность матрицы
    # и распределение, из которого приходят элементы
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

    return Problem(A, b, c), (x, y, s)


def generate_clique_problem(n, adj_matrix):
    col = []
    row = []
    data = []

    # p_i --- vertex i is inside clique
    # c_ij --- variables for edge constraints
    # d_i --- variables for p_i <= 1 constrains
    b = []
    constraint_number = 0
    for i in range(n):
        for j in range(i):
            row.append(constraint_number)
            col.append(i)
            data.append(1)

            row.append(constraint_number)
            col.append(j)
            data.append(1)

            row.append(constraint_number)
            col.append(n + constraint_number)
            data.append(1)

            b.append(1 + adj_matrix[i][j])
            constraint_number += 1

    number_of_constrains = constraint_number
    for i in range(n):
        row.append(constraint_number)
        col.append(i)
        data.append(1)

        row.append(constraint_number)
        col.append(number_of_constrains + n + i)
        data.append(1)

        b.append(1)
        constraint_number += 1

    A = sp.sparse.coo_matrix(
        (data, (row, col)),
        shape=(n + number_of_constrains, 2 * n + number_of_constrains),
    )
    b = np.array(b)
    c = np.zeros((2 * n + number_of_constrains,))
    for i in range(n):
        c[i] = -1
    x = np.zeros(2 * n + number_of_constrains)
    for i in range(n):
        x[i] = 0.3

    constraint_number = 0
    for i in range(n):
        for j in range(i):
            x[n + constraint_number] = 1 + adj_matrix[i][j] - x[i] - x[j]
            constraint_number += 1
    for i in range(n):
        x[n + constraint_number + i] = 1 - x[i]
    y = np.zeros((n + number_of_constrains,))
    y[:] = -1
    s = c - A.T @ y
    A = A.tocsc()
    return Problem(A, b, c), (x, y, s)


def generate_graph_with_clique(n, k):
    adj_matrix = np.zeros((n, n))
    clique = np.arange(0, n, 1)
    np.random.shuffle(clique)
    clique = clique[:k]
    for i in range(n):
        for j in range(i):
            if i in clique and j in clique:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
            elif np.random.rand(1) < 0.3:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
    return adj_matrix

def find_kernel_vector(A):
    tmpA = A.T.copy()  # .tolil().copy()
    n = tmpA.shape[1]
    m = tmpA.shape[0]
    x = np.zeros(n)
    row = 0
    zeros = set()
    for i in range(min(n, m)):
        j = np.argmax(np.abs(tmpA[row:, i]))
        j += row
        # print('pivot = ',tmpA[j, i])
        if np.abs(tmpA[j, i]) < 1e-3:
            x[i] = 1
            zeros.add(i)
        else:
            tmpA[j, :] /= tmpA[j, i]
            tmpA[j], tmpA[row] = tmpA[row], tmpA[j]
            for k in range(row + 1, m):
                if np.abs(tmpA[k, i]) > 1e-3:
                    tmpA[k] -= tmpA[row] * tmpA[k, i]
            row += 1
    row -= 2
    for i in range(2, n + 1):
        col = n - i
        if col not in zeros:
            x[col] = -tmpA[row, col + 1:] @ x[col + 1:]
            row -= 1
        else:
            continue
    if not (np.abs(A.T @ x) < 1e-6).all():
        print(x)
        print(np.abs(A.T @ x))
    assert (np.abs(A.T @ x) < 1e-6).all()
    return x

def solve_sparse_with_one_rank(invM, u, v, b):
    # Mx - u v^T x = b
    
    y = invM.solve(b)
    z = invM.solve(u)
    assert abs(1 - v.T@z) > 1e-9
    lambd = v.T@y/(1 - v.T@z)
    
    return y + lambd*z


class Problem:
    def __init__(self, A, b, c):
        self.A = A
        self.n = A.shape[0]
        self.m = A.shape[1]
        self.b = b
        self.c = c

    def primal_value(self, x):
        return (x * self.c).sum()

    def dual_value(self, y):
        return (y * self.b).sum()


class CentralPathSolver:
    def __init__(self, problem, verbose=True):
        self.problem = problem
        self.verbose = verbose
        self.index_zero = set()
        self.index_free = set()
        # value added to A2^T Q^-1 A2 as regularizer
        self.eps = 1e-6
        # Necessary accuracy for ellipsoidal problem solution
        self.ellipsoidal_acc = 1e-3

    # тут параметры лучше не менять, всё и так работает
    def solve(self, x0, s0, y0, eps, gamma_min=0.9, dgamma=0.2):
        self.x = np.copy(x0)
        self.y = np.copy(y0)
        self.s = np.copy(s0)
        self.primal = self.problem.primal_value(self.x)
        self.dual = self.problem.dual_value(self.y)

        assert gamma_min > dgamma

        indices = [
            i
            for i in range(self.problem.m)
            if i not in self.index_zero and i not in self.index_free
        ]

        # индексы неудалённых пар переменных
        self.remaining = sorted(indices)

        while self.mu(self.x[self.remaining], self.s[self.remaining]) > eps:
            while (
                self.gamma(self.x[self.remaining], self.s[self.remaining]) < gamma_min
            ):
                if self.verbose:
                    print(
                        "mu, gamma = ",
                        self.mu(self.x[self.remaining], self.s[self.remaining]),
                        self.gamma(self.x[self.remaining], self.s[self.remaining]),
                    )
                    print("primal, dual = ", self.primal, self.dual)

                mu = self.mu(self.x[self.remaining], self.s[self.remaining])
                delta_x, delta_y, delta_s = self.center_direction(mu)
                step_size = self.center_step_optimal_len(delta_x, delta_y, delta_s)
                if self.verbose:
                    print("step size = ", step_size)
                self.center_step(step_size, delta_x, delta_y, delta_s)

                if (
                    self.primal - self.dual < 0.1
                    and self.problem.m - len(self.index_zero) - len(self.index_free) > 2
                ):
                    self.filter_variables()
                    # порядок вывода: те пары, у которых x = 0, те, у которых s = 0
                    if self.verbose:
                        print(
                            "pruned: ",
                            self.index_zero,
                            self.index_free,
                            len(self.index_zero),
                            len(self.index_free)
                        )

            if self.verbose:
                print(
                    "mu, gamma = ",
                    self.mu(self.x[self.remaining], self.s[self.remaining]),
                    self.gamma(self.x[self.remaining], self.s[self.remaining]),
                )
                print("primal, dual = ", self.primal, self.dual)

            # вычисление направления Нестерова и длины шага
            delta_x, delta_y, delta_s = self.predict_direction()
            step_size = self.predict_step_optimal_len(
                gamma_min - dgamma, delta_x, delta_y, delta_s
            )
            if self.verbose:
                print("step size = ", step_size)
            time.sleep(2)
            self.predict_step(step_size, delta_x, delta_y, delta_s)

            if (
                self.primal - self.dual < 0.1
                and self.problem.m - len(self.index_zero) - len(self.index_free) > 2
            ):
                self.filter_variables()
                # порядок вывода: те пары, у которых x = 0, те, у которых s = 0
                if self.verbose:
                    print(
                        "pruned: ",
                        self.index_zero,
                        self.index_free,
                        len(self.index_zero),
                        len(self.index_free)
                    )

        print("x = ", self.x)
        print("s = ", self.s)

        print(np.max(np.abs(self.problem.A.T @ self.y + self.s - self.problem.c)))
        print(self.dual, self.primal)
        print(np.max(np.abs(self.problem.A @ self.x - self.problem.b)))

    def mu(self, x, s):
        return (x * s).sum() / x.shape[0]

    def gamma(self, x, s):
        mu = self.mu(x, s)
        return min(x * s) / mu

    # проверялка допустимости точки
    def check(self, x, s):
        return (x > 0).all() and (s > 0).all()

    # попытка удалить пару переменных
    # решаем задачу на текущей точке Нестерова
    # если не получается, берём предыдущие
    def filter_variables(self):
        for i in copy(self.remaining):
            # попытка удалить пару переменных
            # решаем задачу на текущей точке Нестерова
            # если не получается, берём предыдущие
            w = np.zeros_like(self.x)
            w[self.remaining] = (self.x[self.remaining] / (self.s[self.remaining])) ** 0.5
            (upper_bound, x_u, y_u, s_u), status = self.elipsoidal_bound(i, w, "upper")
            
            if status == 2:
                assert False
            
            if not status and upper_bound < self.dual:
                # x_i = 0
                if x_u[i] < 0:
                    alpha = -x_u[i] / (self.x[i] - x_u[i])
                    self.x = alpha * self.x + (1 - alpha) * x_u
                    self.index_zero.add(i)
                
                    self.primal = self.problem.primal_value(self.x)
                    self.dual = self.problem.dual_value(self.y)
                
                    indices = [
                        i
                        for i in range(self.problem.m)
                        if i not in self.index_zero
                        and i not in self.index_free
                    ]
                    self.remaining = sorted(indices)
                    continue

            w = np.zeros_like(self.x)
            w[self.remaining] = (self.x[self.remaining] / (self.s[self.remaining]) )** 0.5
            (lower_bound, x_l, y_l, s_l), status = self.elipsoidal_bound(i, w, "lower")
            
            # нашли однозначную переменную
            if status == 2:
                nonzero = [
                    j
                    for j in range(self.problem.m)
                    if j not in self.index_zero and i != j
                ]
                v = find_kernel_vector(self.problem.A[:, nonzero])
                if max(abs(v)) > 0:
                    self.index_free.add(i)
                    self.y -= (self.s[i]/(v@self.problem.A[:, i]))*v
                    self.s[i] = 0
                else:
                    assert False
                continue

            if not status and lower_bound > self.primal:
                # s_i = 0

                if s_l[i] < 0:
                    alpha = -s_l[i] / (self.s[i] - s_l[i])
                    self.s = alpha * self.s + (1 - alpha) * s_l
                    self.y = alpha * self.y + (1 - alpha) * y_l
                    self.index_free.add(i)

                    self.primal = self.problem.primal_value(self.x)
                    self.dual = self.problem.dual_value(self.y)

                    indices = [
                        i
                        for i in range(self.problem.m)
                        if i not in self.index_zero
                        and i not in self.index_free
                    ]
                    self.remaining = sorted(indices)

    # шаг по направлению с проверкой того, что нет ошибок
    # primal > dual
    # тут часто падает, поэтому выводятся невязки, чтобы понять какая
    # компонента сломалась
    def step(self, step_size, delta_x, delta_y, delta_s):
        self.x += step_size * delta_x
        self.y += step_size * delta_y
        self.s += step_size * delta_s
        free_s = sorted(list(self.index_zero))

        if free_s != []:
            self.s[free_s] = self.problem.c[free_s] - self.problem.A.T[free_s] @ self.y

        self.primal = self.problem.primal_value(self.x)
        self.dual = self.problem.dual_value(self.y)
        if self.dual > self.primal:
            print(np.max(np.abs(self.problem.A.T @ self.y + self.s - self.problem.c)))
            print(self.dual, self.primal)
            print(np.max(np.abs(self.problem.A @ self.x - self.problem.b)))
        assert (np.max(np.abs(self.problem.A.T @ self.y + self.s - self.problem.c)) < 1e-6).all()
        assert (np.max(np.abs(self.problem.A @ self.x - self.problem.b) < 1e-6)).all()
        
        assert self.dual < self.primal

    def predict_step(self, step_size, delta_x, delta_y, delta_s):
        self.step(step_size, delta_x, delta_y, delta_s)

    def center_step(self, step_size, delta_x, delta_y, delta_s):
        self.step(step_size, delta_x, delta_y, delta_s)

    # одномерный оптимизатор для длины шага
    def center_step_optimal_len(self, delta_x, delta_y, delta_s):
        # indices = [i for i in range(self.problem.m) if i not in self.index_zero
        #               and i not in self.index_free and i not in self.index_known]]
        remaining = self.remaining
        upper_bound = 1e-3
        mu = self.mu(self.x[remaining], self.s[remaining])

        # вычисление максимально возможной длины шага

        while (
            (((self.x[remaining] + upper_bound * delta_x[remaining]) <= 1e-6).any()
            or ((self.s[remaining] + upper_bound * delta_s[remaining]) <= 1e-6).any()
            or self.mu(
                self.x[remaining] + upper_bound * delta_x[remaining],
                self.s[remaining] + upper_bound * delta_s[remaining],
            )
            > 1.1 * mu) and upper_bound >= 1e-6
        ):
            upper_bound /= 2
        while (
            ((self.x[remaining] + 2 * upper_bound * delta_x[remaining]) > 1e-6).all()
            and (
                (self.s[remaining] + 2 * upper_bound * delta_s[remaining]) > 1e-6
            ).all()
            and self.mu(
                self.x[remaining] + 2 * upper_bound * delta_x[remaining],
                self.s[remaining] + 2 * upper_bound * delta_s[remaining],
            )
            <= 1.1 * mu
        ):
            upper_bound *= 2

        u = 2 * upper_bound

        while u - upper_bound > 1e-2:
            mid = (u + upper_bound) / 2
            if (
                ((self.x[remaining] + mid * delta_x[remaining]) > 1e-6).all()
                and ((self.s[remaining] + mid * delta_s[remaining]) > 1e-6).all()
                and self.mu(
                    self.x[remaining] + mid * delta_x[remaining],
                    self.s[remaining] + mid * delta_s[remaining],
                )
                <= 1.1 * mu
            ):
                upper_bound = mid
            else:
                u = mid

        # поиск минимума суммы барьеров

        res = sp.optimize.minimize(
            x0=upper_bound,
            bounds=[(1e-9, upper_bound)],
            fun=lambda l: -sum(np.log(self.x[remaining] + l * delta_x[remaining]))
            - sum(np.log(self.s[remaining] + l * delta_s[remaining])),
            method="L-BFGS-B",
            jac=lambda l: -sum(
                delta_x[remaining] / (self.x[remaining] + l * delta_x[remaining])
            )
            - sum(delta_s[remaining] / (self.s[remaining] + l * delta_s[remaining])),
        )
        return res.x

    # одномерный оптимизатор для длины шага

    def predict_step_optimal_len(self, gamma_min, delta_x, delta_y, delta_s):
        # indices = [i for i in range(self.problem.m) if i not in self.index_zero
        #               and i not in self.index_free and i not in self.index_known]]
        remaining = self.remaining
        step = 1e-3

        # вычисление максимально возможной длины шага

        while (
            (not self.check(
                self.x[remaining] + 2 * step * delta_x[remaining],
                self.s[remaining] + 2 * step * delta_s[remaining],
            )
            or self.gamma(
                self.x[remaining] + step * delta_x[remaining],
                self.s[remaining] + step * delta_s[remaining],
            )
            < gamma_min) and step >= 1e-6
        ):
            step /= 2
        while (
            self.check(
                self.x[remaining] + 2 * step * delta_x[remaining],
                self.s[remaining] + 2 * step * delta_s[remaining],
            )
            and self.gamma(
                self.x[remaining] + 2 * step * delta_x[remaining],
                self.s[remaining] + 2 * step * delta_s[remaining],
            )
            >= gamma_min
        ):
            step *= 2

        # бинарный поиск по длине шага, максимизирующий шаг для выхода на границу
        # окрестности по гамма

        u = 2 * step
        while u - step > 1e-2:
            mid = (u + step) / 2
            if (
                self.check(
                    self.x[remaining] + 2 * step * delta_x[remaining],
                    self.s[remaining] + 2 * step * delta_s[remaining],
                )
                and self.gamma(
                    self.x[remaining] + mid * delta_x[remaining],
                    self.s[remaining] + mid * delta_s[remaining],
                )
                >= gamma_min
            ):
                step = mid
            else:
                u = mid

        return step

    def center_direction(self, mu):
        s = self.s

        # indices = [i for i in range(self.problem.m) if i not in self.index_zero
        #               and i not in self.index_free and i not in self.index_known]]
        free = sorted(list(self.index_free))
        # remaining = sorted(indices)

        # центрирующее направление в двух случаях:
        # при отсутствии свободных переменных и при наличии
        if free != []:
            A1 = self.problem.A[:, free]
            A2 = self.problem.A[:, self.remaining]

            x2 = self.x[self.remaining].reshape(-1, 1)

            n2 = len(self.remaining)
            s = s[self.remaining]
            s = s.reshape(-1, 1)
            invH = sp.sparse.diags((x2.squeeze() / s.squeeze()), shape=(n2, n2))
            
            N = A2 @ invH @ A2.T
            M = sp.sparse.bmat(
                [
                    [A1, N],
                    [None, A1.T],
                ]
            )
            invM = splu(M)
            sol = invM.solve(np.vstack((A2 @ (x2 - mu / s), np.zeros((len(free), 1)))))
            delta_x1 = sol[:len(free)]
            delta_y = sol[len(free):]
            delta_s = -A2.T @ delta_y
            delta_x2 = -invH @ delta_s - x2 + mu / s
            
            true_deltax = np.zeros_like(self.x)
            true_deltax[free] = delta_x1.reshape((-1,))
            true_deltax[self.remaining] = delta_x2.reshape((-1,))
            true_deltas = np.zeros_like(self.s)

            true_deltas[self.remaining] = delta_s.reshape((-1,))

            delta_y = delta_y.reshape((-1,))

            return (true_deltax, delta_y, true_deltas)
        else:
            A2 = self.problem.A[:, self.remaining]

            x2 = self.x[self.remaining].reshape(-1, 1)
            s = s[self.remaining]
            s = s.reshape(-1, 1)

            n2 = len(self.remaining)
            invH = sp.sparse.diags((x2.squeeze() / s.squeeze()), shape=(n2, n2))

            invN = splu(
                A2 @ invH @ A2.T
            )
            delta_y = invN.solve(-A2 @ (-x2 + mu / s))
            delta_s = -A2.T @ delta_y
            delta_x2 = -invH @ delta_s - x2 + mu / s

            true_deltax = np.zeros_like(self.x)
            true_deltax[self.remaining] = delta_x2.reshape((-1,))
            true_deltas = np.zeros_like(self.s)

            true_deltas[self.remaining] = delta_s.reshape((-1,))

            delta_y = delta_y.reshape((-1,))

            return (true_deltax, delta_y, true_deltas)

    def predict_direction(self):
        s = self.s
        # indices = [i for i in range(self.problem.m) if i not in self.index_zero
        #               and i not in self.index_free and i not in self.index_known]]
        free = sorted(list(self.index_free))
        # remaining = sorted(indices)
        # предсказывающее направление в двух случаях:
        # при отсутствии свободных переменных и при наличии

        if free != []:
            A1 = self.problem.A[:, free]
            A2 = self.problem.A[:, self.remaining]

            x2 = self.x[self.remaining].reshape(-1, 1)

            n2 = len(self.remaining)
            s = s[self.remaining]
            s = s.reshape(-1, 1)
            invH = sp.sparse.diags((x2.squeeze() / s.squeeze()), shape=(n2, n2))
            
            N = A2 @ invH @ A2.T
            M = sp.sparse.bmat(
                [
                    [A1, N],
                    [None, A1.T],
                ]
            )
            invM = splu(M)
            sol = invM.solve(np.vstack((A2 @ (x2), np.zeros((len(free), 1)))))
            delta_x1 = sol[:len(free)]
            delta_y = sol[len(free):]
            delta_s = -A2.T @ delta_y
            delta_x2 = -invH @ delta_s - x2

            true_deltax = np.zeros_like(self.x)
            true_deltax[free] = delta_x1.reshape((-1,))
            true_deltax[self.remaining] = delta_x2.reshape((-1,))
            true_deltas = np.zeros_like(self.s)

            true_deltas[self.remaining] = delta_s.reshape((-1,))
            delta_y = delta_y.reshape((-1,))

            return (true_deltax, delta_y, true_deltas)
        else:
            A2 = self.problem.A[:, self.remaining]

            x2 = copy(self.x[self.remaining]).reshape(-1, 1)
            s = s[self.remaining]
            s = s.reshape(-1, 1)

            n2 = len(self.remaining)
            invH = sp.sparse.diags((x2.squeeze() / s.squeeze()), shape=(n2, n2))

            invN = splu(
                A2 @ invH @ A2.T
            )
            delta_y = invN.solve(-A2 @ (-x2))
            delta_s = -A2.T @ delta_y
            delta_x2 = -invH @ delta_s - x2

            true_deltax = np.zeros_like(self.x)
            true_deltax[self.remaining] = delta_x2.reshape((-1,))
            true_deltas = np.zeros_like(self.s)

            true_deltas[self.remaining] = delta_s.reshape((-1,))
            delta_y = delta_y.reshape((-1,))

            return (true_deltax, delta_y, true_deltas)

    # вспомогательная задача
    def elipsoidal_bound(self, j, w, bound):
        # Q = alpha*w*w^t - diag(w)^2
        if bound == "upper":
            # return (None, None, None, None), 1
            remaining = [i for i in range(self.problem.m) if i not in self.index_zero
                       and i not in self.index_free and i != j]
            free = sorted(list(self.index_free) + [j])
            w2 = w[remaining]
            n2 = len(remaining)
            # w2 /= np.max(w2)
            # w2 += 1e-3
            # theory
            Q = [
                1 / (w2.reshape(-1, 1) * (n2 - 1)),
                1 / w2.reshape(-1, 1),
                sp.sparse.diags((1 / w2) ** 2, shape=(n2, n2), dtype=np.longdouble),
            ]
            invQ = [
                w2.reshape(-1, 1),
                w2.reshape(-1, 1),
                sp.sparse.diags((w2) ** 2, shape=(n2, n2), dtype=np.longdouble),
            ]

        elif bound == "lower":
            # return (None, None, None, None), 1
            remaining = [i for i in range(self.problem.m) if i not in self.index_zero
                       and i not in self.index_free and i != j]
            free = sorted(list(self.index_free))
            w2 = w[remaining]
            n2 = len(remaining)
            # w2 /= np.max(w2)
            # w2 += 1e-3
            # theory

            Q = [
                1 / w2.reshape(-1, 1),
                1 / w2.reshape(-1, 1),
                sp.sparse.diags((1 / w2) ** 2, shape=(n2, n2), dtype=np.longdouble),
            ]
            invQ = [
                w2.reshape(-1, 1) / (n2 - 1),
                w2.reshape(-1, 1),
                sp.sparse.diags((w2) ** 2, shape=(n2, n2), dtype=np.longdouble),
            ]
        else:
            return (None, None, None, None), 1

        A2 = self.problem.A[:, remaining]
        if free != []:
            A1 = self.problem.A[:, free]
            A = self.problem.A
            c = np.copy(self.problem.c).reshape(-1, 1)
            c1 = np.copy(self.problem.c[free]).reshape(-1, 1)
            c2 = np.copy(self.problem.c[remaining]).reshape(-1, 1)
            b = np.copy(self.problem.b).reshape(-1, 1)
            
            true_b = np.copy(self.problem.b).reshape(-1, 1)
            true_c = c.copy()

            status, true_x, true_y, true_s, cost = self.solve_ellipsoidal_system_with_free(
                A1, A2, invQ, Q, c, c1, c2, b, free, remaining, bound, j
            )
            
            if status == 2:
                #print('status = ',status)
                return (None, None, None, None), status
            
            if status != 0:
                return (None, None, None, None), status

            # проверка точности
            if (
                bound == "upper"
                and (abs(true_b - A @ true_x) > self.ellipsoidal_acc).any()
            ):
                #print('Invalid solution upper ',max(abs(true_b - A @ true_x)))
                return (None, None, None, None), 1
            if (
                bound == "lower"
                and (abs(true_c - A.T @ true_y - true_s) > self.ellipsoidal_acc).any()
            ):
                #print('Invalid solution lower ',max(abs(true_c - A.T @ true_y - true_s)))
                return (None, None, None, None), 1

            if ((true_x[remaining] >= -1e-5).all() and bound == "upper") or (
                bound == "lower" and (true_s[remaining] >= -1e-5).all()
            ):
                return (
                    cost,
                    true_x.reshape((-1,)),
                    true_y.reshape((-1,)),
                    true_s.reshape((-1,)),
                ), 0
            else:
                #print('Invalid solution')
                return (None, None, None, None), 1
        else:
            A = self.problem.A
            c = np.copy(self.problem.c).reshape(-1, 1)
            c2 = np.copy(self.problem.c[remaining]).reshape(-1, 1)
            b = np.copy(self.problem.b).reshape(-1, 1)
            
            true_b = b.copy()
            true_c = c.copy()

            status, true_x, true_y, true_s, cost = self.solve_ellipsoidal_system(
                A2, invQ, Q, c, c2, b, free, remaining, bound, j
            )

            if status == 2:
                #print('status = ',status)
                return (None, None, None, None), status
            
            if status != 0:
                return (None, None, None, None), status
            
            if (
                bound == "upper"
                and (abs(true_b - A @ true_x) > self.ellipsoidal_acc).any()
            ):
                #print('Invalid solution upper ',max(abs(true_b - A @ true_x)))
                return (None, None, None, None), 1
            if (
                bound == "lower"
                and (abs(true_c - A.T @ true_y - true_s) > self.ellipsoidal_acc).any()
            ):
                #print('Invalid solution lower ',max(abs(true_c - A.T @ true_y - true_s)))
                return (None, None, None, None), 1

            if ((true_x[remaining] >= -1e-5).all() and bound == "upper") or (
                bound == "lower" and (true_s[remaining] >= -1e-5).all()
            ):
                return (
                    cost,
                    true_x.reshape((-1,)),
                    true_y.reshape((-1,)),
                    true_s.reshape((-1,)),
                ), 0
            else:
                #print('Invalid solution')
                return (None, None, None, None), 1

    def solve_ellipsoidal_system(
        self, A2, invQ, Q, c, c2, b, free, remaining, bound, j
    ):
        solvable = True
        try:
            #invN = sp.linalg.inv(
            #    (A2 @ invQ[0]) @ (invQ[1].T @ A2.T) - A2 @ invQ[2] @ A2.T
            #)
            N = - A2 @ invQ[2] @ A2.T
            u = - (A2 @ invQ[0])
            v = (A2 @ invQ[1])
            invN = splu(N)
        except:
            solvable = False

        if solvable:
            
            vec = A2.T @ (solve_sparse_with_one_rank(invN, u, v, A2 @ (invQ[0] * (invQ[1].T @ c2) - invQ[2] @ c2)))
            coeff_0 = invQ[0] * invQ[1].T @ (c2 - vec) - invQ[2] @ (c2 - vec)

            coeff_0 = coeff_0.T @ (Q[0] * (Q[1].T @ coeff_0) - Q[2] @ coeff_0)

            vec = A2.T @ (solve_sparse_with_one_rank(invN, u, v, b))
            coeff_1 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

            coeff_1 = coeff_1.T @ (Q[0] * (Q[1].T @ coeff_1) - Q[2] @ coeff_1)

            if coeff_0 / coeff_1 > 0:
                return 1, -1, -1, -1, None

            lambd = (-coeff_0.item() / coeff_1.item()) ** 0.5
            lambda_inv = 1 / lambd
        else:
            #print("Unsolvable system found ", j)
            return 2, -1, -1, -1, None

        if solvable and lambda_inv < 1e6:
            true_x = np.zeros_like(self.x).reshape(-1, 1)
            true_y = np.zeros_like(self.y).reshape(-1, 1)
            true_s = np.zeros_like(self.s).reshape(-1, 1)

            true_y[:] += solve_sparse_with_one_rank(invN, u, v,
                A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2) - lambd * b
            )

            vec = (
                lambda_inv * c2
                - A2.T
                @ (
                    solve_sparse_with_one_rank(invN, u, v,
                        A2
                        @ (
                            lambda_inv * invQ[0] * invQ[1].T @ c2
                            - lambda_inv * invQ[2] @ c2
                        )
                    )
                )
            ) + A2.T @ (solve_sparse_with_one_rank(invN, u, v, b))
            x2 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

            s = (
                c2 - A2.T @ (solve_sparse_with_one_rank(invN, u, v, A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2)))
            ) + A2.T @ (solve_sparse_with_one_rank(invN, u, v, lambd * b))

            true_x[remaining] += x2

            free_s = sorted(list(self.index_zero))
            if bound == "lower":
                free_s = sorted(list(self.index_zero) + [j])

            if free_s != []:
                true_s[free_s] += (
                    self.problem.c.reshape(-1, 1)[free_s]
                    - self.problem.A.T[free_s] @ true_y
                )
            true_s[remaining] += s

            if bound == "upper":
                cost = true_x.T @ self.problem.c
            if bound == "lower":
                cost = true_y.T @ self.problem.b

            return 0, true_x, true_y, true_s, cost
        else:
            #print("big lambda ", j)
            return 1, -1, -1, -1, None

    def solve_ellipsoidal_system_with_free(
        self, A1, A2, invQ, Q, c, c1, c2, b, free, remaining, bound, j
    ):
        m = b.shape[0]
        n_free = A1.shape[1]
        u = np.vstack((A2 @ invQ[0], np.zeros((n_free, 1))))
        v = np.vstack((np.zeros((n_free, 1)), A2 @ invQ[1]))
        M = sp.sparse.bmat(
            [
                [A1, A2 @ invQ[2] @ A2.T],
                [None, A1.T],
            ]
        )
        try:
            #invM = sp.linalg.inv(M)
            invM = splu(M)
        except Exception as e:
            #print('uninvertible matrix')
            return 2, -1, -1, -1, None

        #sol_lambda = invM.solve(np.vstack((b, np.zeros_like(c1))))
        sol_lambda = solve_sparse_with_one_rank(invM, u, v, np.vstack((b, np.zeros_like(c1))))
        #sol_free = invM.solve(np.vstack((-A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2), c1)))
        sol_free = solve_sparse_with_one_rank(invM, u, v, np.vstack((-A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2), c1)))
        y_lambda = sol_lambda[n_free:]
        y_free = sol_free[n_free:]
        
        x2_lambda = - A2.T @ y_lambda
        x2_free = c2 - A2.T @ y_free

        coeff_free = (
            x2_free * (invQ[0] * invQ[1].T @ x2_free - invQ[2] @ x2_free)
        ).sum()
        coeff_lin = (
            x2_free * (invQ[0] * invQ[1].T @ x2_lambda - invQ[2] @ x2_lambda)
        ).sum()
        coeff_sq = (
            x2_lambda * (invQ[0] * invQ[1].T @ x2_lambda - invQ[2] @ x2_lambda)
        ).sum()

        # print('coeff_lin = ', coeff_lin)
        # print('coeff_free = ', coeff_free)
        # print('coeff_sq = ', coeff_sq)
        # print(sp.linalg.det(M))
        if abs(coeff_sq) > 1e-12 and coeff_free / coeff_sq > 0:
            #print('error full matrix lambda unsolvable ',coeff_sq, coeff_free / coeff_sq)
            return 1, -1, -1, -1, None
        lambd = (-coeff_free / coeff_sq) ** 0.5
        lambda_inv = 1.0 / lambd
        if lambda_inv > 1e6:
            #print("lambda big", j)
            return 1, -1, -1, -1, None

        y = lambd * y_lambda + y_free
        vec = c2 - A2.T @ y
        x2 = 1.0 / lambd * (invQ[0] * invQ[1].T @ vec - invQ[2] @ vec)
        x1 = (
            sol_lambda[:n_free] + lambda_inv * sol_free[:n_free]
        )
        
        
        true_x = np.zeros_like(self.x).reshape(-1, 1)
        true_y = np.zeros_like(self.y).reshape(-1, 1)
        true_s = np.zeros_like(self.s).reshape(-1, 1)

        true_x[free] += x1
        true_x[remaining] += x2

        s = lambd * (Q[0] * Q[1].T @ x2 - Q[2] @ x2)

        true_y = y

        if bound == "upper":
            cost = true_x.T @ self.problem.c
        if bound == "lower":
            cost = true_y.T @ self.problem.b

        free_s = sorted(list(self.index_zero))
        if bound == "lower":
            free_s = sorted(list(self.index_zero) + [j])

        if free_s != []:
            true_s[free_s] += c[free_s] - self.problem.A.T[free_s] @ true_y
        true_s[remaining] += s

        return 0, true_x, true_y, true_s, cost
        
    def solve_regularized_ellipsoidal_system_with_free(
        self, A1, A2, invQ, Q, c, c1, c2, b, free, remaining, bound, j
    ):
        invN = sp.linalg.inv(
            (A2 @ invQ[0]) @ (invQ[1].T @ A2.T)
            - A2 @ invQ[2] @ A2.T
            + sp.sparse.eye(A2.shape[0]) * self.eps
        )
        invM = sp.linalg.inv(A1.T @ invN @ A1 + sp.sparse.eye(A1.shape[-1]) * self.eps)

        # формула из листочка на x2

        # find lambda as solution
        # coeff_0 + coeff_1*lambda^2 = 0

        vec = A2.T @ (invN @ (A2 @ (invQ[0] * (invQ[1].T @ c2) - invQ[2] @ c2)))

        coeff_0 = invQ[0] * invQ[1].T @ (c2 - vec) - invQ[2] @ (c2 - vec)

        vec = A2.T @ (
            invN
            @ (
                A1
                @ (
                    invM
                    @ (
                        -c1
                        + A1.T
                        @ (invN @ (A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2)))
                    )
                )
            )
        )

        coeff_0 += invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

        # x.TQx = 0
        coeff_0 = coeff_0.T @ (Q[0] * (Q[1].T @ coeff_0) - Q[2] @ coeff_0)

        vec = A2.T @ (invN @ b)
        coeff_1 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

        vec = A2.T @ (invN @ (A1 @ (invM @ (A1.T @ (invN @ b)))))
        coeff_1 -= invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

        coeff_1 = coeff_1.T @ (Q[0] * (Q[1].T @ coeff_1) - Q[2] @ coeff_1)

        if coeff_0 / coeff_1 > 0:
            return 1, -1, -1, -1, None
        lambd = (-coeff_0.item() / coeff_1.item()) ** 0.5

        lambda_inv = 1 / lambd
        true_x = np.zeros_like(self.x).reshape(-1, 1)
        true_y = np.zeros_like(self.y).reshape(-1, 1)
        true_s = np.zeros_like(self.s).reshape(-1, 1)

        x1 = (
            lambda_inv * invM @ c1
            - invM
            @ (
                A1.T
                @ (
                    invN
                    @ (
                        A2
                        @ (
                            lambda_inv * invQ[0] * (invQ[1].T @ c2)
                            - lambda_inv * invQ[2] @ c2
                        )
                    )
                )
            )
        ) + invM @ (A1.T @ (invN @ b))

        lambda_x1 = (
            invM @ c1
            - invM
            @ (A1.T @ (invN @ (A2 @ (invQ[0] * (invQ[1].T @ c2) - invQ[2] @ c2))))
        ) + lambd * invM @ (A1.T @ (invN @ b))
        true_y[:] += invN @ (
            A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2)
        ) + invN @ (-lambd * b + A1 @ lambda_x1)

        vec = (
            lambda_inv * c2
            - A2.T
            @ (
                invN
                @ (
                    A2
                    @ (
                        lambda_inv * invQ[0] * invQ[1].T @ c2
                        - lambda_inv * invQ[2] @ c2
                    )
                )
            )
        ) - A2.T @ (invN @ (-b + A1 @ x1))

        x2 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec
        s = c2 - A2.T @ true_y

        true_x[free] += x1
        true_x[remaining] += x2

        if bound == "upper":
            cost = true_x.T @ self.problem.c
        if bound == "lower":
            cost = true_y.T @ self.problem.b

        free_s = sorted(list(self.index_zero))
        if bound == "lower":
            free_s = sorted(list(self.index_zero) + [j])

        if free_s != []:
            true_s[free_s] += c[free_s] - self.problem.A.T[free_s] @ true_y
        true_s[remaining] += s

        return 0, true_x, true_y, true_s, cost

    def solve_regularized_ellipsoidal_system(
        self, A2, invQ, Q, c, c2, b, free, remaining, bound, j
    ):
        invN = sp.linalg.inv(
            (A2 @ invQ[0]) @ (invQ[1].T @ A2.T)
            - A2 @ invQ[2] @ A2.T
            + sp.sparse.eye(A2.shape[0]) * self.eps
        )

        vec = A2.T @ (invN @ (A2 @ (invQ[0] * (invQ[1].T @ c2) - invQ[2] @ c2)))
        coeff_0 = invQ[0] * invQ[1].T @ (c2 - vec) - invQ[2] @ (c2 - vec)

        coeff_0 = coeff_0.T @ (Q[0] * (Q[1].T @ coeff_0) - Q[2] @ coeff_0)

        vec = A2.T @ (invN @ b)
        coeff_1 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

        coeff_1 = coeff_1.T @ (Q[0] * (Q[1].T @ coeff_1) - Q[2] @ coeff_1)

        if coeff_0 / coeff_1 > 0:
            return 1, -1, -1, -1, None

        lambd = (-coeff_0.item() / coeff_1.item()) ** 0.5
        lambda_inv = 1 / lambd

        true_x = np.zeros_like(self.x).reshape(-1, 1)
        true_y = np.zeros_like(self.y).reshape(-1, 1)
        true_s = np.zeros_like(self.s).reshape(-1, 1)

        true_y[:] += invN @ (
            A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2) - lambd * b
        )

        vec = (
            lambda_inv * c2
            - A2.T
            @ (
                invN
                @ (
                    A2
                    @ (
                        lambda_inv * invQ[0] * invQ[1].T @ c2
                        - lambda_inv * invQ[2] @ c2
                    )
                )
            )
        ) + A2.T @ (invN @ (b))
        x2 = invQ[0] * invQ[1].T @ vec - invQ[2] @ vec

        s = (
            c2 - A2.T @ (invN @ (A2 @ (invQ[0] * invQ[1].T @ c2 - invQ[2] @ c2)))
        ) + A2.T @ (invN @ (lambd * b))

        true_x[remaining] += x2

        free_s = sorted(list(self.index_zero))
        if bound == "lower":
            free_s = sorted(list(self.index_zero) + [j])

        if free_s != []:
            true_s[free_s] += (
                self.problem.c.reshape(-1, 1)[free_s]
                - self.problem.A.T[free_s] @ true_y
            )
        true_s[remaining] += s

        if bound == "upper":
            cost = true_x.T @ self.problem.c
        if bound == "lower":
            cost = true_y.T @ self.problem.b

        return 0, true_x, true_y, true_s, cost

