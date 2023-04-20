from table import solve_by_gauss
import numpy as np


def u_n(x, n, display=False):
    if (display):
        return '(1 - x) * (x ** {0})'.format(n)
    return (1 - x) * (x ** n)


def alpha_2(xs):
    sum = 0
    for x in xs:
        sum += -2 + 2 * x - 3 * x * x
    return sum


def betta_2(xs):
    sum = 0
    for x in xs:
        sum += -1 + 3 * x ** 2 - 4 * x ** 3
    return sum


def create_matrix_2(xs, n, params=[alpha_2, betta_2]):  # n - кол-во параметров матрицы params == alpha,betta etc...
    matrix = [[i for i in range(n + 1)] for j in range(n)]
    for i in range(n):
        for j in range(n + 1):
            if (j == n):
                matrix[i][j] = (2 * sum(xs) - 1) * params[i](xs)
            elif (i == j):
                matrix[i][j] = params[i](xs) ** 2
            else:
                matrix[i][j] = betta_2(xs) * alpha_2(xs)
    print(matrix)
    return np.array(matrix)


def get_func_2(xs, n, params=[alpha_2, betta_2], display=False):
    matrix = create_matrix_2(xs, n, params)
    Cs = solve_by_gauss(matrix)

    if (display):
        res_str = str(-xs[0] + u_n(xs[0], 0))
        for i in range(n):
            if (Cs[i] >= 0):
                res_str += '+'
            temp_str = '{0} * {1}'.format(Cs[i], u_n(None, i + 1, display=True))
            res_str += temp_str
        return res_str
    else:

        def func(x):
            sum = u_n(x, 0)  # -x + u_n(x, 0)  # -x получается при применении оператора L
            for i in range(n):
                sum += Cs[i] * u_n(x, i + 1)
            return sum

        return func
