from table import solve_by_gauss
import numpy as np


def u_n(x, n, display=False):
    if (display):
        return '(1 - x) * (x ** {0})'.format(n)
    return (1 - x) * (x ** n)


def alpha(xs):
    sum = 0
    for x in xs:
        sum += -3 * x ** 2 + 2 * x - 2
    return sum


def betta(xs):
    sum = 0
    for x in xs:
        sum += 3 * x ** 2 - 4 * x ** 3 - 6 * x + 2
    return sum


def tetta(xs):
    sum = 0
    for x in xs:
        sum += 6 * x - 12 * x ** 2 + 4 * x ** 4 - 5 * x ** 5
    return sum


def create_matrix(xs, n, params=[alpha, betta, tetta]):  # n - кол-во параметров матрицы params == alpha,betta etc...
    matrix = [[0 for i in range(n + 1)] for j in range(n)]
    for i in range(n):
        for j in range(n + 1):
            for x in (xs):
                if (j == n):
                    matrix[i][j] += (4 * x - 1) * params[i]([x])
                elif (i == j):
                    matrix[i][j] += params[i]([x]) ** 2
                else:
                    matrix[i][j] += params[i]([x]) * params[j]([x])
    # print(matrix)
    return np.array(matrix)


def get_func(xs, n, params=[alpha, betta, tetta], display=False):
    matrix = create_matrix(xs, n, params)
    Cs = solve_by_gauss(matrix)

    if (display):
        res_str = '1 - x'  # str(-xs[0] + u_n(xs[0], 0))
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
