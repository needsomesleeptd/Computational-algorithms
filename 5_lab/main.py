import sympy as sym
import numpy as np
from scipy.linalg import solve
from math import *

MAX_VAL = 10000

def make_identity(matrix):
    # перебор строк в обратном порядке
    for nrow in range(len(matrix) - 1, 0, -1):
        row = matrix[nrow]
        for upper_row in matrix[:nrow]:
            factor = upper_row[nrow]
            upper_row -= factor * row
    return matrix


def solve_by_gauss(matrix):
    for nrow in range(len(matrix)):
        # nrow равен номеру строки
        # np.argmax возвращает номер строки с максимальным элементом в уменьшенной матрице
        # которая начинается со строки nrow. Поэтому нужно прибавить nrow к результату
        pivot = nrow + np.argmax(abs(matrix[nrow:, nrow]))
        if pivot != nrow:
            # swap
            # matrix[nrow], matrix[pivot] = matrix[pivot], matrix[nrow] - не работает.
            # нужно переставлять строки именно так, как написано ниже
            matrix[[nrow, pivot]] = matrix[[pivot, nrow]]
        row = matrix[nrow]
        divider = row[nrow]  # диагональный элемент
        if abs(divider) < 1e-20:
            # почти нуль на диагонали. Продолжать не имеет смысла, результат счёта неустойчив
            raise ValueError(f"Матрица несовместна. Максимальный элемент в столбце {nrow}: {divider:.3g}")

        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in matrix[nrow + 1:]:
            factor = lower_row[nrow]  # элемент строки в колонке nrow
            lower_row -= factor * row  # вычитаем, чтобы получить ноль в колонке nrow
        # приводим к диагональному виду
    make_identity(matrix)
    return matrix[:, -1]


def get_YakibianMatrix(vars):


    return np.array([
        np.array([2 * vars[0], 2 * vars[1], 2 * vars[2], 1 - (vars[0] ** 2 + vars[1] ** 2 + vars[2] ** 2)],
                 dtype=np.float64),
        np.array([4 * vars[0], 2 * vars[1], -4, -(2 * (vars[0] ** 2) + vars[1] ** 2 - 4 * vars[2])], dtype=np.float64),
        np.array([6 * vars[0], -4, 2 * vars[2], -(3 * (vars[0] ** 2) - 4 * vars[1] + vars[2] ** 2)], dtype=np.float64),
    ])


def integration_by_trapezoid(a, b, f, n=10000):
    if b > a:
        h = (b - a) / float(n)
    else:
        h = (a - b) / float(n)
    #h = (b - a) / float(n)
    interg_sum = 0
    interg_sum +=  0.5 *h * f(a)
    interg_sum+= 0.5 * h * f(b)
    for i in range(1, n):
        interg_sum += f(a + i * h)
    #interg_sum /= 2
    if (b < a):
        interg_sum *= -1
    return interg_sum  * h


def laplas_function(x):
    laplas =   integration_by_trapezoid(0, x, lambda z: exp(-(z ** 2) / 2)) / sqrt(2 * pi) # Добавить 2 если нужно
    return laplas


def bisection(f, a, b, eps=0.001):
    if (f(a) * f(b) > 0):
        print("Неверно выбраны правая и левая части\n")
        return

    c = a
    while (abs(b - a) >= eps):

        # Find middle point
        c = (a + b) / 2

        # Check if middle point is root
        if (f(c) == 0.0):
            break

        # Decide the side to repeat the steps
        if (f(c) * f(a) < 0):
            b = c
        else:
            a = c

    return c


if __name__ == '__main__':
    variables = np.full((3), 1, dtype=np.float64)
    new_variables = np.full((3), 1, dtype=np.float64)

    EPS = 1e-4

    while (np.any(np.abs(new_variables) > EPS)):
        matrix_s_1 = get_YakibianMatrix(variables)
        new_variables = solve_by_gauss(matrix_s_1)
        variables += new_variables

    print(f'variables = {variables}')
    print(get_YakibianMatrix(variables))



    val = float(input('Введите значение функции лапласа для получения значения x:\n'))
    #ans = laplas_function(val)
    ans = bisection(lambda x: laplas_function(x) - val, -MAX_VAL ,MAX_VAL)
    print(f'Значение x  для функции Лапласа: {ans}')
    print(f'Значение Лапласа в данной точке: {laplas_function(ans)}')
    #Task 03






