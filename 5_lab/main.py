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
    # h = (b - a) / float(n)
    interg_sum = 0
    interg_sum += 0.5 * h * f(a)
    interg_sum += 0.5 * h * f(b)
    for i in range(1, n):
        interg_sum += f(a + i * h)
    # interg_sum /= 2
    if (b < a):
        interg_sum *= -1
    return interg_sum * h


def laplas_function(x):
    laplas = integration_by_trapezoid(0, x, lambda z: exp(-(z ** 2) / 2)) / sqrt(2 * pi)  # Добавить 2 если нужно
    return laplas


def bisection(f, a, b, EPS=0.001):
    if (f(a) * f(b) > 0):
        print("Неверно выбраны правая и левая части\n")
        return

    c = a
    while (abs(b - a) >= EPS):

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

    def eps(C: float, B: float, A: float, prev_eps: float):
        return C / (B - A * prev_eps)

    def etta(A: float, prev_etta: float, F: float, B: float, eps):
        return (A * prev_etta + F) / (B - A * eps)



    def get_A(x):
        return 1

    def get_C(x):
        return 1

    def get_B(x,h):
        return get_A(x) + get_C(x) + (h ** 2) * (3 * x ** 2)

    def get_Y(x): # Исходная функция соответствующая начальным условиям
        return 1 + x * 2
    def get_right_Function(x,y): #Правая часть изначального уравнения
        return x **2 + y **3

    def get_F(xs,y,index,h):
        return get_Y(xs[index - 1]) - 2 * get_Y(xs[index]) + get_Y(xs[index + 1]) - (h**2) * get_right_Function(xs[index],y)


    def get_y_delta(a, b, n=10000): # need to pass y here
        h = (b - a) / n
        ys = [1 for i in range(n + 1)]
        xs = [0 for i in range(n + 1)]
        matrix = np.zeros((n + 1, n + 1), dtype=np.float64)
        epss = [0]
        ettas = [0]
        for i in range(1,n + 1):
            cur_eps = eps(get_C(xs[i]), get_B(xs[i],h), get_A(xs[i]), epss[i - 1])
            cur_etta = etta(get_A(xs[i]), ettas[i - 1], get_F(xs[i],y,i,h), get_B(xs[i],h), epss[i - 1])
            epss.append(cur_eps)
            ettas.append(cur_etta)
        for i in range(n - 1, 0, -1):
            ys[i] = epss[i] * ys[i + 1] + ettas[i]
        return ys

    def solve_diff_eq(x,a,b,EPS = 1e-4,n=10000):
        ys = [1 for i in range(n + 1)] # calculated one from get_Y
        ys_delta = [1 for i in range(n + 1)]
        while (any(ys_delta > EPS)):
            ys_delta = get_y_delta(a,b,n)
            ys += ys_delta








    def get_Cs(self, start_c_init, end_c_init):
        dots_count = len(self.xs)
        epss = [0]
        ettas = [start_c_init / 2]
        Cs = [0 for i in range(dots_count)]
        Cs[0] = start_c_init / 2
        Cs[-1] = end_c_init / 2
        hs = [self.xs[i] - self.xs[i - 1] for i in range(1, dots_count)]

        for i in range(1, dots_count - 1):
            h1 = hs[i]
            h2 = hs[i - 1]
            cur_B = -2 * (h2 + h1)
            cur_A = h2
            cur_D = h1
            cur_F = F(self.ys[i + 1], self.ys[i], self.ys[i - 1], h1, h2)
            cur_eps = EPS(cur_D, cur_B, cur_A, epss[i - 1])
            cur_etta = etta(cur_A, ettas[i - 1], cur_F, cur_B, epss[i - 1])
            epss.append(cur_eps)
            ettas.append(cur_etta)

        for i in range(dots_count - 2, 0, -1):
            Cs[i] = epss[i] * Cs[i + 1] + ettas[i]
        return Cs


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
    # ans = laplas_function(val)
    ans = bisection(lambda x: laplas_function(x) - val, -MAX_VAL, MAX_VAL)
    print(f'Значение x  для функции Лапласа: {ans}')
    print(f'Значение Лапласа в данной точке: {laplas_function(ans)}')

    # Task 03
