import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MultiDimensional import *
from InterpolationTable import *


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


def get_polynom_func(n):
    def polynom_func(x):
        f = x
        s = 1
        if (n == 0):
            return 1
        elif (n == 1):
            return x

        for i in range(2, n + 1):
            new_polynom = 1 / (i) * ((2 * i - 1) * x * f - (i - 1) * s)
            # print(f'new_polynom = {new_polynom}')
            s = f
            f = new_polynom

        return f

    return polynom_func


class Lishandr_Polynom:
    def __init__(self, n, polinoms):
        self.n = n
        self.func = None


def range_of_Polinoms(n):
    polinoms = []
    polinom = Lishandr_Polynom(0, polinoms)
    polinoms.append(polinom)
    for i in range(1, n):
        polinom = Lishandr_Polynom(i, polinoms)
        polinoms.append(polinom)

    return polinoms


def get_next_Lishandr_Polinom(polinoms, index):
    def new_polinom(x):
        return 1 / index * (polinoms[index - 1](x - 1) * (2 * index - 1) - (index - 1) * polinoms[index - 2](x))

    return new_polinom


def bisection(f, a, b, EPS=0.01, EPS_EXTRA=0.1):
    if (f(a) * f(b) > 0):
        # print("Неверно выбраны правая и левая части\n")
        return None

    c = a

    while (abs(b - a) > EPS * abs(c) + EPS):

        c = (a + b) / 2
        if (f(c) == 0.0):
            return c
        if (f(c) * f(a) < 0):
            b = c
        else:
            a = c

    return c


def find_roots(a, b, polinom, n):  # index == Кол-во корней
    split_count = n
    roots = []
    while (len(roots) != n):
        roots = []
        splitters = np.linspace(a, b, num=split_count + 1)
        for i in range(len(splitters) - 1):
            root = bisection(polinom, splitters[i], splitters[i + 1])
            if (root != None):
                roots.append(root)

        split_count = split_count * 2
    return roots


def build_matrix(roots, n):  # Ai всего N
    matrix = np.array([], dtype=np.float64)
    for i in range(n):
        row = np.array([], dtype=np.float64)
        for j in range(n):
            row = np.append(row, roots[j] ** i)
        if (i % 2 == 0):
            row = np.append(row, 2 / (i + 1))
        else:
            row = np.append(row, 0)
        if (len(matrix) == 0):
            matrix = row
        else:
            matrix = np.vstack((matrix, row))
    # matrix.reshape()
    return matrix


def intergral_by_Gauss(f, a, b, n):
    # polynoms = range_of_Polinoms(n)
    polynom_func = get_polynom_func(n)
    roots = find_roots(-1, 1, polynom_func, n=n)  # Заметим что все корни многочленов Лежандра лежат от -1 до 1
    matrix = build_matrix(roots, n)
    As = solve_by_gauss(matrix)
    su = 0
    for i in range(n):
        cur_x = (b + a) / 2 + (b - a) / 2 * roots[i]
        su += As[i] * f(cur_x)
    return (b - a) / 2 * su


def integral_by_Simpson(vals, a, b, n):
    h = (b - a) / n
    res = 0
    for i in range((len(vals) - 1) // 2):
        res += vals[2 * i] + vals[2 * i + 1] + vals[2 * i + 2]

    return (h / 3) * res


'''def get_multiple_integral(Table,a,b,n,n_gauss,n_simpson):
    intergrals_x = []
    h = (b - a) / n
    x = Table[:, 0]
    size = np.shape(Table)[1]
    for i in range(2,size):
        get_'''


class Plane():
    def __init__(self, Table_df):
        self.table = Table_df

    def get_plane_equation_params(self):
        P = [self.table[0][1], self.table[1][0], self.table[1][1]]
        Q = [self.table[0][2], self.table[2][0], self.table[2][2]]
        R = [self.table[0][3], self.table[4][0], self.table[3][4]]

        x1, y1, z1 = P
        x2, y2, z2 = Q
        x3, y3, z3 = R
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x1 - b * y1 - c * z1)
        return a, b, c, d

    def get_plane_value(self, x, y):
        a, b, c, d = self.get_plane_equation_params()
        return -(d + a * x + b * y) / c


def get_integral(f, ax, bx, ay, by, n, n_gauss, n_simpson):
    hx = (bx - ax) / n

    integrals = []
    xs = []

    dataset = []
    for i in range(n):
        cur_x = ax + hx * i
        integral = intergral_by_Gauss(lambda y: f(cur_x, y), ax, bx, n_gauss)
        integrals.append(integral)

    return integral_by_Simpson(integrals,ay,by,n_simpson)


if __name__ == '__main__':
    Table = np.genfromtxt("task_func.txt", delimiter="\t")
    # Table[0] = Table[0][~np.isnan(Table)]
    # Table_df= np.log(Table_df.loc[:, Table_df.columns != "y\\x"])
    # print(Table,Table.shape)
    Table[1:, 1:] = np.log(Table[1:, 1:])

    # print(Table)

    # multidim_table = MultiDim()
    # multidim_table.fit(Table[1:,1:],Table[1,:],Table[:,1],Table[1:,1:])
    # print(multidim_table.MultidimensionalInterpolationNeuton(2,2,0.1,0.1))
    # print(plane.get_plane_value(Table[0][1],Table[1][0]))
    # print(plane.get_plane_value(Table[0][3], Table[2][0]),Table[3,2])
    # findMultidimensionalInterpolation()
    '''xx, yy = np.meshgrid(Table[1:, :], Table[:, 1:])
    print(xx, yy)
    z = Table[1:, 1:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(Table[1:, 0], Table[0, 1:])
    ax.plot_surface(X, Y, Table[1:, 1:])
    plt.show()'''
    plane = Plane(Table)
    multidim = MultiDim()
    multidim.fit(Table[1:,1:],Table[1:,0], Table[0, 1:],None)
    #x_val = multidim.MultidimensionalInterpolationNeuton(10,10,2,1)
    f = lambda x,y: multidim.MultidimensionalInterpolationNeuton(12,12,x,y)
    #f = lambda x,y: plane.get_plane_value(x,y)
    print(np.exp(get_integral(f,0,1,0,1,15,15,15)))

    #x = [i for i in range(100)]
    #y = [i for i in range(100)]





    #print(f(0,0))
    #print(np.exp(get_integral(plane.get_plane_value, ax=0,bx=2,ay=0,by=2,n=10,n_simpson=10,n_gauss=10)))
    # print(np.exp(plane.get_plane_value(0.5,0.1)))
    # Some tests to find that everything is working
    # func = lambda x: x ** 2 + x ** 3 / 4 + cos(x) + sin(x)

    # ans  = intergral_by_Gauss(func,-1,1,3)
    # x = 3
    # ans = get_polynom_value(x,2)
    # ans2 = 1/2 * (5 *x **3 - 3 *x)
    # ans2 =1 /2 * (3*x*x - 1)
    # print(intergral_by_Gauss(func, -1, 0, 300))
    # func_0 = get_polynom_func(0)
    # func_1 = get_polynom_func(1)
    # func_2 = get_polynom_func(2)
    # func_3 = get_polynom_func(3)
    # print(func_0(200),func_1(200),func_2(200),func_3(200))

    # print(ans,'',ans2)
