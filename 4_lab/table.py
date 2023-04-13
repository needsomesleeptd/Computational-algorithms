from Point import Point
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt


def dot(f, w, p, x, n):
    su = 0
    for i in range(n):
        su += p[i] * f[i] * w[i]
    return su


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
        if abs(divider) < 1e-10:
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
    return matrix[:,-1]


class Table:
    def __init__(self) -> None:
        self.weights = np.array([])
        self.table = np.array([])
        self.feature_count = 0

    def generate_random_dots(self, dots_count, equal = False,l=-100, r=100,):
        self.table = np.random.uniform(l, r, [dots_count, 2])
        if (equal):
            self.weights = np.ones(dots_count)
        # for i in range(len(self.table)):
        #     self.table[i][2] = self.table[i][2][0]
        # for j in range(len(self.table[0])):
        #  self.table[i][j] = Point(*self.table[i][j])

    def fit(self, table, feature_count, weights=None):
        self.weights = weights
        self.table = table
        self.feature_count = feature_count
        if (weights == None):
            self.weights = np.random.uniform(-100, 100, len(weights))
    def read_weights(self):
        self.weights = list(map(float,input().split()))


    def print(self):
        spaces_cnt = 3
        headers = ['x', 'y', 'p']
        template = "{:^8} " * len(headers)
        print(template.format(*headers))
        i = 0
        for row in self.table:
            vals = list(map(lambda x: str(x), row.round(2)))
            print(f'{vals}', self.weights[i])
            print()
            i += 1

    def get_system(self, n):
        n+=1
        Slau = np.zeros([n, n + 1])
        for i in range(n):
            for j in range(n):
                sums = 0
                for k in range(len(self.table)):
                    sums += self.weights[k] * self.table[k][0] ** (i + j)
                Slau[i][j] = sums

            sums = 0
            for k in range(len(self.table)):
                sums += self.weights[k] * self.table[k][1] * self.table[k][0] ** i
            Slau[i][n] = sums

        return Slau

    def get_function(self, n):  # n - степень полинома
        system = self.get_system(n)
        for i in range(len(system)):
            for j in range(len(system[0])):
                print(system[i][j])
            print()
        a = solve_by_gauss(system)

        def approximate_1D(x):
            su = 0
            for i in range(n + 1):
                su += a[i] * x ** i
            return su

        return approximate_1D

    def plot_graph(self, func, nums=100):
        x = [dot[0] for dot in self.table]
        y = [dot[1] for dot in self.table]
        plt.scatter(x, y)
        l = min(x)
        r = max(x)
        xs = np.linspace(l, r)
        ys = [func(x) for x in xs]
        plt.xlabel("x")
        plt.xlabel("y")
        plt.grid()
        plt.plot(xs, ys)


    def set_weights(self,value):
        self.weights = np.ones(len(self.table))
        self.weights.fill(value)

    def dual_plot(self,n):
        func = self.get_function(n)
        save_weights = np.copy(self.weights)
        self.plot_graph(func)
        self.set_weights(1)
        func = self.get_function(n)
        self.plot_graph(func)
        self.weights = save_weights
        plt.plot()



