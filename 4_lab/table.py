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
    return matrix[:, -1]


def get_value_2d(x,y,powx,powy):
    return x ** powx * y ** powy

class Table:
    def __init__(self) -> None:
        self.weights = np.array([])
        self.table = np.array([])
        self.feature_count = 0

    def generate_random_dots(self, dots_count,count=2, equal = False,l=-100, r=100):
        self.table = np.random.uniform(l, r, [dots_count, count])
        if (equal):
            self.weights = np.ones(dots_count)
        else:
            self.weights = np.random.uniform(0.1, 100, len(self.table))
        self.weights = np.sort(self.weights)
        # for i in range(len(self.table)):
        #     self.table[i][2] = self.table[i][2][0]
        # for j in range(len(self.table[0])):
        #  self.table[i][j] = Point(*self.table[i][j])

    def fit(self, table, feature_count, weights=None):
        self.weights = weights
        self.table = table
        self.feature_count = feature_count
        if (weights == None):
            self.weights = np.random.uniform(0.1, 100, len(table))
    def read_weights(self):
        self.weights = np.array([])
        for i in range(len(self.table)):
            value = float(input("Enter weight for dot: {0} {1}: ".format(self.table[i][0],self.table[i][1])))
            self.weights = np.append(self.weights,value)



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

    def get_function_1d(self, n):  # n - степень полинома
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

    def plot_graph(self, func, label,nums=100):
        x = [dot[0] for dot in self.table]
        y = [dot[1] for dot in self.table]
        plt.scatter(x, y)
        l = min(x)
        r = max(x)
        xs = np.linspace(l, r)
        ys = [func(x) for x in xs]
        plt.plot(xs, ys,label=label)


    def set_weights(self,value):
        self.weights = np.ones(len(self.table))
        self.weights.fill(value)

    def dual_plot(self,n,get_func):
        func = get_func(n)
        save_weights = np.copy(self.weights)
        self.plot_graph(func,"custom weights")
        self.set_weights(1)
        func = get_func(n)
        self.plot_graph(func,"equal weights")
        self.weights = save_weights
        plt.xlabel("x")
        plt.xlabel("y")
        plt.grid()
        plt.legend()


    def elem_count_2d(self,n):
        return ((n+1) * (n+2)) / 2 # Как в полиноме Ньютона

    def get_value_2d(self,row,powx,powy):
        return row[0] ** powx * row[1] ** powy

    def get_system_2d(self,n = 1):
        #n += 1
        #rows_count = self.elem_count_2d(n)
        #slau = np.ones([rows_count, rows_count + 1])
        #slau = slau.astype("float64")

        a = list()
        b = list()

        for i in range(n + 1):
            for j in range(n + 1 - i):
                a_row = []
                for k in range(n + 1):
                    for t in range(n + 1 - k):
                        a_row.append(sum(list(map(
                            lambda row: self.get_value_2d(row, k + i, t + j) * self.weights[i],
                            self.table
                        ))))
                a.append(a_row)
                b.append(sum(list(map(
                    lambda row: self.get_value_2d(row, i, j) * row[2] * self.weights[i],
                    self.table
                ))))
        slau = list()
        for i in range(len(a)):
            slau.append(a[i])
            slau[i].append(b[i])
        return slau

    def get_function_2d(self, n=1):

        slau = self.get_system_2d(n)
        #print("\nМатрица СЛАУ:")
        #printMatrix(slau)

        slau = np.array([np.array(xi) for xi in slau]) #convert to np.array
        c = solve_by_gauss(slau)
        #printCoeff(c)

        def approximateFunction_2D(x, y):
            result = 0
            c_index = 0
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    result += c[c_index] * get_value_2d(x, y, i, j)
                    c_index += 1
            return result

        return approximateFunction_2D

    def drawGraficBy_AproxFunction_2D(self,approximateFuction,delta =30):
        x = [dot[0] for dot in self.table]
        y = [dot[1] for dot in self.table]
        z = [dot[2] for dot in self.table]

        x_min,x_max = min(x),max(x)
        y_min, y_max = min(x), max(x)

        #minX, maxX = getIntervalX(pointTable)
        #minY, maxY = getIntervalY(pointTable)

        xValues = np.linspace(x_min - delta, x_max + delta, 60)
        yValues = np.linspace(y_min - delta, y_max + delta, 60)
        zValues = [approximateFuction(xValues[i], yValues[i]) for i in range(len(xValues))]

        def make_2D_matrix():
            # Создаем двумерную матрицу-сетку
            xGrid, yGrid = np.meshgrid(xValues, yValues)
            # В узлах рассчитываем значение функции
            zGrid = np.array([
                [
                    approximateFuction(
                        xGrid[i][j],
                        yGrid[i][j],
                    ) for j in range(len(xValues))
                ] for i in range(len(yValues))
            ])
            return xGrid, yGrid, zGrid

        fig = plt.figure("График функции, полученный аппроксимации наименьших квадратов")
        xpoints, ypoints, zpoints = x,y,z
        axes = fig.add_subplot(projection='3d')
        axes.scatter(xpoints, ypoints, zpoints, c='red')
        axes.set_xlabel('OX')
        axes.set_ylabel('OY')
        axes.set_zlabel('OZ')
        xValues, yValues, zValues = make_2D_matrix()
        axes.plot_surface(xValues, yValues, zValues)
        plt.show()


