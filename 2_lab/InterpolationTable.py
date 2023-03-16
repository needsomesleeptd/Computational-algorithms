import matplotlib.pyplot as plt
import math
import copy
EPS = 1e-4
import numpy as np

class InterpolationTable:
    def __init__(self):
        self.Data = []
        self.n = 0
        self.x = 0

    def fit(self, dataset: list, n: int, x: float):
        self.Data = dataset
        self.n = n
        self.x = x

    def is_float(self, num: str):
        num = num.replace("-", "", 1)
        num = num.replace(".", "", 1)
        return num.isnumeric()

    def is_row_valid_float(self, row: list):
        for i in range(len(row)):
            if (not (self.is_float(row[i]))):
                return False
        return True

    def print_table(self):
        spaces_cnt = 3
        for row in self.Data:
            template = "{:^8} " * len(row)
            print(template.format(*row))


    def select_rows_hermit(self,x:float):
        if (len(self.Data) < self.n + 1):
            return None
        self.Data.sort(key=lambda data: data[0])
        selected_rows = []

        min_delta_row_index = 0
        min_delta = abs(self.Data[0][0] - x)
        for i in range(len(self.Data)):
            if (abs(self.Data[i][0] - x) < min_delta):
                min_delta = abs(self.Data[i][0] - x)
                min_delta_row_index = i
        l_bound = min_delta_row_index
        r_bound = min_delta_row_index + 1
        while (len(selected_rows) < self.n + 1):
            if (l_bound >= 0 and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[l_bound])
                if (len(selected_rows) < self.n + 1):
                    selected_rows.append(self.Data[l_bound])
                l_bound -= 1
            if (r_bound < len(self.Data) and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[r_bound])
                if (len(selected_rows) < self.n + 1):
                    selected_rows.append(self.Data[r_bound])
                r_bound += 1
        selected_rows.sort(key=lambda x: x[0])
        return selected_rows

    def select_rows_neuton(self,x:float):
        if (len(self.Data) < self.n + 1):
            return None
        self.Data.sort(key=lambda data: data[0])
        selected_rows = []

        min_delta_row_index = 0
        min_delta = abs(self.Data[0][0] - x)
        for i in range(len(self.Data)):
            if (abs(self.Data[i][0] - x) < min_delta):
                min_delta = abs(self.Data[i][0] - x)
                min_delta_row_index = i
        l_bound = min_delta_row_index
        r_bound = min_delta_row_index + 1
        while (len(selected_rows) < self.n + 1):
            if (l_bound >= 0 and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[l_bound])
                l_bound -= 1
            if (r_bound < len(self.Data) and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[r_bound])
                r_bound += 1
        selected_rows.sort(key=lambda x: x[0])
        return selected_rows

    def neuton_interpolation(self,x = None):
        #print("Neuton:\n")
        if (x == None):
            x = self.x

        chosen_dots = self.select_rows_neuton(x)
        if (chosen_dots == None):
            print("Данных недостаточно для образования полинома данной степени")
            return None
        #print(chosen_dots)
        part_sums = self.get_part_sums_neuton(chosen_dots)
        #print(part_sums)

        # (part_sums[0], part_sums[1]) = (part_sums[1], part_sums[0])
        def interpolation_on_point(x: int):
            inter_sum = part_sums[1]
            for i in range(2, len(part_sums)):
                inter_mul = 1
                for j in range(i - 1):
                    inter_mul *= x - chosen_dots[j][0]
                inter_sum += inter_mul * part_sums[i]
            return inter_sum

        return interpolation_on_point

    def draw_graphs(self, interpolation_func,function_name:str,show = True):
        x = [self.Data[i][0] for i in range(len(self.Data))]
        y = [self.Data[i][1] for i in range(len(self.Data))]
        if (show):
            plt.plot(x, y, label="real_function", marker='*')
        xs_inter = np.linspace(x[0] - 0.5,x[-1] + 0.5,endpoint=True,num=150)
        if (interpolation_func != None):
            y_interpolated = list(map(interpolation_func, xs_inter))
            plt.plot(xs_inter, y_interpolated, label=function_name, marker='.')
        plt.legend()
        plt.grid()
        if (show):
            plt.show()

    def draw_graphs_neuton(self, interpolation_func, function_name: str,show = True):
        x = [self.Data[i][0] for i in range(len(self.Data))]
        y = [self.Data[i][1] for i in range(len(self.Data))]
        if (show):
            plt.plot(x, y, label="real_function", marker='*')
        xs_inter = np.linspace(x[0] - 0.5, x[-1] + 0.5, endpoint=True, num=150)
        if (interpolation_func != None):
            y_funcs = list(map(interpolation_func, xs_inter))
            y_interpolated = [y_funcs[i](xs_inter[i]) for i in range(len(y_funcs))]
            plt.plot(xs_inter, y_interpolated, label=function_name, marker='.')
        plt.legend()
        plt.grid()
        if (show):
            plt.show()

    def scan_data(self):
        database = []

        print("Введите таблицу значений(Для окончания ввода введите все -)")
        row = input().split()
        while self.is_row_valid_float(row):
            row = list(map(lambda x: float(x), row))
            database.append(row)
            row = input().split()
        print("Введите степень аппроксимирующих полиномов")
        n = int(input())  # Степень
        print("Введите значения текущего x")
        cur_x = float(input())
        if (len(database) < n + 1):
            print("Нехватает данных для верной интерполяции")
            return None
        self.fit(database, n, cur_x)


    def get_part_sums_hermit(self, chosen_dots:list):
        n = len(chosen_dots)
        part_sums = [[0 for i in range(n + 1)] for i in range(n + 2)]  # n*n dim
        for i in range(n):
            part_sums[i][0] = chosen_dots[i][0]
            part_sums[i][1] = chosen_dots[i][1]
            part_sums[i][2] = chosen_dots[i][2]
        row_iter = n - 1
        col_count = 1
        i = 0

        for j in range(2, n + 1):
            while (i < row_iter):
                if  not(abs(part_sums[i][0] - part_sums[i + col_count][0]) < EPS and abs(part_sums[i][j - 1] - part_sums[i + 1][j - 1]) < EPS):
                    part_sums[i][j] = (part_sums[i][j - 1] - part_sums[i + 1][j - 1]) / (
                            part_sums[i][0] - part_sums[i + col_count][0])
                i += 1
            row_iter -= 1
            i = 0
            col_count += 1
        return part_sums[0]

    def get_part_sums_neuton(self, chosen_dots:list):
        n = len(chosen_dots)
        part_sums = [[0 for i in range(n + 1)] for i in range(n + 2)]  # n*n dim
        for i in range(n):
            part_sums[i][0] = chosen_dots[i][0]
            part_sums[i][1] = chosen_dots[i][1]
        row_iter = n - 1
        col_count = 1
        i = 0

        for j in range(2, n + 1):
            while (i < row_iter):
                    part_sums[i][j] = (part_sums[i][j - 1] - part_sums[i + 1][j - 1]) / (
                            part_sums[i][0] - part_sums[i + col_count][0])
                    i += 1
            row_iter -= 1
            i = 0
            col_count += 1
        return part_sums[0]


    def read_from_file(self, filename:str):
        f = open(filename, "r")
        database = []

        row = f.readline().split()
        while self.is_row_valid_float(row):
            row = list(map(lambda x: float(x), row))
            database.append(row)
            row = f.readline().split()

        n = int(f.readline())  # Степень

        cur_x = float(f.readline())
        if (len(database) < n + 1):
            print("Нехватает данных для верной интерполяции")
            return None
        self.fit(database, n, cur_x)
        f.close()

    def hermit_interpolation(self,x = None):
       # print("Hermit:")
        if (x == None):
            x = self.x
        chosen_dots = self.select_rows_hermit(x)

        if (chosen_dots == None):
            print("Данных недостаточно для образования полинома данной степени")
            return None
       # print(chosen_dots)
        part_sums = self.get_part_sums_hermit(chosen_dots)
       # print(part_sums)
        # (part_sums[0], part_sums[1]) = (part_sums[1], part_sums[0])
        def interpolation_on_point(x: int):
            inter_sum = part_sums[1]
            for i in range(2, len(part_sums)):
                inter_mul = 1
                for j in range(i - 1):
                    inter_mul *= x - chosen_dots[j][0]
                inter_sum += inter_mul * part_sums[i]
            return inter_sum

        return interpolation_on_point

    def reverse_interpolation_table(self):
        n_data = len(self.Data)
        reversed_interpolation = InterpolationTable()
        data_reversed = copy.deepcopy(self.Data)
        reversed_interpolation.fit(data_reversed,self.n,self.x)
        reversed_interpolation.rev_function()
        x_rev_interpolation  = 0
        for i in range(1,n_data):
            if (reversed_interpolation.Data[i][0]*reversed_interpolation.Data[i - 1][0] < 0):
                x_rev_interpolation = (reversed_interpolation.Data[i][0] + reversed_interpolation.Data[i - 1][0]) / 2
        reversed_interpolation.x = x_rev_interpolation
        return reversed_interpolation
    def rev_function(self):
        for i in range(len(self.Data)):
            self.Data[i][0],self.Data[i][1] = self.Data[i][1],self.Data[i][0]
            if (len(self.Data[0]) > 2):
                self.Data[i][2] = 1 / self.Data[i][2]

    def get_xy(self):
        xs = []
        ys = []
        for i in range(len(self.Data)):
            xs.append(self.Data[i][0])
            ys.append(self.Data[i][1])
        return xs, ys


def solve_equations_table(first_table,secound_table):
    first_table.rev_function()
    function_difference = []
    for i in range(len(first_table.Data)):
            second_y = secound_table.neuton_interpolation(first_table.Data[i][0])
            function_difference.append([first_table.Data[i][0],first_table.Data[i][1] - second_y(first_table.Data[i][0])])


    i = 0
    while i < len(function_difference) - 1:
        if function_difference[i + 1][1] > function_difference[i][1]:
            function_difference.pop(i)
        else:
            i += 1

    new_interpolation_table = InterpolationTable()
    new_interpolation_table.fit(function_difference,secound_table.n,secound_table.x)

    rev_interpolation_table_equation =  new_interpolation_table.reverse_interpolation_table()
    first_table.rev_function()
    return rev_interpolation_table_equation






















