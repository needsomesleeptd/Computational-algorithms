import matplotlib.pyplot as plt
import math
import copy
EPS = 1e-4

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

    def select_rows(self,polynom_type:str):
        self.Data.sort(key=lambda x: x[0])
        selected_rows = []

        min_delta_row_index = 0
        min_delta = abs(self.Data[0][0] - self.x)
        for i in range(len(self.Data)):
            if (abs(self.Data[i][0] - self.x) < min_delta):
                min_delta = abs(self.Data[i][0] - self.x)
                min_delta_row_index = i
        l_bound = min_delta_row_index
        r_bound = min_delta_row_index + 1
        while (len(selected_rows) < self.n + 1):
            if (l_bound >= 0 and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[l_bound])
                if (len(selected_rows) < self.n + 1 and polynom_type == "hermit"):
                    selected_rows.append(self.Data[l_bound])
                l_bound -= 1
            if (r_bound < len(self.Data) and len(selected_rows) < self.n + 1):

                selected_rows.append(self.Data[r_bound])
                if (len(selected_rows) < self.n + 1 and polynom_type == "hermit"):
                    selected_rows.append(self.Data[r_bound])
                r_bound += 1
        selected_rows.sort(key=lambda x: x[0])
        return selected_rows

    def neuton_interpolation(self,x = None):
        #print("Neuton:\n")


        chosen_dots = self.select_rows("neuton")
        #print(chosen_dots)
        part_sums = self.get_part_sums(chosen_dots, "neuton")
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

    def draw_graphs(self, interpolation_func,function_name:str):
        x = [self.Data[i][0] for i in range(len(self.Data))]
        y = [self.Data[i][1] for i in range(len(self.Data))]
        plt.plot(x, y, label="real_function", marker='*')
        if (interpolation_func != None):
            y_interpolated = list(map(interpolation_func, x))
            plt.plot(x, y_interpolated, label=function_name, marker='.')
        plt.legend()
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


    def get_part_sums(self, chosen_dots:list, interpolation_func:str):
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
                if interpolation_func == "hermit" and abs(part_sums[i][0] - part_sums[i + col_count][0]) < EPS and abs(part_sums[i][j - 1] - part_sums[i + 1][j - 1]) < EPS:
                    part_sums[i][j] = self.Data[i][2]
                else:
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

    def hermit_interpolation(self):
       # print("Hermit:")
        chosen_dots = self.select_rows("hermit")
       # print(chosen_dots)
        part_sums = self.get_part_sums(chosen_dots, "hermit")
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

    def reverse_interpolation(self,interpolation_func:str):
        n_data = len(self.Data)
        data_reversed =copy.deepcopy(self.Data)
        for i in range(n_data):
            data_reversed[i][0],data_reversed[i][1] =  data_reversed[i][1],data_reversed[i][0]
            if (len(self.Data[0]) > 2):
                data_reversed[i][2] = 1/data_reversed[i][2]
        rev_interpolation_func = None
        save_data = self.Data
        self.Data = data_reversed
        x_rev_interpolation = 0
        for i in range(1,n_data):
            if (self.Data[i][0]*self.Data[i - 1][0] < 0):
                x_rev_interpolation = (self.Data[i][0] + self.Data[i - 1][0]) / 2
        x_save = self.x
        self.x = x_rev_interpolation
        if interpolation_func == "neuton":
            rev_interpolation_func =  self.neuton_interpolation()
        elif interpolation_func == "hermit":
            rev_interpolation_func = self.hermit_interpolation()
        self.x = x_save

        self.Data = save_data
        return rev_interpolation_func
    def rev_function(self):
        for i in range(len(self.Data)):
            self.Data[i][0],self.Data[i][1] = self.Data[i][1],self.Data[i][0]
            if (len(self.Data[0]) > 2):
                self.Data[i][2] = 1 / self.Data[i][2]


    def solve_equations(self,secound_table):
        self.rev_function()
        save_x = secound_table.x
        secound_table_interpol_func = secound_table.neuton_interpolation()
        function_difference = []
        for i in range(len(self.Data)):
                secound_table.x = self.Data[i][0]
                secound_y = secound_table.neuton_interpolation()
                function_difference.append([self.Data[i][0],self.Data[i][1] - secound_y(self.Data[i][0])])

        new_interpolation_table = InterpolationTable()
        new_interpolation_table.fit(function_difference,self.n,self.x)


        rev_interpolation = new_interpolation_table.reverse_interpolation("neuton")
        self.draw_graphs(rev_interpolation, "new_inter")
        return rev_interpolation(0)



















if __name__ == '__main__':
    table_1 = InterpolationTable()
    table_1.read_from_file("solve_eq_test_first_file")
    table_2 = InterpolationTable()
    table_2.read_from_file("solve_eq_test_sec_file")

    table_1.draw_graphs(None,None)
    table_2.draw_graphs(None,None )
    print(table_1.solve_equations(table_2))

    '''func = table.neuton_interpolation()
    func_h = table.hermit_interpolation()
    func_rev_n = table.neuton_interpolation()
    func_rev_h = table.hermit_interpolation()
    for i in range(5):
        table.n = i
        func = table.neuton_interpolation()
        func_h = table.hermit_interpolation()
        #table.print_table()
        print("Neuton_res: ",func(table.x),"Hermit_res: ",func_h(table.x),"try: ",i)
        func_rev_h = table.reverse_interpolation("hermit")
        func_rev_n = table.reverse_interpolation("neuton")
        print("Neuton_rev_res: ", func_rev_n(0), "Hermit_rev_res: ", func_rev_h(0), "try: ", i)
    table.draw_graphs(func,"neuton")
    table.draw_graphs(func_h,"hermit")
    table.draw_graphs(func_rev_h, "neuton")
    table.draw_graphs(func_rev_n, "hermit")'''
