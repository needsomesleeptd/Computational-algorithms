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
    #rev_interpolation = new_interpolation_table.reve rse_interpolation("neuton")
    first_table.rev_function()
    return rev_interpolation_table_equation



















if __name__ == '__main__':
    '''table_1 = InterpolationTable()
    table_1.read_from_file("task_solve_equations_first_file")
    table_2 = InterpolationTable()
    table_2.read_from_file("task_solve_equations_sec_file")

    table_1.draw_graphs(None,None)
    table_2.draw_graphs(None,None )
    print(table_1.solve_equations(table_2))'''

    table_1 = InterpolationTable()
   #table_2 = InterpolationTable()
    table_1.read_from_file("task_test")
    rev_table_1 = table_1.reverse_interpolation_table()
    for i in range(2,8):
        table_1.n = i
        rev_table_1.n = i
        func_hermit = table_1.hermit_interpolation(0)
        func_neuton = table_1.neuton_interpolation(0)
        rev_func_hermit = rev_table_1.hermit_interpolation()
        rev_func_neuton = rev_table_1.neuton_interpolation()
        print(("n == {}\n_______________________________________").format(i))
        print("inter_front hermit:",func_hermit(0.675))
        print("inter_front_neuton:",func_neuton(0.675))
        print("inter_reversed hemit:",rev_func_hermit(0))
        print("inter_reversed hemit:", rev_func_neuton(0))
        print("________________________________________________")

    table_2 = InterpolationTable()
    table_1.read_from_file("task_solve_equations_first_file")
    table_2.read_from_file("task_solve_equations_sec_file")
    table_1.rev_function()
    xs_1  = [table_1.Data[i][0] for i in range(len(table_1.Data))]
    ys_1 = [table_1.Data[i][1] for i in range(len(table_1.Data))]
    xs_2 =  [table_2.Data[i][0] for i in range(len(table_1.Data))]
    ys_2 =  [table_2.Data[i][1] for i in range(len(table_1.Data))]
    plt.plot(xs_1,ys_1)
    plt.plot(xs_2,ys_2)
    #plt.show()
    table_1.rev_function()
    for i in range(1,8):
        table_2.n = i
        table_1.n = i
        equation_solve_table = solve_equations_table(table_1, table_2)
        equation_solve_table.n = i
        equation_solve_func = equation_solve_table.neuton_interpolation(equation_solve_table.x)
        eq_root_val = equation_solve_func(0)
        print(("roots of equations: {}, n == {}").format(eq_root_val,i))

    plt.show()





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
