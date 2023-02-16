import matplotlib.pyplot as plt


class InterpolationTable:
    def __init__(self):
        self.Data = []
        self.n = 0
        self.x = 0

    def fit(self, dataset: list, n: int, x: int):
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

    def select_rows(self):
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
            if (l_bound >= 0):
                selected_rows.append(self.Data[l_bound])
                l_bound -= 1
            if (r_bound < len(self.Data)):
                selected_rows.append(self.Data[r_bound])
                r_bound += 1
        selected_rows.sort(key=lambda x: x[0])
        return selected_rows

    def neuton_interpolation(self):
        chosen_dots = self.select_rows()
        part_sums = self.get_part_sums_neuton(chosen_dots)

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

    def draw_graphs(self, interpolation_func):
        x = [self.Data[i][0] for i in range(len(self.Data))]
        y = [self.Data[i][1] for i in range(len(self.Data))]
        y_interpolated = list(map(interpolation_func, x))
        plt.plot(x, y, label="real_function", marker='*')
        plt.plot(x, y_interpolated, label="newton_interpolation", marker='.')
        plt.legend()
        plt.show()
        for i in range(len(x)):
            print(x[i], y[i], y_interpolated[i])

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


    def get_part_sums_neuton(self,chosen_dots:list):
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






if __name__ == '__main__':
    table = InterpolationTable()
    table.scan_data()
    func = table.neuton_interpolation()
    table.print_table()
    table.draw_graphs(func)
