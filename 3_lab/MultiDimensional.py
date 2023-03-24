from Spline import SplineTable


class MultiDim(SplineTable):
    def __init__(self):
        self.DataSet = []
        self.Data = []
        self.xs = []
        self.ys = []
        self.zs = []

    def fit(self, dataset, xs, ys, zs):
        self.DataSet = dataset
        self.Data = dataset[0]
        self.xs = xs
        self.ys = ys
        self.zs = zs

    def neuton_interpolation(self, x=None, Data=None, n=None):
        # print("Neuton:\n")
        if (x == None):
            x = self.x
        if (Data == None):
            Data = self.Data

        if (n == None):
            n = self.n

        chosen_dots = self.select_rows_neuton(x, Data, n)
        if (chosen_dots == None):
            print("Данных недостаточно для образования полинома данной степени")
            return None
        # print(chosen_dots)
        # print(chosen_dots)
        part_sums = self.get_part_sums_neuton(chosen_dots)

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

        return interpolation_on_point(x)

    def init_table(self, values):
        self.xs = values
        self.ys = values
        self.zs = values

    def MultidimensionalInterpolationNeuton(self, nx, ny, nz, xp, yp, zp):


        z_vals = []
        for z_index in range(len(self.zs)):  # z
            y_vals = []
            for y_index in range(len(self.ys)):  # y
                x_vals = []
                for x_index in range(len(self.xs)):  # x
                    x_vals.append([self.xs[x_index], self.DataSet[z_index][y_index][x_index]])

                func_u_x_val = self.neuton_interpolation(xp, Data=x_vals, n=nx)
                y_vals.append([self.ys[y_index], func_u_x_val])

            func_u_y_val = self.neuton_interpolation(yp, Data=y_vals, n=ny)
            z_vals.append([self.zs[z_index], func_u_y_val])

        func_u_z_val = self.neuton_interpolation(zp, Data=z_vals, n=nz)
        return func_u_z_val

    def MultidimensionalInterpolationSpline(self, nx, ny, nz, xp, yp, zp):

       # self.init_table([0, 1, 2, 3, 4])
        z_vals = []
        for z_index in range(len(self.zs)):  # z
            y_vals = []
            for y_index in range(len(self.ys)):  # y
                x_vals = []
                for x_index in range(len(self.xs)):  # x
                    x_vals.append([self.xs[x_index], self.DataSet[z_index][y_index][x_index]])
                func_u_x = self.spline_interpolation(x_vals)
                val_x =  func_u_x(xp)
                self.init_table([0, 1, 2, 3, 4])
                y_vals.append([self.ys[y_index], val_x])


            func_u_y = self.spline_interpolation(y_vals)
            val_y = func_u_y(yp)
            self.init_table([0, 1, 2, 3, 4])
            z_vals.append([self.zs[z_index], val_y])


        func_u_z = self.spline_interpolation(z_vals)
        return func_u_z(zp)

    def MultidimensionalInterpolationCombine(self, nx, ny, nz, xp, yp, zp):
        #self.init_table([0, 1, 2, 3, 4])
        z_vals = []
        for z_index in range(len(self.DataSet)):  # z
            y_vals = []
            for y_index in range(len(self.DataSet[0])):  # y
                x_vals = []
                for x_index in range(len(self.DataSet[0][0])):  # x
                    x_vals.append([self.xs[x_index], self.DataSet[z_index][y_index][x_index]])
                func_u_x_val = self.neuton_interpolation(xp, Data=x_vals, n=nx)
                y_vals.append([self.ys[y_index], func_u_x_val])

            func_u_y = self.spline_interpolation(y_vals)
            val_y = func_u_y(yp)
            self.init_table([0, 1, 2, 3, 4])
            z_vals.append([self.zs[z_index],val_y])


        func_u_z_val = self.neuton_interpolation(zp, Data=z_vals, n=nz)
        return func_u_z_val
