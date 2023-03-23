from Spline import SplineTable
class MultiDim(SplineTable):
    def __init__(self):
        self.DataSet = []
        self.Data = [] 
        self.xs = []
        self.ys = []
        self.zs = []

    def fit(self,dataset,xs,ys,zs):
        self.DataSet = dataset
        self.Data = dataset[0]
        self.xs = xs
        self.ys = ys
        self.zs = zs




    def MultidimensionalInterpolationNeuton(self, nx, ny, nz, xp, yp, zp):


        z_vals = []
        for z_index in range(len(self.DataSet)):  # z
            y_vals = []
            for y_index in range(len(self.DataSet[0])):  # y
                x_vals = []
                for x_index in range(len(self.DataSet[0][0])):  # x
                    x_vals.append([self.xs[x_index], self.DataSet[z_index][y_index][x_index]])
                func_u_x = self.neuton_interpolation(xp, Data=x_vals, n=nx)
                y_vals.append([self.ys[y_index], func_u_x(xp)])

            func_u_y = self.neuton_interpolation(yp, Data=y_vals, n=ny)
            z_vals.append([self.zs[y_index], func_u_y(yp)])



        return self.neuton_interpolation(zp, Data=z_vals, n=nz)


def MultidimensionalInterpolationSpline(self, nx, ny, nz, xp, yp, zp):


    z_vals = []
    for z_index in range(len(self.DataSet)):  # z
        y_vals = []
        for y_index in range(len(self.DataSet[0])):  # y
            x_vals = []
            for x_index in range(len(self.DataSet[0][0])):  # x
                x_vals.append([self.x[x_index], self.DataSet[z_index][y_index][x_index]])
            func_u_x = self.spline_interpolation(xp,x_vals)
            y_vals.append([self.y[y_index], func_u_x(xp)])

        func_u_y = self.spline_interpolation(yp,y_vals)
        z_vals.append([self.z[y_index],func_u_y(yp)])

    return self.spline_interpolation(zp, Data=z_vals)

def MultidimensionalInterpolationCombine(self, nx, ny, nz, xp, yp, zp):


    z_vals = []
    for z_index in range(len(self.DataSet)):  # z
        y_vals = []
        for y_index in range(len(self.DataSet[0])):  # y
            x_vals = []
            for x_index in range(len(self.DataSet[0][0])):  # x
                x_vals.append([self.x[x_index], self.DataSet[z_index][y_index][x_index]])
            func_u_x = self.neuton_interpolation(xp, Data=x_vals, n=nx)
            y_vals.append([self.ys[y_index], func_u_x(xp)])

        func_u_y = self.spline_interpolation(yp,y_vals)
        z_vals.append([self.z[y_index],func_u_y(yp)])

    return self.neuton_interpolation(zp, Data=z_vals, n=nz)

