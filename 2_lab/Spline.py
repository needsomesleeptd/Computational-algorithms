from InterpolationTable import *




def eps(D: float, B: float, A: float, prev_eps: float):
    return D / (B - A * prev_eps)


def etta(A: float, prev_etta: float, F: float, B: float, eps):
    return (A * prev_etta + F) / (B - A * eps)


def F(cur_y: float, prev_y: float, before_prev_y: float, cur_h, prev_h):  # yi yi-1 yi-2 hi hi-1
    return -3 * (((cur_y - prev_y) / cur_h) - ((prev_y - before_prev_y) / prev_h))


def C(cur_eps: float, prev_c: float, cur_etta: float):
    return cur_eps * prev_c + cur_etta


def d(cur_C: float, next_C: float, cur_h: float):
    return (next_C - cur_C) / (3 * cur_h)


def b(y: float, prev_y: float, h: float, C: float, next_C: float):
    return ((y - prev_y) / h) - 1 / 3 * h * (next_C + 2 * C)


class SplineTable(InterpolationTable):
    def __init__(self):
        super().__init__()
        self.Data = []
        self.n = 0
        self.x = 0
        self.xs = []
        self.ys = []

    def fit(self, dataset: list, n: int, x: float):
        self.Data = dataset
        self.n = n
        self.x = x
        self.xs, self.ys = self.get_xy()

    def get_Cs(self, start_c_init, end_c_init):
        dots_count = len(self.xs)
        epss = [0]
        ettas = [start_c_init / 2]  # zeros to fix len
        Cs = [0 for i in range(dots_count)]
        Cs[0] = start_c_init / 2
        Cs[-1] = end_c_init / 2
        hs = [self.xs[i] - self.xs[i - 1] for i in range(1,dots_count)]

        for i in range(1, dots_count - 1):
            h1 = hs[i]
            h2  = hs[i - 1]
            cur_B = -2 * (h2 + h1)
            cur_A = h2
            cur_D = h1
            cur_F = F(self.ys[i + 1], self.ys[i], self.ys[i - 1], h1, h2)
            cur_eps = eps(cur_D, cur_B, cur_A, epss[i - 1])
            cur_etta = etta(cur_A, ettas[i - 1], cur_F, cur_B, epss[i - 1])
            epss.append(cur_eps)
            ettas.append(cur_etta)

        for i in range(dots_count - 2, 0, -1):
            Cs[i] = epss[i] * Cs[i + 1] + ettas[i]
        return Cs

    def get_as(self):
        return self.ys[:-1]

    def get_ds(self, Cs: list):
        ds = []
        for i in range(1, len(Cs)):
            h1 = self.xs[i] - self.xs[i - 1]
            ds.append(d(Cs[i - 1], Cs[i], h1))

        h1 = self.xs[-1] - self.xs[-2]
        ds.append(d(0, Cs[-1], h1))
        return ds

    def get_bs(self, Cs: list):
        bs = []
        for i in range(1, len(Cs) - 1):
            h1 = self.xs[i] - self.xs[i - 1]
            bs.append(b(self.ys[i], self.ys[i - 1], h1, Cs[i - 1], Cs[i]))

        h1 = self.xs[-1] - self.xs[-2]
        bs.append(b(self.ys[-2], self.ys[-1], h1, Cs[i - 1], Cs[i]))
        return bs

    def get_coefs(self,start_coef,end_coef):
        a = self.get_as()
        c = self.get_Cs(0,0)
        b = self.get_bs(c)
        d = self.get_ds(c)
        return a,b,c,d

    def find_slice(self,x:float):
        index = 0
        while (self.xs[index] < x and index < len(self.xs)):
            index += 1
        if (index >= len(self.xs) - 1):
            index = len(self.xs) - 2
        return index






    def spline_interpolation(self,start_coef,end_coef):
        a,b,c,d = self.get_coefs(start_coef,end_coef)
        coefs = [a,b,c,d]
        print(a)
        print(b)
        print(c)
        print(d)

        def spline_func(x:float):
            func_sum = 0
            index = self.find_slice(x)
            for i in range(4):  # 4 == coefs count
                func_sum += coefs[i][index - 1] * ((x - self.xs[index - 1]) ** i)
            return func_sum
        return spline_func






if __name__ == '__main__':
    table_1 = SplineTable()
    table_1.read_from_file("test_first_file.txt")
    spline_func = table_1.spline_interpolation(0,0)
    target_x = 2.54
    ans = spline_func(target_x)
    print("ans:",ans)
    table_1.draw_graphs(spline_func,"spline_func")
