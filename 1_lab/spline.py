import matplotlib.pyplot as plt
def spline_func(coefs:list,index:int,x:float,x_table:list): #coefs a,b,c,d
    delta = x - x_table[index]
    res = 0
    for i in range(4):
        res += coefs[i][index] * (delta ** i);
    return res


def get_EPS(D,B,A,eps_bef):
    return D/(B - (A * eps_bef))

def get_Tetta(A,F,B,tetta_bef,eps_bef):
    return (A*tetta_bef + F) / (B - A * eps_bef)

def calc_epsilon_arr(D:list, B:list, A:list):
    eps_arr = [B[0]/A[0]]
    for i in range(len(A)):
        new_eps = D[i] / (B[i] - eps_arr[i])
        eps_arr.append(new_eps)
    return eps_arr

def calc_tetta_arr(F:list, A:list, B:list, eps:list):
    tetta_arr = [-F[0]/A[0]]
    for i in range(len(A)):
        new_tetta = (F[i] + A[i]*tetta_arr[i]) / (B[i] - A[i] * eps[i])
        tetta_arr.append(new_tetta)
    return tetta_arr

def h(x_bef:float,x_af:float):
    return x_af - x_bef

def b(y_1,y_2,c_1,c_2,h):
    return (y_2 - y_1) / h - 1/3 *(h * (c_2 + 2 * c_1))
def h_array(x:list):
    h_arr = []
    for i in range(1,len(x)):
        h_arr.append(h(x[i - 1], x[i]))
    return h_arr


def B_array(h_array:list):
    b_arr = []
    for i in range(1,len(h_array)):
        b_arr.append( 2* (h_array[i -1] + h_array[i]))
    return b_arr
def get_F(y_i,y_i_1,y_i_2,h_i,h_i_1): #y[i] y[i-1] y[i - 2] h[i] h[i-1]
    return 3 * (((y_i - y_i_1) / (h_i)) - ((y_i_1 - y_i_2) / (h_i_1)))
def F_array(y:list, h:list):
    d_arr = []
    for i in range(2,len(h)):
        d_arr.append(3 * ( ((y[i] - y[i - 1]) / (h[i])) - ((y[i - 1] - y[i - 2]) / (h[i - 1])) ))
    return d_arr

def calc_c_array(eps:list, tetta:list):
     c = [0 for i in range(len(eps) + 1)]
     for i in range(len(eps),-1,-1):
         c[i] = c[i + 1] * eps[i + 1] + tetta[i + 1]
     return c
class SplineTable:
    def __init__(self,table):
        self.xs = [table[i][0] for i in range(len(table))]
        self.ys = [table[i][1] for i in range(len(table))]

    def get_a(self):
        return self.ys[:-1:]
    def get_c(self,start_rule,end_rule):
        dots_count = len(self.xs)
        C = [0] * (dots_count - 1)
        eps = [0,0]
        tettas = [0,0]
        for i in range(2,dots_count):
            h_bef = h(self.xs[i - 2],self.xs[i - 1])
            h_cur = h(self.xs[i - 1], self.xs[i])
            D = h_cur
            A = h_bef
            B = (h_bef+h_cur)*2
            F = get_F(self.ys[i],self.ys[i - 1], self.ys[i - 2],h_cur,h_bef)
            eps_cur = get_EPS(D,B,A,eps[i - 1])
            tetta_cur = get_Tetta(A,F,B,tettas[i-1],eps[i - 1])
            eps.append(eps_cur)
            tettas.append(tetta_cur)
        C[-1] = tettas[-1]

        for i in range(dots_count - 2, 0, -1):
            C[i - 1] = eps[i] * C[i] + tettas[i]
        return C
    def get_d(self,c:list):
        d = []
        for i in range(1,len(c)):
            h_cur = self.xs[i] -  self.xs[i-1]
            d.append((c[i] - c[i-1]) / h_cur)
        h = self.xs[-1] - self.xs[-2]
        d.append((0 - c[-1])/h) #TOdo:NTF
        return d

    def get_b(self,c:list):
        b_arr = []
        for i in range(1,len(self.xs) - 1):
            h_cur = self.xs[i] - self.xs[i - 1]
            cur_b = (b(self.ys[i - 1], self.xs[i], c[i - 1], c[i], h_cur))
            b_arr.append(cur_b)
        cur_h = self.xs[-1] - self.xs[-2]
        b_cur = b(self.ys[-2],self.ys[-1],c[0],c[-1],h_cur)
        b_arr.append(b_cur)
        return b_arr

    def get_all_coefficients(self):
        a_arr = self.get_a()
        c_arr = self.get_c(0,0) #Todo: fix starting and ending rules
        b_arr = self.get_b(c_arr)
        d_arr = self.get_d(c_arr)
        coefficients = [a_arr,b_arr,c_arr,d_arr]
        return coefficients

    def find_index(self,x:float):
        index = 0
        while (index < len(self.xs) - 1 and x > self.xs[index]):
            index += 1
        return index - 1

    def spline_func(coefs: list, index: int, x: float):  # coefs a,b,c,d
        delta = x - self.xs[index]
        res = 0
        for i in range(4):
            res += coefs[i][index] * (delta ** i);
        return res
    def spline_interpolation(self,x):
        coefs = self.get_all_coefficients()
        index = self.find_index(x)
        return spline_func(coefs,index,x,self.xs)
    def draw_graphs(self,function_name:str):
        x = self.xs
        y = self.ys
        plt.plot(x, y, label="real_function", marker='*')
        y_interpolated = list(map(self.spline_interpolation, x))
        plt.plot(x, y_interpolated, label=function_name, marker='.')
        plt.legend()
        plt.show()







