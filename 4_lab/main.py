import matplotlib.pyplot as plt
import numpy as np

from table import *
from diff_eq import *


def array_map(x, f):
    return np.array(list(map(f, x)))


table = Table()
table.generate_random_dots(5, equal=True, count=3)
table.print()
# table.read_weights()
# func = table.get_function_1d(2)
# table.print()
# table.dual_plot(1,table.get_function_1d)
# plt.show()
# table.generate_random_dots(5,count=3)
# table.print()
# approc_function = table.get_function_2d()
# table.drawGraficBy_AproxFunction_2D(approc_function)
xs = np.linspace(0, 1)
xs_test = np.linspace(-10, 10)
print('n==2 : {0}'.format(get_func(xs, 2, display=True)))
print('n==3 : {0}'.format(get_func(xs, 3, display=True)))
func_2 = get_func(xs, 2)
func_3 = get_func(xs, 3)
vals_2 = array_map(xs_test, func_2)
vals_3 = array_map(xs_test, func_3)
plt.plot(xs, vals_3, label='n == 3')
plt.plot(xs, vals_2, label='n == 2')
plt.legend()
plt.show()
# print(get_func_2(xs,3,display=True))
# x = float(input())
# print(func(x))
