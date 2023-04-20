import matplotlib.pyplot as plt
import numpy as np

from table import *
from diff_eq import *


def menu():
    str = '''
       1.Вывести таблицу полностью
       2.Ввести веса в таблице
       3.Получить график функции
       4.Получить плоскость функцию
       5.Сгенерировать таблицу со случайными параметрами
       6.Ввести таблицу
       7.Вывести разницу между решениями дифф уравнения
       0.Выйти
    '''
    print(str)

table = Table()
ans = 1
while (ans != 0):
    menu()
    ans = int(input("Введите пункт меню:"))
    if (ans == 1):
        table.print()
    elif (ans == 2):
        table.read_weights()
    elif (ans == 3):
        n = int(input('Введите степень полинома:'))
        table.dual_plot(n, table.get_function_1d)
        plt.show()
    elif (ans == 4):
        table.drawGraficBy_AproxFunction_2D(table.get_function_2d())
    elif (ans == 5):
        n = int(input('Введите количество точек:'))
        table.generate_random_dots(n, count=3)
    elif (ans == 6):
        table.read_dots_params()
    elif (ans == 7):
        plot_n_diff()



'''table.generate_random_dots(5, equal=True, count=3)
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
func_2 = get_func(xs_test, 2)
func_3 = get_func(xs_test, 3)
vals_2 = array_map(xs_test, func_2)
vals_3 = array_map(xs_test, func_3)
plt.plot(xs_test, vals_3, label='n == 3')
plt.plot(xs_test, vals_2, label='n == 2')
plt.legend()
plt.show()
# print(get_func_2(xs,3,display=True))
# x = float(input())
# print(func(x))
'''