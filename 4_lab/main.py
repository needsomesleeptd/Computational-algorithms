import numpy as np

from table import *
from diff_eq import *

table = Table()
table.generate_random_dots(5,equal=True,count=3)
table.print()
#table.read_weights()
#func = table.get_function_1d(2)
#table.print()
#table.dual_plot(1,table.get_function_1d)
#plt.show()
#table.generate_random_dots(5,count=3)
#table.print()
#approc_function = table.get_function_2d()
#table.drawGraficBy_AproxFunction_2D(approc_function)
xs = np.linspace(0,1)
func = get_func_2(xs,2)
print(get_func_2(xs,2,display=True))
x = float(input())
print(func(x))



