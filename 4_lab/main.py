from table import *


table = Table()
table.generate_random_dots(5,equal = True)
table.print()
table.read_weights()
func = table.get_function(2)
table.print()
table.dual_plot(1)
plt.show()