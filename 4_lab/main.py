from table import *


table = Table()
table.generate_random_dots(5,equal = True)
table.print()
table.read_weights()
func = table.get_function(3)
table.dual_plot(3)
plt.show()