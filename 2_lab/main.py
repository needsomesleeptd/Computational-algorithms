from Spline import *
from beautifultable import BeautifulTable

if __name__ == '__main__':
    x = float(input())


    table_1 = SplineTable()
    table_1.read_from_file("test_first_file.txt")
    spline_func = table_1.spline_interpolation(0, 0)
    table_1.draw_graphs(spline_func, "spline_func")

    left_bound = table_1.inter_sec_derivarive(table_1.xs[0])
    right_bound = table_1.inter_sec_derivarive(table_1.xs[-1])
    spline_func_2 = table_1.spline_interpolation(left_bound, right_bound)
    table_1.draw_graphs(spline_func_2, "both_bounds_spline_func")

    spline_func_3 = table_1.spline_interpolation(left_bound, 0)
    table_1.draw_graphs(spline_func_3, "left_bound_spline_func")

    spline_func_4 = table_1.spline_interpolation(0, right_bound)
    table_1.draw_graphs(spline_func_4, "right_bound_spline_func")

    table_1.draw_graphs_neuton(table_1.neuton_interpolation, "neuton_interpolation")

    plt.plot(table_1.xs, table_1.ys, label="real function")
    table_1.draw_graphs(spline_func, "spline_func",show=False)
    table_1.draw_graphs(spline_func_2, "both_bounds_spline_func",show=False)
    table_1.draw_graphs(spline_func_3, "left_bound_spline_func",show=False)
    table_1.draw_graphs(spline_func_4, "right_bound_spline_func",show=False)
    table_1.draw_graphs_neuton(table_1.neuton_interpolation, "neuton_interpolation",show=False)
    plt.show()

    table_1.print_coefs(left_bound,right_bound,"both_bounds")
    table_1.print_coefs(0, 0, "no_bounds")
    table_1.print_coefs(left_bound, 0, "left_bound")
    table_1.print_coefs(0, right_bound, "right_bound")



