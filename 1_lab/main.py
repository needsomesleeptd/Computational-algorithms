import matplotlib.pyplot as plt
import copy
import numpy as np
def is_float(num : str):
    num = num.replace("-","",1)
    num = num.replace(".", "", 1)
    return num.isnumeric()


def is_row_valid_float(row:list):
    for i  in range(len(row)):
        if (not(is_float(row[i]))):
            return False
    return True





def print_table(table:list):
    spaces_cnt = 3
    #print(table)
    for row in table:
        template = "{:^8} " * len(row)
        print(template.format(*row))


#def part_sum_neuton(dataset:list,):
#    for i in range()

def get_part_sums_neuton(dataset:list):
    n = len(dataset)
    part_sums = [[0 for i in range(n + 1)] for i in range(n + 2)] # n*n dim
    for i in range(n):
        part_sums[i][0] = dataset[i][0]
        part_sums[i][1] = dataset[i][1]
    row_iter = n - 1
    col_count = 1
    i = 0
    for j in range(2,n + 1):
        while (i < row_iter):
            part_sums[i][j] = (part_sums[i][j - 1] - part_sums[i + 1][j - 1]) / (part_sums[i][0] - part_sums[i + col_count][0])
            i+=1
        row_iter -=1
        i = 0
        col_count += 1
    return part_sums[0]
        
        

 






def get_dots_interolation(dataset:list,n:int,x:int):
    dataset.sort(key = lambda x: x[0])
    selected_rows = []

    min_delta_row_index = 0
    min_delta = abs(dataset[0][0] - x)
    for i in range(len(dataset)):
        if (abs(dataset[i][0] - x) < min_delta):
            min_delta = abs(dataset[i][0] - x) 
            min_delta_row_index = i
    l_bound = min_delta_row_index
    r_bound = min_delta_row_index + 1
    while (len(selected_rows) < n + 1):
        if (l_bound >= 0):
            selected_rows.append(dataset[l_bound])
            l_bound-=1
        if (r_bound < len(dataset)):
            selected_rows.append(dataset[r_bound])
            r_bound+=1
    selected_rows.sort(key = lambda x:x[0])
    return selected_rows


def Neuton_interpolation(dataset:list,n:int,x:int):
    chosen_dots = get_dots_interolation(database,n,x)
    part_sums = get_part_sums_neuton(chosen_dots)
    #(part_sums[0], part_sums[1]) = (part_sums[1], part_sums[0])
    def interpolation_on_point(x:int):
        inter_sum = part_sums[1]
        for i in range(2,len(part_sums)):
            inter_mul = 1
            for j in range(i - 1):
                inter_mul *=  x - chosen_dots[j][0]
            inter_sum += inter_mul  * part_sums[i]
        return inter_sum
    return interpolation_on_point

def draw_graphs(interpolation_func,x:list,y:list):
    y_interpolated = list(map(interpolation_func,x))
    plt.plot(x,y,label = "real_function",marker = '*')
    plt.plot(x,y_interpolated,label = "newton_interpolation",marker = '.')
    plt.legend()
    plt.show()
    for i in range(len(x)):
        print(x[i],y[i],y_interpolated[i])
    
  


            











#def Neuton_interpolation(dataset:list, x:int,n: int):





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''col_count = 4

    database = []
    row = ["0" for i in range(col_count)]

    print("Введите таблицу значений(Для окончания ввода введите все -)")
    row = input().split()
    while(is_row_valid_float(row)):
        row = list(map(lambda x:float(x),row))
        database.append(row)
        row = input().split()
    print("Введите степень аппроксимирующих полиномов")
    n = int(input()) # Степень
    print("Введите значения текущего x")
    cur_x = float(input())
    print_table(database)
    if (len(database) < n +1):
        print("Нехватает данных для верной интерполяции")
        exit()'''
    database = [[i,i ** 2 // 2] for i in range(-1000,1000)]
    n = 5
    cur_x = 100
    #chosen_dots = get_dots_interolation(database,n,cur_x)
    #part_sums = get_part_sums_neuton(chosen_dots)
    #print_table(part_sums)
    func = Neuton_interpolation(database,n,cur_x)
    x = [database[i][0] for i in range(len(database))]
    y = [database[i][1] for i in range(len(database))]
    draw_graphs(func,x,y)
    #print(func(0.6))



    







