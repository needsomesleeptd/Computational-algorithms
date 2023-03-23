from MultiDimensional import *

if __name__ == '__main__':
    dataset = [
        [
            [0, 1, 4, 9, 16],
            [1, 2, 5, 10, 17],
            [4, 5, 8, 13, 20],
            [9, 10, 13, 18, 25],
            [16, 17, 20, 25, 32]
        ],
        [
            [1, 2, 5, 10, 17],
            [2, 3, 6, 11, 18],
            [5, 6, 9, 14, 21],
            [10, 11, 14, 19, 26],
            [17, 18, 21, 26, 33]
        ],
        [
            [4, 5, 8, 13, 20],
            [5, 6, 9, 14, 21],
            [8, 9, 12, 17, 24],
            [13, 14, 17, 22, 29],
            [20, 21, 24, 29, 36]
        ],
        [
            [9, 10, 13, 18, 25],
            [10, 11, 14, 19, 26],
            [13, 14, 17, 22, 29],
            [18, 19, 22, 27, 34],
            [25, 26, 29, 34, 41]
        ],
        [
            [16, 17, 20, 25, 32],
            [17, 18, 21, 26, 33],
            [20, 21, 24, 29, 36],
            [25, 26, 29, 34, 41],
            [32, 33, 36, 41, 48]
        ]

    ]

    xs = [i for i in range(5)]
    ys = [i for i in range(5)]
    zs = [i for i in range(5)]

    xp, yp, zp = map(int, input("Введите значения xp yp zp через пробел").split())
    nx, ny, nz = map(int, input("Введите значения nx ny nz через пробел").split())

    multidim_table = MultiDim()
    multidim_table.fit(dataset,xs,ys,zs)
    print("Neuton interpolation result {0}".format(multidim_table.MultidimensionalInterpolationNeuton(nx,ny,nz,xp,yp,zp)))
    print("Spline interpolation result {0}".format(multidim_table.MultidimensionalInterpolationNeuton(nx, ny, nz, xp, yp, zp)))
    print("Combined interpolation result {0}".format(multidim_table.MultidimensionalInterpolationNeuton(nx, ny, nz, xp, yp, zp)))
