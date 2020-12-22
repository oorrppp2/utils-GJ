import numpy as np

number = 5
# arr = np.array([
#     [0, 5, np.inf, 8],
#     [7, 0, 9, np.inf],
#     [2, np.inf, 0, 4],
#     [np.inf, np.inf, 3, 0]
# ])

arr = np.array([
    [0, 3, 8, np.inf, -4],
    [np.inf, 0, np.inf, 1, 7],
    [np.inf, 4, 0, np.inf, np.inf],
    [2, np.inf, -5, 0, np.inf],
    [np.inf, np.inf, np.inf, 6, 0]
])

print(arr)

def floydWarshall():
    f_arr = arr
    for k in range(number):
        print("---------------------")
        print(f_arr)
        for i in range(number):
            for j in range(number):
                if f_arr[i][k] + f_arr[k][j] < f_arr[i][j]:
                    f_arr[i][j] = f_arr[i][k] + f_arr[k][j]
    print(f_arr)

if __name__ == '__main__':
    floydWarshall()
