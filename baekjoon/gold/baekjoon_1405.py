# Baekjoon 1405

from sys import stdin
from collections import deque

R = list(map(int, stdin.readline().split()))
N = R[0]
pb_E = R[1] / sum(R[1:])
pb_W = R[2] / sum(R[1:])
pb_S = R[3] / sum(R[1:])
pb_N = R[4] / sum(R[1:])

robot_map = [[False for _ in range(29)] for _ in range(29)]
robot_map[14][14] = True

probability_not_simple = 0

def dfs(row, col, probability, depth):
    global probability_not_simple
    if depth == 0:
        return
    # East
    if robot_map[row][col+1]:
        probability_not_simple += probability * pb_E
    else:
        robot_map[row][col+1] = True
        dfs(row, col+1, probability * pb_E, depth -1)
        robot_map[row][col+1] = False

    # West
    if robot_map[row][col-1]:
        probability_not_simple += probability * pb_W
    else:
        robot_map[row][col-1] = True
        dfs(row, col-1, probability * pb_W, depth -1)
        robot_map[row][col-1] = False

    # South
    if robot_map[row+1][col]:
        probability_not_simple += probability * pb_S
    else:
        robot_map[row+1][col] = True
        dfs(row+1, col, probability * pb_S, depth -1)
        robot_map[row+1][col] = False

    # North
    if robot_map[row-1][col]:
        probability_not_simple += probability * pb_N
    else:
        robot_map[row-1][col] = True
        dfs(row-1, col, probability * pb_N, depth -1)
        robot_map[row-1][col] = False

dfs(14, 14, 1, N)
# # 1 - 왔던 곳을 지나치는 경우의 확률

print(1 - probability_not_simple)