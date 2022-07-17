# Baekjoon 12100

from sys import stdin
from collections import deque
import copy

N = int(stdin.readline())
board = []

for i in range(N):
    line = list(map(int, stdin.readline().split()))
    board.append(line)
    


board_state = [board]
for t in range(5):
    for tt in range(len(board_state)):
        current_state = board_state.pop(0)
        # left
        next_state = copy.deepcopy(current_state)
        for i in range(N):
            for j in range(N):
                if next_state[i][j] != 0:   # 0이 아닌 숫자가 나오면 그 다음에 같은 숫자가 있는지 검사, 있으면 합친다
                    target = next_state[i][j]
                    for jj in range(j+1, N):
                        if next_state[i][jj] == target:
                            next_state[i][j] *= 2
                            next_state[i][jj] = 0
                            break
                        elif next_state[i][jj] != 0:    # target도 아닌데 0도 아니면 합치지 않음
                            break
            # 한 줄에서 합치는 과정이 끝나면 사이드로 다 밀어넣음
            new_line = []
            for j in range(N):
                if next_state[i][j] != 0:
                    new_line.append(next_state[i][j])
            for _ in range(N - len(new_line)):  # N만큼 0 채워넣기
                new_line.append(0)
            next_state[i] = new_line
        board_state.append(next_state)


        # right
        next_state = copy.deepcopy(current_state)
        for i in range(N):
            for j in range(N-1, -1, -1):
                if next_state[i][j] != 0:   # 0이 아닌 숫자가 나오면 그 다음에 같은 숫자가 있는지 검사, 있으면 합친다
                    target = next_state[i][j]
                    for jj in range(j-1, -1, -1):
                        if next_state[i][jj] == target:
                            next_state[i][j] *= 2
                            next_state[i][jj] = 0
                            break
                        elif next_state[i][jj] != 0:    # target도 아닌데 0도 아니면 합치지 않음
                            break
            # 한 줄에서 합치는 과정이 끝나면 사이드로 다 밀어넣음
            new_line = []
            for j in range(N-1, -1, -1):
                if next_state[i][j] != 0:
                    new_line.append(next_state[i][j])
            for _ in range(N - len(new_line)):  # N만큼 0 채워넣기
                new_line.append(0)
            next_state[i] = list(reversed(new_line))
        board_state.append(next_state)


        # # up
        next_state = copy.deepcopy(current_state)
        for j in range(N):
            for i in range(N):
                if next_state[i][j] != 0:   # 0이 아닌 숫자가 나오면 그 다음에 같은 숫자가 있는지 검사, 있으면 합친다
                    target = next_state[i][j]
                    for ii in range(i+1, N):
                        if next_state[ii][j] == target:
                            next_state[i][j] *= 2
                            next_state[ii][j] = 0
                            break
                        elif next_state[ii][j] != 0:    # target도 아닌데 0도 아니면 합치지 않음
                            break
            # 한 줄에서 합치는 과정이 끝나면 사이드로 다 밀어넣음
            new_line = []
            for i in range(N):
                if next_state[i][j] != 0:
                    new_line.append(next_state[i][j])
            for _ in range(N - len(new_line)):  # N만큼 0 채워넣기
                new_line.append(0)
            for i in range(N):
                next_state[i][j] = new_line[i]
            next_state[:][j] = new_line
        board_state.append(next_state)


        # # down
        next_state = copy.deepcopy(current_state)
        for j in range(N):
            for i in range(N-1, -1, -1):
                if next_state[i][j] != 0:   # 0이 아닌 숫자가 나오면 그 다음에 같은 숫자가 있는지 검사, 있으면 합친다
                    target = next_state[i][j]
                    for ii in range(i-1, -1, -1):
                        if next_state[ii][j] == target:
                            next_state[i][j] *= 2
                            next_state[ii][j] = 0
                            break
                        elif next_state[ii][j] != 0:    # target도 아닌데 0도 아니면 합치지 않음
                            break
            # 한 줄에서 합치는 과정이 끝나면 사이드로 다 밀어넣음

            new_line = []
            for i in range(N-1, -1, -1):
                if next_state[i][j] != 0:
                    new_line.append(next_state[i][j])
            for _ in range(N - len(new_line)):  # N만큼 0 채워넣기
                new_line.append(0)
            new_line = list(reversed(new_line))
            for i in range(N):
                next_state[i][j] = new_line[i]
            next_state[:][j] = new_line
        board_state.append(next_state)

ans = 0
for board in board_state:
    for i in range(N):
        ans = max(board[i]) if max(board[i]) > ans else ans

print(ans)