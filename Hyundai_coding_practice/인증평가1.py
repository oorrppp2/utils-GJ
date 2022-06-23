import sys

signal_dir = {1:["r","u","d"], 2:["l","u","r"], 3:["u","l","d"], 4:["l","d","r"],
              5:["r","u"], 6:["l","u"], 7:["l","d"], 8:["d","r"],
              9:["r","d"], 10:["r","u"], 11:["l","u"], 12:["d","l"]}
entrance_dir = ["", "right", "up", "left", "down",
                "right", "up", "left", "down",
                "right", "up", "left", "down"]

S = []
T, N = map(int, (sys.stdin.readline().split()))
for i in range(N ** 2):
    S.append(list(map(int, (sys.stdin.readline().split()))))

num_visit = [[0 for col in range(N)] for row in range(N)]

current_state = [[0,0,0,"up"]]   # [row, col, T, entrance_dir(진입방향    [up, down, left, right])]
num_visit[0][0] = 1

while len(current_state) != 0:
    state = current_state.pop()
    current_position = (state[0]) * N + (state[1])
    current_time = state[2]
    current_signal = S[current_position][current_time%4]
    current_entrance_dir = entrance_dir[current_signal]
    current_signal_dir = signal_dir[current_signal]
    driving_dir = state[3]

    if current_time > T-1:
        continue

    if driving_dir == current_entrance_dir:
        current_row, current_col = state[0], state[1]
        for next_dir in current_signal_dir:
            next_row, next_col = current_row, current_col
            next_entrance_dir = ""
            if next_dir == "r":
                next_col += 1
                next_entrance_dir = "right"
            if next_dir == "l":
                next_col -= 1
                next_entrance_dir = "left"
            if next_dir == "d":
                next_row += 1
                next_entrance_dir = "down"
            if next_dir == "u":
                next_row -= 1
                next_entrance_dir = "up"
            if next_row < 0 or next_row >= N or next_col < 0 or next_col >= N:
                continue
            num_visit[next_row][next_col] += 1
            current_state.append([next_row, next_col, current_time + 1, next_entrance_dir])

answer = 0
for i in range(N):
        for j in range(N):
                if num_visit[i][j] > 0:
                        answer += 1

print(answer)