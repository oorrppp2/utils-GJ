import sys
# 시작위치 : 사면중 적어도 인접한 두 면이 열려있는 위치.  위,왼쪽, or 위, 오른쪽 or 아래, 왼쪽 or 아래,오른쪽

Map = []
H, W = map(int, (sys.stdin.readline().split()))

for i in range(H):
        input_str = list(sys.stdin.readline())
        Map.append(input_str[:-1])

print(Map)
print(Map[0][3])

start_position = []

current_position = []
for i in range(H):
    for j in range(W):
        start_from_here = False
        if Map[i][j] == '#':
            if i == 0 or Map[i-1][j] == '.':
                if j == 0 or Map[i][j-1] == '.':
                    start_from_here = True
                if j == W-1 or Map[i][j+1] == '.':
                    start_from_here = True
            if i == H-1 or Map[i+1][j] == '.':
                if j == 0 or Map[i][j-1] == '.':
                    start_from_here = True
                if j == W-1 or Map[i][j+1] == '.':
                    start_from_here = True

        if start_from_here:
            start_row, start_col = i, j
            start_dir = ""
            if Map[i+1][j] == "#" and Map[i+2][j] == "#":
                start_dir = "v"
                Map[i+1][j] = "."
                Map[i+2][j] = "."
                while True:
                    Ma
            searching_queue = []
