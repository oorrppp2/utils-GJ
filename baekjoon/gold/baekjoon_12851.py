# Baekjoon 12851

from sys import stdin

N, K = list(map(int, stdin.readline().split()))

if K == N:
    print(0)
    print(1)
    exit(0)
if K < N:
    print(N-K)
    print(1)
    exit(0)

prev_pos = set()
prev_pos.add(N)
position = set()
position.add(N)
combis = 0
time = 0
fastest_time = 0
while len(position) != 0:
    if fastest_time != 0:
        break
    next_position = []

    for current_position in position:
        if 0 <= current_position <= 100000:

            if current_position == K:
                if fastest_time != 0:
                    if fastest_time == time:
                        combis += 1
                else:
                    fastest_time = time
                    combis += 1
                continue

            else:
                if current_position-1 not in prev_pos:
                    next_position.append(current_position-1)
                if current_position+1 not in prev_pos:
                    next_position.append(current_position+1)
                if current_position*2 not in prev_pos:
                    next_position.append(current_position*2)

    position = next_position
    prev_pos |= set(next_position)
    time += 1

print(fastest_time)
print(combis)