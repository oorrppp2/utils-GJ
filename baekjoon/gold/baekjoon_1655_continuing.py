# Baekjoon 1655


from sys import stdin
from queue import PriorityQueue

N = int(stdin.readline())
s = PriorityQueue() # max is top
l = PriorityQueue() # min is top
l_min = 10001
s_max = -10001

for i in range(N):
    num = int(stdin.readline())
    s_len = s.qsize()
    l_len = l.qsize()

    if s_len > l_len:
        l.put(num)
        if num < l_min:
            l_min = num
    else:
        s.put((-num, num))
        if num > s_max:
            s_max = num
    if not s.empty() and not l.empty() and s_max > l_min:
        max_val = s.get()[1]
        min_val = l.get()
        s.put((-min_val, min_val))
        l.put(max_val)
        s_max = s.queue[0][1]
        l_min = l.queue[0]

    print(s_max)
    