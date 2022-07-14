# Baekjoon 7662

from sys import stdin
import heapq

T = int(stdin.readline())

for t in range(T):
    k = int(stdin.readline())
    min_heap = []
    max_heap = []
    all_elements = {}
    
    for i in range(k):
        command, n = list(stdin.readline().split())
        if command == 'I':
            heapq.heappush(min_heap, int(n))
            heapq.heappush(max_heap, -int(n))
            try:
                all_elements[int(n)] += 1
            except:
                all_elements[int(n)] = 1
        else:
            if n == '1':
                while True:
                    if len(max_heap) == 0:
                        break
                    if all_elements[-max_heap[0]] == 0:
                        heapq.heappop(max_heap)
                    else:
                        all_elements[-max_heap[0]] -= 1
                        heapq.heappop(max_heap)
                        break
            else:
                while True:
                    if len(min_heap) == 0:
                        break
                    if all_elements[min_heap[0]] == 0:
                        heapq.heappop(min_heap)
                    else:
                        all_elements[min_heap[0]] -= 1
                        heapq.heappop(min_heap)
                        break

    while True:
        if len(min_heap) == 0:
            break
        if all_elements[min_heap[0]] == 0:
            heapq.heappop(min_heap)
        else:
            break
    while True:
        if len(max_heap) == 0:
            break
        if all_elements[-max_heap[0]] == 0:
            heapq.heappop(max_heap)
        else:
            break

    if len(max_heap) > 0:
        print(-heapq.heappop(max_heap), heapq.heappop(min_heap))
    else:
        print("EMPTY")