# Baekjoon 1389

from sys import stdin
from queue import PriorityQueue


class node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        self.distance_from_start = 0

N, M = list(map(int, stdin.readline().split()))
nodes_bfs = [node(i+1) for i in range(N)]
for i in range(M):
    p1, p2 = list(map(int, stdin.readline().split()))
    nodes_bfs[p1-1].neighbors.append(p2)
    nodes_bfs[p2-1].neighbors.append(p1)

keep_going = True

ans = 0
smallest_bacon = 1e9
for i in range(N, 0, -1):
    bacon = 0
    for target in range(N):
        target += 1
        if i == target:
            continue
        bfs_queue = [i]
        visited_bfs = [False for _ in range(N)]
        # print(bfs_queue)
        # print('target: ', target)
        while len(bfs_queue) != 0:
            bfs_node_n = bfs_queue.pop(0)
            # print("bfs_node_n: ", bfs_node_n)
            # print(bfs_queue)
            # print("visited_bfs: ", visited_bfs)
            if visited_bfs[bfs_node_n-1]:
                continue
            bfs_node = nodes_bfs[bfs_node_n-1]
            # print("visited node: ", bfs_node.name)
            # print('neighbors : ', bfs_node.neighbors)
            # print('distance : ', bfs_node.distance_from_start)
            visited_bfs[bfs_node_n-1] = True

            if bfs_node.name == target:
                bacon += bfs_node.distance_from_start
                print("-- current bacon: ", bfs_node.distance_from_start)
                # print()
                for n in nodes_bfs:
                    n.distance_from_start = 0
                break
            
            for n in bfs_node.neighbors:
                if nodes_bfs[n-1].distance_from_start == 0:
                    nodes_bfs[n-1].distance_from_start += bfs_node.distance_from_start + 1
                # print(n, ":", nodes_bfs[n-1].distance_from_start)
            bfs_queue += bfs_node.neighbors

    # print("===== bacon: ", bacon)
    if bacon <= smallest_bacon:
        smallest_bacon = bacon
        ans = i

print(ans)