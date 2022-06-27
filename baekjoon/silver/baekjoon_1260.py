# Baekjoon 1018

from sys import stdin
from queue import PriorityQueue


class node:
    def __init__(self, name):
        self.name = name
        self.neighbors = PriorityQueue()

N, M, V = list(map(int, stdin.readline().split()))
nodes_dfs = [node(i+1) for i in range(N)]
visited_dfs = [False for _ in range(N)]
nodes_bfs = [node(i+1) for i in range(N)]
visited_bfs = [False for _ in range(N)]
for i in range(M):
    p1, p2 = list(map(int, stdin.readline().split()))
    nodes_dfs[p1-1].neighbors.put(p2-1)
    nodes_dfs[p2-1].neighbors.put(p1-1)
    nodes_bfs[p1-1].neighbors.put(p2-1)
    nodes_bfs[p2-1].neighbors.put(p1-1)

keep_going = True
dfs_queue = [V-1]
bfs_queue = [V-1]
dfs_answer = []
bfs_answer = []
while len(dfs_queue) != 0:
    dfs_node_n = dfs_queue.pop(0)
    if visited_dfs[dfs_node_n]:
        continue
    dfs_node = nodes_dfs[dfs_node_n]
    visited_dfs[dfs_node_n] = True
    dfs_answer.append(dfs_node.name)

    next_queue = []
    while not dfs_node.neighbors.empty():
        next_queue.append(dfs_node.neighbors.get())
    dfs_queue = next_queue + dfs_queue
    
    if len(dfs_answer) == N:
        break

    
while len(bfs_queue) != 0:
    bfs_node_n = bfs_queue.pop(0)
    if visited_bfs[bfs_node_n]:
        continue
    bfs_node = nodes_bfs[bfs_node_n]
    visited_bfs[bfs_node_n] = True
    bfs_answer.append(bfs_node.name)

    while not bfs_node.neighbors.empty():
        bfs_queue.append(bfs_node.neighbors.get())
    
    if len(bfs_answer) == N:
        break

for d in dfs_answer:
    print(d, end=' ')
print()
for b in bfs_answer:
    print(b, end=' ')
