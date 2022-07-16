# Baekjoon 15900

# 이기는 조건: 루트까지 홀수의 간선으로 이루어진 가지 (ex. 1 - 3 - 5 - 7)의 수가 홀수일 때. 
# neighbor 정보를 입력으로 받고,
# 루트부터 리프까지 DFS로 돌면서 edge 갯수 count

from sys import stdin
N = int(stdin.readline())
neighbors = [[] for _ in range(N)]
edge_count = [0 for _ in range(N)] # 각 노드의 edge 갯수
visited = [False for _ in range(N)]
leaf_nodes = []
odd_edge_count = 0

for i in range(N-1):
    p1, p2 = list(map(int, stdin.readline().split()))
    p1 -= 1
    p2 -= 1
    neighbors[p1].append(p2)
    neighbors[p2].append(p1)

queue = [0] # root
visited[0] = True
while len(queue) > 0:
    cur_node = queue.pop()
    cur_neighbor = neighbors[cur_node]
    if len(cur_neighbor) == 1 and cur_node != 0: # leaf node
        leaf_nodes.append(cur_node)
    else:
        for n in cur_neighbor:
            if not visited[n]:
                queue.append(n)
                edge_count[n] += edge_count[cur_node] + 1
                visited[n] = True
# print(edge_count)
# print(leaf_nodes)

for leaf in leaf_nodes:
    if edge_count[leaf] % 2 == 1:
        odd_edge_count += 1
if odd_edge_count % 2 == 1:
    print('Yes')
else:
    print('No')