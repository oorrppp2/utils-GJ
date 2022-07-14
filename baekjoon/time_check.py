import time
import numpy as np

# m = 0
# t = time.time()
# a = [np.random.rand(1) for _ in range(1000000)]
# print(max(a))
# print(time.time() - t)
# t = time.time()
# for i in range(len(a)):
#     if a[i] > m:
#         m = a[i]

# print(m)
# print(time.time() - t)


# m = 0
# t = time.time()
# a = [np.random.rand(1) for _ in range(1000000)]
# print(max(a))
# print(time.time() - t)
# t = time.time()
# for i in range(len(a)):
#     if a[i] > m:
#         m = a[i]

# print(m)
# print(time.time() - t)


# from sys import stdin
# from queue import PriorityQueue

# s = PriorityQueue() # max is top
# l = PriorityQueue() # min is top

# for i in range(100):
#     s.put(np.random.rand(1))
# t = time.time()
# print(s.qsize())
# print(time.time() - t)

# for i in range(1000000):
#     l.put(np.random.rand(1))
# t = time.time()
# print(l.qsize())
# print(time.time() - t)

from sys import stdin

N = 9
paper = [ [1, 2, 3], 
          [4, 5, 6], 
          [8, 0, 2]  ]
paper = []
for i in range(9):
    paper.append(list(map(int, stdin.readline().split())))
next_papers = []
# next_papers.append(paper[:N//3][0][:N//3])
# next_papers.append(paper[N//3:N//3*2][0][:N//3])
# next_papers.append(paper[N//3*2:][0][:N//3])

# next_papers.append(paper[:N//3][0][N//3:N//3*2])
# next_papers.append(paper[N//3:N//3*2][0][N//3:N//3*2])
# next_papers.append(paper[N//3*2:][0][N//3:N//3*2])

# next_papers.append(paper[:N//3][0][N//3*2:])
# next_papers.append(paper[N//3:N//3*2][0][N//3*2:])
# next_papers.append(paper[N//3*2:][0][N//3*2:])

# print(next_papers)

# print(paper[N//3*2:][0][N//3*2:])
# print(paper[N//3*2:][:])

# new_paper = []
# for i in range(N//3):
#     c_paper = paper[i]
#     # print(c_paper)
#     for j in range(N//3):
#         new_paper.append(c_paper[j])
#     next_papers.append(new_paper)
# new_papers = [[] for _ in range(9)]
# for i in range(N//3):
#     for j in range(N//3):
#         row = []
#         for e in paper[i][0]:
#             row.append(e)
#         new_papers[i*3+j].append(paper[N//3*j:N//3*(j+1)])

# print(new_papers)
# for p in new_papers:
#     for r in p:
#         print(r)

    # print()
# import numpy as np
# paper = np.asarray(paper)
# print(paper[:3,:3])

paper1 = []
paper2 = []
paper3 = []
for i in range(N//3):
    paper1.append(paper[i][:N//3])
    paper2.append(paper[i][N//3:N//3*2])
    paper3.append(paper[i][N//3*2:])
next_papers.append(paper1)
next_papers.append(paper2)
next_papers.append(paper3)

paper1 = []
paper2 = []
paper3 = []
for i in range(N//3, N//3*2):
    paper1.append(paper[i][:N//3])
    paper2.append(paper[i][N//3:N//3*2])
    paper3.append(paper[i][N//3*2:])
next_papers.append(paper1)
next_papers.append(paper2)
next_papers.append(paper3)

paper1 = []
paper2 = []
paper3 = []
for i in range(N//3*2, N):
    paper1.append(paper[i][:N//3])
    paper2.append(paper[i][N//3:N//3*2])
    paper3.append(paper[i][N//3*2:])
next_papers.append(paper1)
next_papers.append(paper2)
next_papers.append(paper3)
# print(nex)
for p in next_papers:
    for row in p:
        print(row)
    print("="*50)