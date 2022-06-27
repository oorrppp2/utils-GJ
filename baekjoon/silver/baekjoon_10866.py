# Baekjoon 10845

from sys import stdin
N = int(stdin.readline())

stack = []
for i in range(N):
    message = stdin.readline().split()
    if message[0] == 'push_front':
        stacked = [message[1]]
        stack = stacked + stack
        # stack.append(message[1])
    elif message[0] == 'push_back':
        stack.append(message[1])
    elif message[0] == 'pop_front':
        if len(stack) != 0:
            print(stack.pop(0))
        else:
            print(-1)
    elif message[0] == 'pop_back':
        if len(stack) != 0:
            print(stack.pop())
        else:
            print(-1)
    elif message[0] == 'size':
        print(len(stack))
    elif message[0] == 'empty':
        print(1 if len(stack) == 0 else 0)
    elif message[0] == 'front':
        if len(stack) != 0:
            print(stack[0])
        else:
            print(-1)
    elif message[0] == 'back':
        if len(stack) != 0:
            print(stack[-1])
        else:
            print(-1)