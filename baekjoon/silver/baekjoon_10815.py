# Baekjoon 10815

from sys import stdin

cards = [0 for _ in range(20000001)]
N = int(stdin.readline())
sg_cards = list(map(int, stdin.readline().split()))
for card in sg_cards:
    cards[card+10000000] = 1

M = int(stdin.readline())
in_cards = list(map(int, stdin.readline().split()))

for card in in_cards:
    print(cards[card+10000000], end=' ')