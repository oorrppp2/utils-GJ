# Baekjoon 2579

from sys import stdin
N = int(stdin.readline())

l = {}
ans = 0
scores = []
for i in range(N):
    score = int(stdin.readline())
    scores.append(score)
    l[i % 2] = []
    if i == 0:
        if N == 1:
            print(score)
            exit(0)
        l[i].append([score, True])
        l[i].append([0, False])
    elif i == 1:
        if N == 2:
            print(sum(scores))
            exit(0)

        for c in l[i-1]:
            prev_score = c[0]
            prev_step = c[1]
            if prev_step:
                l[i%2].append([prev_score, prev_score+score, True, True])
                l[i%2].append([prev_score, prev_score, True, False])
            else:
                l[i%2].append([prev_score, score, False, True])
    else:
        if N == 3:
            print(scores[2] + max(scores[0], scores[1]))
            exit(0)
        if i == N-2:
            last_score = int(stdin.readline())
            for c in l[(i-1)%2]:
                prev_score = c[1]
                prev_step = c[3]
                if prev_step:
                    if prev_score + last_score > ans:
                        ans = prev_score + last_score
                else:
                    if prev_score + scores[-1] + last_score > ans:
                        ans = prev_score + scores[-1] + last_score
            break
        else:
            TrueTrue_group = []
            TrueFalse_group = []
            FalseTure_group = []
            for c in l[(i-1)%2]:
                prev_score = c[1]
                prev_step = c[3]
                pprev_score = c[0]
                pprev_step = c[2]

                if not pprev_step:
                    TrueTrue_group.append([prev_score, prev_score + score, True, True])
                    if prev_step:
                        TrueFalse_group.append([prev_score, prev_score, True, False])
                else:
                    if not prev_step:
                        FalseTure_group.append([prev_score, prev_score + score, False, True])
                    else:
                        TrueFalse_group.append([prev_score, prev_score, True, False])
            l[i%2].append(max(TrueTrue_group))
            l[i%2].append(max(TrueFalse_group))
            l[i%2].append(max(FalseTure_group))
    # print(i, l)
print(ans)
