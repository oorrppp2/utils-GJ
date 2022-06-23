num = int(input())
for i in range(num):
    result = list(input())
    score = 0
    scores = []
    for r in result:
        if r == 'O':
            scores.append('O')
        else:
            score += (1+len(scores))*len(scores)/2
            scores.clear()

    score += (1+len(scores))*len(scores)/2
    print(int(score))