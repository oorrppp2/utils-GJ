def solution(priorities, location):
    answer = location+1
    i = 0
    while(len(priorities) > 0):
    # for i in range(len(priorities)):
        # print("i: ", i)
        # print("answer: ", answer)
        current = priorities[0]
        exist = False
        for p in priorities:
            if current < p:
                exist = True
                break
        if exist:
            priorities = priorities[1:]
            priorities.append(current)
            if i == location:
                answer += len(priorities)-1
                i = -1
                location = len(priorities)-1
            else:
                answer -= 1
        else:
            if i == location:
                break
            priorities = priorities[1:]
        i += 1
    return answer

# priority = [9,1,2,1,2,3]
# location = 3
priority = [2,1,3,2]
location = 2

ans = solution(priority, location)
print(ans)