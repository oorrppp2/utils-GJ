def solution(clothes):
    answer = 1
    clothes_dict = {}
    for cloth in clothes:
        try:
            clothes_dict[cloth[1]].append(cloth[0])
        except:
            clothes_dict[cloth[1]] = [cloth[0]]
    for cloth in clothes_dict:
        answer *= len(clothes_dict[cloth]) + 1
    return answer-1

cloth = [["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]]

print(solution(cloth))