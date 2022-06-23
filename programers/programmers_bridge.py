def solution(bridge_length, weight, truck_weights):
    answer = 0
    pass_away = []
    num_trucks = len(truck_weights)
    on_the_bridge = []
    times = []
    while len(pass_away) != num_trucks:
        answer += 1
        print("answer : ", answer)
        if len(on_the_bridge) > 0:
            if times[0] == 0:
                pass_away.append(on_the_bridge[-1])
                times = times[1:]
                on_the_bridge = on_the_bridge[1:]
        if len(truck_weights) > 0:
            current = truck_weights[0]
            if sum(on_the_bridge) + current <= weight:
                on_the_bridge.append(current)
                times.append(bridge_length)
                truck_weights = truck_weights[1:]
        for i in range(len(times)):
            times[i] -= 1

        print("pass_away: ", pass_away)
        print("on_the_bridge: ", on_the_bridge)
        print("times: ", times)
        # if answer == 30:
        #     break
    return answer


bridge_length = 100
weight = 100
truck_weights = [10,10,10,10,10,10,10,10,10,10]
# bridge_length = 2
# weight = 10
# truck_weights = [7,4,5,6]

ans = solution(bridge_length, weight, truck_weights)
print(ans)