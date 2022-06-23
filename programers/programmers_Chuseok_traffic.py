def solution(lines):
    answer = 0
    start_t = []
    end_t = []
    for line in lines:
        time = line.split(' ')[1].split(':')
        T = float(line.split(' ')[2][:-1])
        # print(time)
        # print(T)
        time_ms = int(time[0]) * 3600 * 1000 + int(time[1]) * 60 * 1000 + float(time[2]) * 1000
        start_t.append(time_ms - T * 1000 + 1)
        # print(time_ms - T * 1000 + 1)
        # print(time_ms)
        # print("====")
        end_t.append(time_ms)
    # print("start_t: ", start_t)
    # print("end_t: ", end_t)
    for end in end_t:
        throughput = 0
        # print("end: ", end)
        for i in range(len(start_t)):
            if start_t[i] >= end and start_t[i] <= end + 999:
                throughput += 1
            if start_t[i] < end and end_t[i] >= end:
                throughput += 1
        answer = throughput if throughput > answer else answer
    return answer

line = [
"2016-09-15 01:00:04.001 2.0s",
"2016-09-15 01:00:07.000 2s"
]
# line = [
# "2016-09-15 01:00:04.002 2.0s",
# "2016-09-15 01:00:07.000 2s"
# ]
# line = [
# "2016-09-15 20:59:57.421 0.351s",
# "2016-09-15 20:59:58.233 1.181s",
# "2016-09-15 20:59:58.299 0.8s",
# "2016-09-15 20:59:58.688 1.041s",
# "2016-09-15 20:59:59.591 1.412s",
# "2016-09-15 21:00:00.464 1.466s",
# "2016-09-15 21:00:00.741 1.581s",
# "2016-09-15 21:00:00.748 2.31s",
# "2016-09-15 21:00:00.966 0.381s",
# "2016-09-15 21:00:02.066 2.62s"
# ]
ans = solution(line)
print(ans)