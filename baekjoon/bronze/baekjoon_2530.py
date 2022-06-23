current_time = input().split(' ')       # Input을 받아서 split
plus_time = int(input())                # 요리하는데 필요한 시간

h = int(current_time[0])                # hour
m = int(current_time[1])                # minute
s = int(current_time[2])                # second

in_seconds = h * 3600 + m * 60 + s      # 현재 시각을 초단위로 변환
end_time = in_seconds + plus_time       # 요리하는데 필요한 시간을 현재 시간에 더함 (아직 초단위임)

h = end_time // 3600                    # 초단위 총 시간에서 hour를 구함 
h %= 24                                 # 디지털 시계의 앞자리가 24가 넘으면 0으로 넘어감 
m = (end_time % 3600) // 60             # 초단위 총 시간에서 minute를 구함 
s = end_time % 60                       # 초단위 총 시간에서 second를 구함 

print(h, m, s)


current_time = input().split(' ')       # Input을 받아서 split
plus_time = int(input())                # 요리하는데 필요한 시간

h = int(current_time[0])                # hour
m = int(current_time[1])                # minute
s = int(current_time[2])                # second

in_seconds = ???                        # 현재 시각을 초단위로 변환
end_time = ???                          # 요리하는데 필요한 시간을 현재 시간에 더함 (아직 초단위임)

h = ???                                 # 초단위 총 시간에서 hour를 구함 
h %= ???                                # 디지털 시계의 앞자리가 24가 넘으면 0으로 넘어감 
m = ???                                 # 초단위 총 시간에서 minute를 구함 
s = ???                                 # 초단위 총 시간에서 second를 구함 

print(h, m, s)
