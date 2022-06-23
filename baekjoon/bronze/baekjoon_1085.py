xywh = list(map(int, input().split(' ')))
x = xywh[0]
y = xywh[1]
w = xywh[2]
h = xywh[3]

x_min = x if w - x > x else w - x
y_min = y if h - y > y else h - y
print(x_min if x_min < y_min else y_min)