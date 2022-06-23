# import numpy as np
# def min_max_normalize(lst):
#     normalized = []
    
#     for value in lst:
#         normalized_num = (value - min(lst)) / (max(lst) - min(lst))
#         normalized.append(normalized_num)
    
#     return normalized


# # lst = [82.1, 83.7, 55.3, 49.2, 90.6, 85.2, 79.2]
# # lst = [80.6, 84.9, 53.9, 49.9, 89.3, 85.1, 79.7]
# lst = [82.0, 84.2, 51.3, 44.1, 87.4, 83.8, 76.8,
#  80.6, 84.7, 53.0, 46.8, 86.8, 84.7, 79.1,
#  81.2, 84.4, 53.0, 48.2, 89.5, 84.3, 78.0,
#  82.1, 83.7, 55.3, 49.2, 90.6, 85.2, 79.2]

# c1 = [lst[0], lst[7], lst[14], lst[21]]
# c2 = [lst[1], lst[8], lst[15], lst[22]]
# c3 = [lst[2], lst[9], lst[16], lst[23]]
# c4 = [lst[3], lst[10],lst[ 17],lst[ 24]]
# c5 = [lst[4], lst[11],lst[ 18],lst[ 25]]
# c6 = [lst[5], lst[12],lst[ 19],lst[ 26]]
# c7 = [lst[6], lst[13],lst[ 20],lst[ 27]]

# norm1 = min_max_normalize(c1)
# norm2 = min_max_normalize(c2)
# norm3 = min_max_normalize(c3)
# norm4 = min_max_normalize(c4)
# norm5 = min_max_normalize(c5)
# norm6 = min_max_normalize(c6)
# norm7 = min_max_normalize(c7)
# # print(norm)

# # print(np.average(norm[:7]))
# print(norm1[3])
# print(norm2[3])
# print(norm3[3])
# print(norm4[3])
# print(norm5[3])
# print(norm6[3])
# print(norm7[3])

# print(np.average([norm1[3],
#                   norm2[3],
#                   norm3[3],
#                   norm4[3],
#                   norm5[3],
#                   norm6[3],
#                   norm7[3]]))


import numpy as np
def min_max_normalize(lst):
    normalized = []
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    
    return normalized

AUC = [90.59, 90.87, 94.54, 94.74]
cm2 = [89.44, 89.34, 95.99, 96.14]
cm1 = [72.65, 72.57, 90.09, 90.62]
cm05 = [47.99, 52.86, 78.97, 81.39]


norm1 = min_max_normalize(AUC)
norm2 = min_max_normalize(cm2)
norm3 = min_max_normalize(cm1)
norm4 = min_max_normalize(cm05)


# print(np.average([norm1[0],
#                   norm2[0],
#                   norm3[0],
#                   norm4[0]]))
lst1 = [90.59, 89.44, 72.65, 47.99]
lst2 = [90.87, 89.34, 72.57, 52.86]
lst3 = [94.57, 95.99, 90.09, 78.97]
lst4 = [94.74, 96.14, 90.62, 81.39]

print(np.mean(lst1))
print(np.mean(lst2))
print(np.mean(lst3))
print(np.mean(lst4))
