# from matplotlib import pyplot as plt
# import numpy as np

# # arr = np.array([[10.2, 9.3, 13.4, 11.2, 12.3]])
# # print(arr.shape)
# # arr = arr[0]
# # print(arr.shape)

# # arr = []
# # arr.append(1)
# # arr.append(13)
# # arr.append(10)
# # plt.plot(arr)
# # plt.show()

# # arr = np.array([1,6,3,0,2,2,0])
# # arr_nonzero = arr[arr.nonzero()]
# # print(len(arr_nonzero[arr_nonzero > 2]))

# def a(x):
#     if x > 10:
#         return True, x
#     else:
#         return False
# for i in range(5,15):
#     try:
#         print(a(i))
#     except:
#         print("false")

# # result, b =


f = open("/home/user/python_projects/zipsa_models/water_bottle/models_obj/ii.obj")
wf = open("/home/user/python_projects/zipsa_models/water_bottle/textured_simple_s.obj", mode='wt')
# f.readlines()
while 1:
    input_line = f.readline()
    if not input_line:
        break
    input_line = input_line[:-1]
    input_line_split = input_line.split(' ')
    save_line = ""
    if input_line_split[0] == 'v':
        # print(input_line)
        save_line += "v " + str(float(input_line_split[1]) * 0.001) + " " + str(float(input_line_split[2]) * 0.001) + " " + str(float(input_line_split[3]) * 0.001) + "\n"
        wf.write(save_line)

    # elif input_line_split[0] == 'f':
    #     # print(input_line_split)
    #     save_line += "f " + input_line_split[1][:-1] + input_line_split[2][:-1] + input_line_split[3][:-2]
    #     input_line = f.readline()
    #     input_line_split = input_line.split(' ')

    #     try:
    #         save_line += " " + input_line_split[1][:-1] + input_line_split[3][:-2] + input_line_split[2][:-2]
    #     except:
    #         save_line += "\n"
    #         continue
    #     input_line = f.readline()
    #     input_line_split = input_line.split(' ')

    #     try:
    #         save_line += " " + input_line_split[1][:-1] + input_line_split[3][:-2] + input_line_split[2][:-2] + "\n"
    #     except:
    #         save_line += "\n"
    #         continue
        
        # print(save_line)
        # wf.write(save_line)
    else:
        print(input_line)
        wf.write(input_line + "\n")
        pass
f.close()
wf.close()