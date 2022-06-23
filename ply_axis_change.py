

# f = open("/home/user/python_projects/zipsa_models/coca_cola/models_obj/ii.obj")
# wf = open("/home/user/python_projects/zipsa_models/coca_cola/textured_simple_s.obj", mode='wt')
# # f.readlines()
# while 1:
#     input_line = f.readline()
#     if not input_line:
#         break
#     input_line = input_line[:-1]
#     input_line_split = input_line.split(' ')
#     save_line = ""
#     if input_line_split[0] == 'v':
#         # print(input_line)
#         save_line += "v " + str((input_line_split[1])) + " " + str((input_line_split[3])) + " " + str((input_line_split[2])) + "\n"
#         wf.write(save_line)

#     elif input_line_split[0] == 'f':
#         # print(input_line_split)
#         save_line += "f " + input_line_split[1] + " " + input_line_split[3] + " " + input_line_split[2] + "\n"
        
#         # print(save_line)
#         wf.write(save_line)
#     else:
#         # print(input_line)
#         wf.write(input_line + "\n")
#         pass
# f.close()
# wf.close()


models = {1:'Ape', 5:'Can', 6:'Cat', 8:'Driller', 9:'Duck', 10:'Eggbox', 11:'Glue', 12:'Holepuncher'}

# dataset_root_dir = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/lmo_test_all/test/000002'
# f = open("/home/user/python_projects/zipsa_models/water_bottle/models_obj/ii.obj")
# wf = open("/home/user/python_projects/zipsa_models/water_bottle/textured_simple_s.obj", mode='wt')

# for model_id in models:
#     model = models[model_id]
#     f = open(dataset_root_dir + "/models/{0}/model.obj".format(model))
#     wf = open(dataset_root_dir + "/models/{0}/model_s.obj".format(model), mode='wt')
#     while 1:
#         input_line = f.readline()
#         if not input_line:
#             break
#         input_line = input_line[:-1]
#         input_line_split = input_line.split(' ')
#         save_line = ""
#         if input_line_split[0] == 'v':
#             # print(input_line)
#             save_line += "v " + str((input_line_split[1])) + " " + str((input_line_split[3])) + " " + str((float(input_line_split[2])-0.11)) + "\n"
#             # save_line += "v " + str(float(input_line_split[1])*0.001) + " " + str(float(input_line_split[2])*0.001) + " " + str(float(input_line_split[3])*0.001) + "\n"
#             wf.write(save_line)

#         # elif input_line_split[0] == 'f':
#         #     # print(input_line_split)
#         #     save_line += "f " + input_line_split[1] + " " + input_line_split[3] + " " + input_line_split[2] + "\n"
            
#         #     # print(save_line)
#         #     wf.write(save_line)
#         else:
#             # print(input_line)
#             wf.write(input_line + "\n")
#             pass
#     f.close()
#     wf.close()


# f = open("/home/user/python_projects/ycb_models/coca_cola/textured_simple.obj")
# wf = open("/home/user/python_projects/ycb_models/coca_cola/textured_simple_s.obj", mode='wt')

f = open("/home/user/python_projects/6D_pose_estimation_particle_filter/models/lmo/models_obj/ox.obj")
wf = open("/home/user/python_projects/6D_pose_estimation_particle_filter/models/lmo/models_obj/ox_s.obj", mode='wt')

while 1:
    input_line = f.readline()
    if not input_line:
        break
    input_line = input_line[:-1]
    input_line_split = input_line.split(' ')
    save_line = ""
    if input_line_split[0] == 'v':
        # print(input_line)
        save_line += "v " + str((input_line_split[1])) + " " + str((input_line_split[3])) + " " + str((float(input_line_split[2]) * -1.0)) + "\n"
        # save_line += "v " + str(float(input_line_split[1])*0.001) + " " + str(float(input_line_split[2])*0.001) + " " + str(float(input_line_split[3])*0.001) + "\n"
        wf.write(save_line)

    elif input_line_split[0] == 'f':
        # print(input_line_split)
        save_line += "f " + input_line_split[1] + " " + input_line_split[3] + " " + input_line_split[2] + "\n"
        
    #     # print(save_line)
        wf.write(save_line)
    else:
        # print(input_line)
        wf.write(input_line + "\n")
        pass
f.close()
wf.close()

# dataset_root = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/OCCLUSION_LINEMOD'
# models = {1:'Ape', 4:'Can', 5:'Cat', 6:'Driller', 7:'Duck', 8:'Eggbox', 9:'Glue', 10:'Holepuncher'}

# for model in models:
#     f = open('{0}/models/{1}/00'.format(dataset_root, models[model])+str(model)+'.xyz')
#     wf = open('{0}/models/{1}/00'.format(dataset_root, models[model])+str(model)+'.xzy', mode='wt')
#     while 1:
#         input_line = f.readline()
#         if not input_line:
#             break
#         input_line = input_line[:-1]
#         input_line_split = input_line.split(' ')
#         save_line = ""
#         save_line += str(input_line_split[0]) + " " + str(input_line_split[2]) + " " + str(input_line_split[1]) + "\n"
#         wf.write(save_line)

#     f.close()
#     wf.close()




# import open3d as o3d
# import numpy as np

# pcd_path = '/home/user/python_projects/zipsa_models/water_bottle/textured_simple_s.pcd'
# pcd = o3d.io.read_point_cloud(pcd_path)
# pcd = np.asarray(pcd.points)

# x_max = np.max(pcd[:,0])
# x_min = np.min(pcd[:,0])
# y_max = np.max(pcd[:,1])
# y_min = np.min(pcd[:,1])
# z_max = np.max(pcd[:,2])
# z_min = np.min(pcd[:,2])

# print(z_max)
# print(z_min)