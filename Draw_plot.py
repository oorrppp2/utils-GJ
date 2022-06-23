from matplotlib import pyplot as plt
import numpy as np

class_name = ['master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
              'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
              'large_clamp', 'extra_large_clamp', 'foam_brick']

full_points_all_how_match_score = []
full_points_all_dis = []
few_points_all_how_match_score = []
few_points_all_dis = []
for classes in class_name:
    full_points_input_file = open('/home/user/python_projects/Densefusion_posecnn_base/experiments/score_distance_results_full_points/' + classes + '/result.txt', mode='rt')
    few_points_input_file = open('/home/user/python_projects/Densefusion_posecnn_base/experiments/score_distance_results/' + classes + '/result.txt', mode='rt')
    how_match_score = []
    dis = []
    while 1:
        input_line = full_points_input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        how_match_score.append(float(input_line[5]))
        dis.append(float(input_line[7]))

        full_points_all_how_match_score.append(float(input_line[5]))
        full_points_all_dis.append(float(input_line[7]))

    while 1:
        input_line = few_points_input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')

        few_points_all_how_match_score.append(float(input_line[5]))
        few_points_all_dis.append(float(input_line[7]))

    # how_match_score = np.array(how_match_score)
    # dis = np.array(dis)
    #
    # sorted_arg = np.argsort(how_match_score)
    #
    # how_match_score = how_match_score[sorted_arg]
    # dis = dis[sorted_arg]
    #
    # # plt.subplot(2,1,1)
    # plt.plot(how_match_score, dis, 'r')
    # plt.title(classes + ' distance ratio')
    #
    # # plt.figure(classes)
    # plt.show()


full_points_all_how_match_score = np.array(full_points_all_how_match_score)
full_points_all_dis = np.array(full_points_all_dis)

sorted_arg = np.argsort(full_points_all_how_match_score)

full_points_all_how_match_score = full_points_all_how_match_score[sorted_arg]
full_points_all_dis = full_points_all_dis[sorted_arg]


few_points_all_how_match_score = np.array(few_points_all_how_match_score)
few_points_all_dis = np.array(few_points_all_dis)

sorted_arg = np.argsort(few_points_all_how_match_score)

few_points_all_how_match_score = few_points_all_how_match_score[sorted_arg]
few_points_all_dis = few_points_all_dis[sorted_arg]

# prefered_final_dis = 0.0001 * np.random.randn(len(all_initial_dis))

print("full_points_all_how_match_score size : " + str(len(full_points_all_how_match_score)))
print("few_points_all_how_match_score size : " + str(len(few_points_all_how_match_score)))

plt.subplot(2,2,1)
plt.plot(full_points_all_how_match_score, full_points_all_dis, 'r')
plt.title('Full points (.25M +) match score/distance ratio')
# plt.plot(all_initial_dis, (all_initial_dis - prefered_final_dis)/all_initial_dis, 'r')
# plt.ylim([0, 1.5])
# plt.title('Distance ratio( (init_dis - desired_final_dis) / init_dis )')

plt.subplot(2,2,2)
plt.plot(few_points_all_how_match_score, few_points_all_dis, 'r')
plt.title('Few points (2621) match score/distance ratio')

plt.subplot(2,2,3)
plt.hist(full_points_all_how_match_score, bins=10, histtype='step')
plt.title('Full points score histogram')
# plt.plot(all_initial_dis, (all_initial_dis - prefered_final_dis)/all_initial_dis, 'r')
# plt.ylim([0, 1.5])
# plt.title('Distance ratio( (init_dis - desired_final_dis) / init_dis )')

plt.subplot(2,2,4)
plt.hist(few_points_all_how_match_score, bins=10, histtype='step')
plt.title('Few points score histogram')

plt.show()