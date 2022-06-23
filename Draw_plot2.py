from matplotlib import pyplot as plt
import numpy as np

class_name = ['master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
              'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
              'large_clamp', 'extra_large_clamp', 'foam_brick']

all_match_score = []
all_dist = []
for classes in class_name:
    # distance_input_file = open('/home/user/python_projects/Densefusion_posecnn_base/experiments/score_distance_results_full_points/' + classes + '/result.txt', mode='rt')
    matching_scores_input_file = open('/home/user/python_projects/Densefusion_posecnn_base/experiments/matching_scores/' + classes + '/result.txt', mode='rt')
    dist = []
    matching_score = []

    while 1:
        input_line = matching_scores_input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')

        dist.append(float(input_line[1]))
        all_dist.append(float(input_line[1]))

        matching_score.append(float(input_line[3]))
        all_match_score.append(float(input_line[3]))

    dist = np.array(dist)
    matching_score = np.array(matching_score)

    sorted_arg = np.argsort(dist)

    dist = dist[sorted_arg]
    matching_score = matching_score[sorted_arg]

    # plt.subplot(2,1,1)
    plt.plot(dist, matching_score, 'r')
    plt.title(classes + ' x : distance / y : matching_score')

    # plt.figure(classes)
    plt.show()


all_match_score = np.array(all_match_score)
all_dist = np.array(all_dist)

sorted_arg = np.argsort(all_dist)

all_match_score = all_match_score[sorted_arg]
all_dist = all_dist[sorted_arg]

plt.plot(all_dist, all_match_score, 'r')

plt.show()