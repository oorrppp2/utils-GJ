import numpy as np

from matplotlib import pyplot as plt

x = np.arange(0.0, 1.0, 0.1)
our_success = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/ours_360_particles_0.01/success.npy")
our_trial = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/ours_360_particles_0.01/trial.npy")
our_success_rate = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/ours_360_particles_0.01/success_rate.npy")

perch_success = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/perch_0.01/success.npy")
perch_trial = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/perch_0.01/trial.npy")
perch_success_rate = np.load("/home/user/python_projects/6D_pose_estimation_particle_filter/acc_under_occ/perch_0.01/success_rate.npy")

print(our_success)
print(our_trial)
print("================")
print(perch_success)
print(perch_trial)

our_success = our_success[::-1]
our_trial = our_trial[::-1]
perch_success = perch_success[::-1]
perch_trial = perch_trial[::-1]

print(our_success)
print(our_trial)
print("================")
print(perch_success)
print(perch_trial)
# success_rate = np.fliplr(success_rate)
# success_rate = success_rate[::-1]
# success_rate = success_rate[60:]
# print(success_rate)
linewidth = 5
plt.plot(x, our_success/our_trial, label="Ours (360 particles)", linewidth=linewidth)
plt.plot(x, perch_success/perch_trial, label="PERCH", linewidth=linewidth)
plt.legend(fontsize=24, loc="lower right") # 꼭 호출해 주어야만 legend가 달립니다

plt.xlim([0.0, 0.55])
plt.ylim([0.0, 1.0])
plt.xlabel('Invisible surface ratio', fontsize=30)
plt.ylabel('Accuracy (ADD-S < 1cm)', fontsize=30)
plt.show()