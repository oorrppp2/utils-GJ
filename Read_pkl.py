import pickle

# temp_dict = {'name': 'S', 'id': 1}

# # 데이터 저장
# with open('filename.pkl', 'wb') as f:
# 	pickle.dump(temp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
root_dir = '/media/user/433c5472-5bea-42d9-86c4-e0794e47477f/NOCS_CVPR2019/data/gts/real_test/'
# 데이터 로드
file_name = root_dir + 'results_real_test_scene_1_0000.pkl'
with open(file_name, 'rb') as f:
	data = pickle.load(f)

print(data)