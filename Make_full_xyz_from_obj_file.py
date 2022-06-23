

with open('/media/user/ssd_1TB/YCB_dataset/models/') as f:
    lines = f.readlines()
    vStrings = [x.strip('v') for x in lines if x.startswith('v ')]