import torch
import numpy as np

root_dir = 'F:/3D-Object-Detection/dataset/'

class_list = ["Car", "Pedestrian", "Cyclist"]

CLASS_NAME_TO_ID = {
	'Car': 				0,
	'Pedestrian': 		1,
	'Cyclist': 			2,
	'Van': 				0,
	'Person_sitting': 	1,
}

boundary = {
	"minX": 0,
	"maxX": 40,
	"minY": -40,
	"maxY": 40,
	"minZ": -2.73,
	"maxZ": 1.27
}
BEV_WIDTH = 1024 # across y axis -40m ~ 40m
BEV_HEIGHT = 512 # across x axis 0m ~ 40m
'''

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

BEV_WIDTH = 608 # across y axis -40m ~ 40m
BEV_HEIGHT = 608 # across x axis 0m ~ 40m
'''

DISCRETIZATION = (boundary["maxX"] - boundary["minX"])/BEV_HEIGHT

nms_iou_threshold = 0.1
colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
#anchors = [[1.26,1.56,0,0], [1.26,1.56,1,0], [0.472,0.32,0,0], [0.472,0.32,1,0], [0.472,0.704,0,0]]
# Car/Van [0,90], Cyclist[0,90] Ped/Person_sitting[90]
#anchors = [[0.8290, 1.7840, 0, 1], [0.8290, 1.7840, 1, 0], [0.8905, 0.9111, 0, 1], [0.8905, 0.9111, 1, 0], [0.9012, 0.5303, 1, 0]]
anchors = [[0.8290, 1.7840, 0, 0], [0.8290, 1.7840, 1, 0], [0.5303, 0.9012, 0, 0], [0.5303, 0.9012, 1, 0], [0.8905, 0.9111, 0, 0]]

conf_thresh = 0.3
nms_thresh = 0.4

Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
	])

# cal mean from train set
R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
])

P2 = np.array([[719.787081,         0., 608.463003,    44.9538775],
               [        0., 719.787081, 174.545111,     0.1066855],
               [        0.,         0.,         1., 3.0106472e-03],
			   [0., 0., 0., 0]
])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)

# select gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")