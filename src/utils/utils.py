import numpy as np
import torch
import cv2

from src.postprocess import compute_matches, non_max_suppression
from src.utils.kitti_utils import get_corners
import src.config as cnf

def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] + abs(minZ)

    return PointCloud

def makeBVFeature(PointCloud_, Discretization, bc):

    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height - 1, Width - 1, 3))
    RGB_Map[:, :, 0] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[:, :, 1] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[:, :, 2] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    return RGB_Map

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors):
    if output.dim() == 3:
        output = output.unsqueeze(0)

    assert (output.size(1) == (7 + num_classes) * num_anchors)

    nA = num_anchors  # num_anchors = 5
    nB = output.size(0)
    nC = num_classes  # num_classes = 3
    nH = output.size(2)  # nH  16
    nW = output.size(3)  # nW  32

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if output.is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if output.is_cuda else torch.ByteTensor

    prediction = output.view(nB, nA, 7 + num_classes, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    im = prediction[..., 4]  # Im
    re = prediction[..., 5]  # Re
    pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

    # Calculate offsets for each grid
    grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
    grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
    scaled_anchors = FloatTensor([(a_w, a_h) for a_w, a_h, _, _ in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction.shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    pred_boxes[..., 4] = im.data
    pred_boxes[..., 5] = re.data

    pred_boxes[..., 6] = pred_conf
    pred_boxes[..., 7:(7 + nC)] = pred_cls

    # Divide position and size by cell size to normalize in range [0~1]
    pred_boxes[..., 0] /= nW
    pred_boxes[..., 1] /= nH
    pred_boxes[..., 2] /= nW
    pred_boxes[..., 3] /= nH

    pred_boxes = to_cpu(pred_boxes)

    all_boxes = []
    for j in range(nB):
        boxes = pred_boxes[j, ...].view(-1, 10)
        selected_box_index = boxes[:, 6] > conf_thresh
        all_boxes.append(boxes[selected_box_index])
    return all_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def to_cpu(tensor):
    return tensor.detach().cpu()

def get_matching_statistics(target, all_boxes, eval_class_id, iou_threshold=0.5, img=None, show_image=False):

    target_corners = []
    pred_corners = []
    pred_scores = []

    gts = 0
    preds = 0

    for j in range(50):
        if (np.sum(target[j, 1:]) == 0):
            continue
        if target[j][0] == eval_class_id:
            x = int(target[j][1] * float(cnf.BEV_WIDTH))  # 32 cell = 1024 pixels
            y = int(target[j][2] * float(cnf.BEV_HEIGHT))  # 16 cell = 512 pixels
            w = int(target[j][3] * float(cnf.BEV_WIDTH))  # 32 cell = 1024 pixels
            l = int(target[j][4] * float(cnf.BEV_HEIGHT))
            angle = np.arctan2(target[j][5], target[j][6])

            bev_corners = get_corners(x, y, w, l, angle)
            target_corners.append(bev_corners)

            if show_image:
                corners_int = bev_corners.astype(int).reshape(-1, 1, 2)
                cv2.polylines(img, [corners_int], True, (255, 255, 255), 1)

    gts += len(target_corners)
    target_corners = np.array(target_corners)

    for i in range(len(all_boxes)):
        cls_prob = all_boxes[i][7:]
        val, cls_id = torch.max(cls_prob, 0)

        if cls_id.item() == eval_class_id:
            x = int(all_boxes[i][0] * cnf.BEV_WIDTH)
            y = int(all_boxes[i][1] * cnf.BEV_HEIGHT)
            w = int(all_boxes[i][2] * cnf.BEV_WIDTH)
            l = int(all_boxes[i][3] * cnf.BEV_HEIGHT)

            angle = np.arctan2(all_boxes[i][4], all_boxes[i][5])

            bev_corners = get_corners(x, y, w, l, angle)
            pred_corners.append(bev_corners)
            pred_scores.append(val.item())

    pred_scores = np.array(pred_scores)
    pred_corners = np.array(pred_corners)
    if len(pred_scores) > 0:
        selected_ids = non_max_suppression(pred_corners, pred_scores, cnf.nms_iou_threshold)
        pred_scores = pred_scores[selected_ids]
        pred_corners = pred_corners[selected_ids]

    if show_image:
        for i in range(len(pred_corners)):
            corners_int = pred_corners[i].astype(int).reshape(-1, 1, 2)
            cv2.polylines(img, [corners_int], True, cnf.colors[eval_class_id], 2)
            corners_int = corners_int.reshape(-1, 2)
            cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]),
                     (255, 255, 0), 2)

    preds += pred_scores.shape[0]
    _, pred_match, _ = compute_matches(target_corners, pred_corners, pred_scores, iou_threshold=iou_threshold)

    return pred_scores, pred_match, gts, preds

#send parameters in bev image coordinates format
def drawRotatedBox(img,x,y,w,l,yaw,color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)