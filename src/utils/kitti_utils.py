import torch
import numpy as np
import math
import src.utils.object3d as object3d
from shapely.geometry import Polygon
import src.config as cnf

class_list = cnf.class_list
anchors = cnf.anchors # old   //"anchors": [[0.33,0.70], [0.69,0.70], [1.41,0.60], [1.67,0.62], [2.08,0.90]]
bc = cnf.boundary
colors = cnf.colors

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [object3d.Object3d(line) for line in lines]
    return objects

def read_labels_for_bevbox(objects):
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.pos[0], obj.pos[1], obj.pos[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)
    
    if (len(bbox_selected) == 0):
        return np.zeros((1, 8), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False

# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners

def interpret_kitti_label(bbox):
    w, h, l, y, z, x, yaw = bbox[8:15]
    y = -y
    yaw =  (yaw + np.pi / 2)

    return x, y, w, l, yaw

def get_target2(label_file):
    target = np.zeros([50, 7], dtype=np.float32)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()

        if obj_class in cnf.CLASS_NAME_TO_ID.keys():
            bbox = []
            bbox.append(cnf.CLASS_NAME_TO_ID[obj_class])
            bbox.extend([float(e) for e in obj[1:]])

            x, y, w, l, yaw = interpret_kitti_label(bbox)
            cls_id = bbox[0]

            w = w + 0.5
            l = l + 0.5

            location_x = x
            location_y = y

            if (location_x > bc["minX"]) & (location_x < bc["maxX"]) & (location_y > bc["minY"]) & (location_y < bc["maxY"]):
                target[index][0] = float(cls_id)
                target[index][1] = (y - bc["minY"]) / (bc["maxY"]-bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
                target[index][2] = (x - bc["minX"]) / (bc["maxX"]-bc["minX"])  # make sure target inside the covering area (0,1)

                target[index][3] = float(l) #/ (bc["maxY"]-bc["minY"])
                target[index][4] = float(w) #/ (bc["maxX"]-bc["minX"])  # get target width, length

                target[index][5] = math.sin(float(yaw))  # complex YOLO   Im
                target[index][6] = math.cos(float(yaw))  # complex YOLO   Re

                for i in range(len(class_list)):
                    if obj_class == class_list[i]:  # get target class
                        target[index][0] = i
                index = index + 1
    return target

def build_yolo_target(labels):

    target = np.zeros([50, 7], dtype=np.float32)
    
    index = 0
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]

        # ped and cyc labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3

        yaw = np.pi - yaw
        if (x > bc["minX"]) and (x < bc["maxX"]) and (y > bc["minY"]) and (y < bc["maxY"]):
            y1 = (y - bc["minY"]) / (bc["maxY"]-bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
            x1 = (x - bc["minX"]) / (bc["maxX"]-bc["minX"])  # we should put this in [0,1], so divide max_size  40 m
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])

            target[index][0] = cl
            target[index][1] = y1 
            target[index][2] = x1
            target[index][3] = w1
            target[index][4] = l1
            target[index][5] = math.sin(float(yaw))
            target[index][6] = math.cos(float(yaw))

            index = index+1

    return target

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / (box.union(b).area + 1e-12) for b in boxes]

    return np.array(iou, dtype=np.float32)

def rotated_bbox_iou(gt_box, anchor_shapes):
    x,y,w,l,im,re = gt_box[0,:]
    angle = np.arctan2(im, re)
    bbox1 = np.array(get_corners(x, y, w, l, angle)).reshape(-1,4,2)
    bbox1 = convert_format(bbox1)

    bbox2 = []
    for i in range(anchor_shapes.shape[0]):
        x,y,w,l,im,re = anchor_shapes[i,:]
        angle = np.arctan2(im, re)
        bev_corners = get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = convert_format(np.array(bbox2))

    return compute_iou(bbox1[0], bbox2)