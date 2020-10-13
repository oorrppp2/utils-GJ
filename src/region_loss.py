from __future__ import division
import numpy as np
import torch
import math
import torch.nn as nn
from torch.autograd import Variable

from src.utils.kitti_utils import rotated_bbox_iou
from src.utils.utils import to_cpu
import src.config as cnf

def build_targets(pred_boxes, pred_cls, target, anchors, num_anchors, num_classes, nH, nW, ignore_thres):
    nB = target.size(0)
    nA = num_anchors   #5
    nC = num_classes   #3
    obj_mask = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    noobj_mask = torch.ByteTensor(nB, nA, nH, nW).fill_(1)
    class_mask = torch.zeros(nB, nA, nH, nW).float()
    iou_scores = torch.zeros(nB, nA, nH, nW).float()
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW)
    tw         = torch.zeros(nB, nA, nH, nW)
    tl         = torch.zeros(nB, nA, nH, nW)
    tim        = torch.zeros(nB, nA, nH, nW)
    tre        = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls       = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)

    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b][t].sum() == 0:
                continue

            # Convert to position relative to box
            gx = target[b, t, 1] * nW
            gy = target[b, t, 2] * nH
            gw = target[b, t, 3] * nW
            gl = target[b, t, 4] * nH
            gim = target[b, t, 5]
            gre = target[b, t, 6]
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box (centered at (100,100))
            gt_box = torch.FloatTensor([100, 100, gw, gl, gim, gre]).unsqueeze(0)
            # Get shape of anchor boxes (centered at (100,100))
            anchor_shapes = torch.ones((len(cnf.anchors), 6)).float() * 100
            anchor_shapes[:, 2:] = torch.FloatTensor(cnf.anchors)
            # Calculate iou between gt and anchor shapes
            anch_ious = torch.from_numpy(rotated_bbox_iou(gt_box, anchor_shapes))
            # Find the best matching anchor box
            best_n = torch.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor([gx, gy, gw, gl, gim, gre]).unsqueeze(0)
            # Get class of target box
            target_label = int(target[b, t, 0])
            # Get the ious of prediction at each anchor
            pred_ious = torch.from_numpy(rotated_bbox_iou(gt_box, pred_boxes[b, :, gj, gi]))
            # Get label correctness
            predicted_labels = (torch.argmax(pred_cls[b, :, gj, gi], -1) == target_label).float()
            # Masks
            obj_mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            tl[b, best_n, gj, gi] = math.log(gl / anchors[best_n][1] + 1e-16)
            # Im and Re
            tim[b, best_n, gj, gi] = gim
            tre[b, best_n, gj, gi] = gre
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1
            # Class Mask
            class_mask[b, best_n, gj, gi] = predicted_labels[best_n]
            # iou between ground truth and best matching prediction
            iou_scores[b, best_n, gj, gi] = pred_ious[best_n]
            # Where the overlap is larger than threshold set mask to zero (ignore)
            noobj_mask[b, anch_ious > ignore_thres, gj, gi] = 0

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, tl, tim, tre, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=3, num_anchors=5):
        super(RegionLoss, self).__init__()
        self.anchors = cnf.anchors
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bbox_attrs = 7+num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        #self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, output, targets):

        # output : batch_size*num_anchorsx(6+1+num_classes)*H*W    [batch_size,50,16,32]
        # targets : targets define in kitti_utils.py(build_yolo_target) get_target function [12,50,7]

        nA = self.num_anchors  # num_anchors = 5
        nB = output.size(0)  # batch_size
        nH = output.size(2)  # nH  16
        nW = output.size(3)  # nW  32

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if output.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if output.is_cuda else torch.ByteTensor

        # prediction [12,5,16,32,15]
        prediction = output.view(nB, nA, self.bbox_attrs, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        im = prediction[..., 4] # Im
        re = prediction[..., 5] # Re
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w, a_h, im, re) for a_w, a_h, im, re in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :6].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        pred_boxes[..., 4] = im.data
        pred_boxes[..., 5] = re.data

        if output.is_cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            #self.ce_loss = self.ce_loss.cuda()

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tconf, tcls = build_targets(
            pred_boxes = to_cpu(pred_boxes),
            pred_cls = to_cpu(pred_cls),
            target = to_cpu(targets),
            anchors = to_cpu(scaled_anchors),
            num_anchors = nA,
            num_classes = self.num_classes,
            nH = nH,
            nW = nW,
            ignore_thres = self.ignore_thres
        )

        # Handle target variables
        tx = tx.type(FloatTensor)
        ty = ty.type(FloatTensor)
        tw = tw.type(FloatTensor)
        th = th.type(FloatTensor)
        tim = tim.type(FloatTensor)
        tre = tre.type(FloatTensor)
        tconf = tconf.type(FloatTensor)
        tcls = tcls.type(FloatTensor)

        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_im = self.mse_loss(im[obj_mask], tim[obj_mask])
        loss_re = self.mse_loss(re[obj_mask], tre[obj_mask])
        loss_Euler = loss_im + loss_re
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss = loss_x + loss_y + loss_w + loss_h + loss_Euler + loss_conf + loss_cls

        # Matrices
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_cls[obj_mask].mean()
        conf_noobj = pred_cls[noobj_mask].mean()
        class_mask = class_mask.type(FloatTensor)
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).type(FloatTensor)
        iou70 = (iou_scores > 0.7).type(FloatTensor)
        obj_mask = obj_mask.type(FloatTensor)
        detected_mask = conf50 * class_mask * obj_mask
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall70 = torch.sum(iou70 * detected_mask) / (obj_mask.sum() + 1e-16)

        return (
            loss,
            {
                "loss": to_cpu(loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "im": to_cpu(loss_im).item(),
                "re": to_cpu(loss_re).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall70": to_cpu(recall70).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item()
            }
        )