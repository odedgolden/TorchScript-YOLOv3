import torch
from torch import nn
from src.YOLOModule import YOLOModule


# All code should be compatible with the supported types mentioned here:
# https://pytorch.org/docs/master/jit_unsupported.html

class YOLOLoss(YOLOModule):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # For Bounding Box Prediction
        self.bce_loss = nn.BCELoss()  # For Class Prediction

    def forward(self, predictions, ground_truth):
        self.metrics, total_loss = self.calculate_metrics(pred_boxes=pred_boxes, pred_cls=pred_cls, targets=y, x=x_0,
                                                          y=y_0, w=w, h=h, pred_conf=pred_conf)

        return

    def build_targets(self, pred_boxes, pred_cls, target):
        """

        :param pred_boxes:
        :param pred_cls:
        :param target:
        :param anchors:
        :param ignore_thres:
        :return:
        """

        ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]
        # Get anchors with best iou
        ious = torch.stack([self.bbox_wh_iou(anchor, gwh) for anchor in self.scaled_anchors])
        best_ious, best_n = ious.max(0)
        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()
        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > self.threshold, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / self.scaled_anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / self.scaled_anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = self.bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask.float()
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


    def calculate_metrics(self, pred_boxes, pred_cls, targets, x, y, w, h, pred_conf):

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
        )

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        return {
            "loss": self.to_cpu(total_loss).item(),
            "x": self.to_cpu(loss_x),
            "y": self.to_cpu(loss_y),
            "w": self.to_cpu(loss_w),
            "h": self.to_cpu(loss_h),
            "conf": self.to_cpu(loss_conf),
            "cls": self.to_cpu(loss_cls),
            "cls_acc": self.to_cpu(cls_acc),
            "recall50": self.to_cpu(recall50),
            "recall75": self.to_cpu(recall75),
            "precision": self.to_cpu(precision),
            "conf_obj": self.to_cpu(conf_obj),
            "conf_noobj": self.to_cpu(conf_noobj),
            "grid_size": self.grid_size,
        }, total_loss

