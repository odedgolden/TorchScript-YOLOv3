import torch
import torch.nn as nn


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, image_size=416):
        """
        YOLO Layer - consistent with the darknet code at: https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c
                     and with the PyTorch-YOLOv3 code at: https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py

        :param anchors: A list of the masked anchors with the format: [(x_i,y_i),...]
        :param num_classes: Maximum number of classes that will be predicted from one grid cell
        :param image_size: The expected image size in pixels, input_size == image width == image height
        """

        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # Useful video explaining this: https://youtu.be/RTlwl2bv0Tg (thanks Andrew!)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.threshold = 0.5
        self.mse_loss = nn.MSELoss()  # For Bounding Box Prediction
        self.bce_loss = nn.BCELoss()  # For Class Prediction
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}
        self.image_size = image_size
        self.grid_size = image_size  # The grid should cover the image exactly
        self.stride = 1.0

    def forward(self, x, y=None):
        """
        The forward function has the following steps: 1. Extract predictions from previous layer.
                                                      2. Create prediction bounding boxes.
                                                      3. Compare to ground truth bounding boxes, update metrics and loss

        :param x: The input, with shape: (batch_size, channels_size, image_size, image_size)
        :param y: The ground truth, with shape: (batch_size, predicted_boxes, predicted_confidence, predicted_class)
        :return: The YOLO output - (batch_size, predicted_boxes, predicted_confidence, predicted_class), layer_loss
        """

        batch_size = x.size(0)
        grid_size = x.size(2)  # that is - the image size

        # In order to get the predictions we only reorganizing the input tensor, replacing the class prediction to be
        # the last dimension
        prediction = (
            # Reshape tensor after convolution layers
            x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)  # Class prediction should be the last dimension
                .contiguous()  # Make the actual changes inplace
        )

        # Extract predictions for bounding boxes, confidence and class
        # We use sigmoid in order to make sure the values are between (0,1)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.update_grid(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape)
        if x.is_cuda:
            pred_boxes.cuda()
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # For output we only concatenating
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4) * self.stride,
                pred_conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )

        if y is None:
            total_loss = 0
        else:
            self.metrics, total_loss = self.calculate_metrics()

        return output, total_loss

    # Helper Functions

    @staticmethod
    def to_cpu(tensor):
        """

        :param tensor: Tensor with a single number
        :return: The number in the tensor
        """
        return tensor.detach().cpu().item()

    def update_grid(self, new_grid_size, cuda=True):
        """


        :param new_grid_size: New grid size derived from the actual input size - i.e. new_grid_size=416
        :param cuda: Is Cuda available
        :return:
        """
        self.grid_size = int(new_grid_size.item())  # Update grid size
        self.stride = self.image_size / self.grid_size  # Calculate stride

        # Calculate offsets for each grid
        self.grid_x = torch.arange(self.grid_size).repeat(self.grid_size, 1).view([1, 1, self.grid_size, self.grid_size]).type(torch.FloatTensor)
        self.grid_y = torch.arange(self.grid_size).repeat(self.grid_size, 1).t().view([1, 1, self.grid_size, self.grid_size]).type(torch.FloatTensor)
        self.scaled_anchors = torch.FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        if cuda:
            self.scaled_anchors.cuda()
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

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

    @staticmethod
    def box_iou(box1_tensor, box2_tensor):
        """
        Calculating Intersection over Union of the two bounding boxes.
        Useful video explaining the idea: https://youtu.be/ANIzQ5G-XPE (thanks again Andrew)
        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Tensor of intersection boxes
        """
        intersection = YOLOLayer.box_intersection(box1_tensor, box2_tensor)
        union = YOLOLayer.box_union(box1_tensor, box2_tensor)
        return intersection/union

    @staticmethod
    def box_union(box1_tensor, box2_tensor):
        """

        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Union tensor of the two boxes tensors
        """

        intersection = YOLOLayer.box_union(box1_tensor, box2_tensor)
        union = box1_tensor[:, 0]*box1_tensor[:, 3] + box2_tensor[:, 0]*box2_tensor[:, 3] - intersection
        return union

    @staticmethod
    def box_intersection(box1_tensor, box2_tensor):
        """

        :param box1_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :param box2_tensor: Bounding box tensor with shape: (batch_size, 4), where the 4 are (x, y, w, h)
        :return: Intersection tensor of the two boxes tensors
        """
        w = YOLOLayer.one_dim_overlap(box1_tensor[:, 0], box1_tensor[:, 2], box1_tensor[:, 1], box1_tensor[:, 3])
        h = YOLOLayer.one_dim_overlap(box2_tensor[:, 0], box2_tensor[:, 2], box2_tensor[:, 1], box2_tensor[:, 3])
        if w < 0 or h < 0:
            area = 0
        else:
            area = w*h
        return area

    @staticmethod
    def one_dim_overlap(x1_tensor, w1_tensor, x2_tensor, w2_tensor):
        """

        :param x1_tensor:
        :param w1_tensor:
        :param x2_tensor:
        :param w2_tensor:
        :return:
        """
        right = torch.min(x1_tensor + w1_tensor/2, x2_tensor + w2_tensor/2)
        left = torch.max(x1_tensor - w1_tensor/2, x2_tensor - w2_tensor/2)
        return right - left
