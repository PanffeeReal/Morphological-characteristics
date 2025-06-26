import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.flow import cost_of_flow

from utils.metrics import bbox_iou


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        ## lpf add: consider pos_overlaps(scale,shape,area,distance) and cls & location discrepancy --> weight
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)

        ### lpf : target_labels are the original gt-labels : no use later ;target_scores are one hot labels from target_scores
        target_scores = target_scores * norm_align_metric    ### lpf : norm(align_metric) * IoU --> cls branch's targets & regress branch's weights

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        # get anchor_align metric, (b, max_num_obj, h*w)
        # align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # lpf step1 : improved metric cost
        align_metric, overlaps = self.get_box_metrics_lpf(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)

        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (b, max_num_obj, h*w)

        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        #lpf step2 : auto select metric mask, (b, max_num_obj, h*w)
        # mask_topk = self.select_auto_candidates(align_metric * mask_in_gts,
        #                                         topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())

        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps


    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        gt_labels = gt_labels.to(torch.long)  # b, max_num_obj, 1
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_box_metrics_lpf(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        '''
        '''
        gt_labels = gt_labels.to(torch.long)  # b, max_num_obj, 1
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        #pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        #lpf add: improve cost
        cost_impr=self.cost_distance(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False).squeeze(3).clamp(0)

        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(
            0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta) * cost_impr
        return align_metric, overlaps

    def cost_distance(self, box1, box2, xywh=True,eps=1e-7):
        '''
        #distance cost
        gt_w/gt_h=para1
        sqrt(distence_w(pra1)**2+distence_h(pra1)**2)<--> cost_distance

        #shape cost
        pred_bbox_w/pred_bbox_h=para2
        (para1-para2)/max(para1,para2) or (para1/para2) = costs_shape

        #area cost
        (area_pred-area_gt)/max(area_pred,area_gt) or area_pred/area_gt=cost_area

        Args:
            box2:
            xywh:
            eps:

        Returns:

        '''

        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  #bs, max_num_obj, 1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps  #bs, 1 , num_total_anchors

        #width / height: *********************************************************
        gt_wh_ratio = torch.min(w1,h1)/torch.max(w1,h1) # bs, max_num_obj
        pd_wh_ratio = torch.min(w2,h2)/torch.max(w2,h2) # bs, num_total_anchors
        #cost_shape = torch.sqrt(1-(torch.abs(gt_wh_ratio-pd_wh_ratio)/torch.max(gt_wh_ratio,pd_wh_ratio))**2)  # bs, max_num_obj, num_total_anchors
        #cost_shape = (1-(torch.abs(gt_wh_ratio-pd_wh_ratio)/torch.max(gt_wh_ratio,pd_wh_ratio))**3)**1/3
        cost_shape = 1-torch.abs(gt_wh_ratio-pd_wh_ratio)/torch.max(gt_wh_ratio,pd_wh_ratio)  # linear
        norm_cost_shape=cost_shape/(cost_shape.amax(-2).unsqueeze(2)+eps)
        #cost_shape = torch.min(gt_wh_ratio,pd_wh_ratio)/torch.max(gt_wh_ratio,pd_wh_ratio) # simple linear

        #************************************************************************
        # distance cost w.r.t gt_box ： use centerness encoding
        # gt_cx=(b1_x2 + b1_x1)/2   #bs, max_num_obj, 1
        # gt_cy=(b1_y2 + b1_y1)/2
        # pd_cx=(b2_x2 + b2_x1)/2   #bs, 1 , num_total_anchors
        # pd_cy=(b2_y2 + b2_y1)/2
        #
        # d_l=pd_cx-b1_x1
        # d_r=b1_x2-pd_cx
        # d_t=pd_cy-b1_y1
        # d_b=b1_y2-pd_cy
        # cost_distance=torch.min(d_l,d_r)*torch.min(d_t,d_b)/(torch.max(d_l,d_r)*torch.max(d_t,d_b)+self.eps)
        # norm_cost_distance=cost_distance/(cost_distance.amax(-2).unsqueeze(2)+eps)
        # distance_cx=torch.abs(gt_cx-pd_cx)/(w1/2)
        # distance_cy=torch.abs(gt_cy-pd_cy)/(h1/2)
        # distance_cx=(torch.abs(gt_cx-pd_cx) * (w1/h1 <= 1)  + torch.abs(gt_cx-pd_cx) * (w1/h1 > 1)) / (h1/2) # norm by short axis
        # distance_cy=(torch.abs(gt_cy-pd_cy) * (w1/h1 > 1) / (h1/2) +torch.abs(gt_cy-pd_cy) * gt_wh_ratio * (w1/h1 <= 1)) / (w1/2)
        # cost_distance=torch.ones(self.bs, self.n_max_boxes,box2.size(2))
        # cost_distance=cost_distance * (1-torch.sqrt(distance_cx**2+distance_cy**2)).clamp(0)  # bs, max_num_obj, num_total_anchors linear function
        # # cost_distance = (1 - torch.sqrt(1-((distance_cx ** 2 + distance_cy ** 2)-1)**2)).clamp(0)


        #area *********************************************************************
        gt_area=w1 * h1
        pd_area=w2 * h2
        # cost_area=torch.min(gt_area,pd_area)/torch.max(gt_area,pd_area) #  linear
        # cost_area=1-torch.sqrt(1-cost_area**2)  #  circle
        #cost_area=cost_area**2  #  **2
        cost_area=1-torch.abs(gt_area-pd_area)/torch.max(gt_area,pd_area) # linear diff
        norm_cost_area = cost_area / (cost_area.amax(-2).unsqueeze(2) + eps)

        #cost wh_ratio_diff  final: **2    *******************************************************
        wh_ratio_diff=torch.min(w1/h1,w2/h2)/torch.max(w1/h1,w2/h2) # linear
        cost_wh_ratio=wh_ratio_diff#**2 # wh_ratio_diff**3   [bs,num_gt,num_anchor,1]
        norm_cost_wh_ratio = cost_wh_ratio / (cost_wh_ratio.amax(-2).unsqueeze(2) + eps)

        # cost_return = norm_cost_shape * norm_cost_area * norm_cost_wh_ratio #* cost_distance # norm once
        # cost_return = norm_cost_shape + norm_cost_area + norm_cost_wh_ratio  # * cost_distance # norm once
        cost_return = (norm_cost_shape + norm_cost_wh_ratio + norm_cost_area )/3 #+ norm_cost_wh_ratio norm_cost_shape
        # cost_return = norm_cost_shape * norm_cost_area * norm_cost_wh_ratio * norm_cost_distance  # norm twice
        #norm
        # max_value,_=torch.max(cost_return,dim=2)
        # # print('lpf',cost_wh_ratio.shape,max_value.unsqueeze(dim=2).shape)
        # cost_return=cost_return/(max_value.unsqueeze(dim=2)+eps)
        # cost_wh_ratio = 1-torch.abs(w1/h1-w2/h2)/torch.max(w1/h1,w2/h2) #linear diff

        return  cost_return # IoU *cost_distance cost_shape * cost_wh_ratio cost_area

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest) ## to do --> dynamic selection
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        # assigned topk should be unique, this is for dealing with empty labels
        # since empty labels will generate index `0` through `F.one_hot`
        # NOTE: but what if the topk_idxs include `0`?
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)
    def select_auto_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: (b, max_num_obj, h*w).
            topk_mask: (b, max_num_obj, topk) or None
        lpf:
        select samples above middle value + var along dim(-1) for each gt
        or analysis score gap to select positives along dim(-1) for each gt
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        # assigned topk should be unique, this is for dealing with empty labels
        # since empty labels will generate index `0` through `F.one_hot`
        # NOTE: but what if the topk_idxs include `0`?
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
