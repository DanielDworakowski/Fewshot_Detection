import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *
from cfg import cfg
from numbers import Number
from random import random
import pdb
import debug as db
import Algo

def neg_filter(pred_boxes, target, withids=False):
    assert pred_boxes.size(0) == target.size(0)
    if cfg.neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(cfg.neg_ratio, Number):
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        ratio = cfg.neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            inds = np.argwhere(flags).squeeze()
            pred_boxes, target = pred_boxes[inds], target[inds]
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target

def genConfMask(conf_mask, pred_boxes, target, nAnchors, nB, nW, nH, sil_thresh):
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(cfg.max_boxes):
            if target[b, t*5+1] == 0:
                break
            cur_gt_boxes = target[b, t*5+1:t*5+5].float() * torch.tensor([nW, nH, nW, nH], dtype=torch.float)
            cur_gt_boxes = cur_gt_boxes.view(4, 1).expand(-1, nAnchors)
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b, (cur_ious>sil_thresh).view(conf_mask.shape[1], conf_mask.shape[2], conf_mask.shape[3])] = 0
    return conf_mask

#
def genConfMask2(conf_mask, pred_boxes, target, nAnchors, nB, nW, nH, sil_thresh):
    pred_boxes = pred_boxes.view(nB, nAnchors, 4)
    target = target.view(nB, cfg.max_boxes, 5)
    target = target[..., 1:4] * torch.tensor([nW, nH, nW, nH])
    pred_boxes_bltr = Algo.bb_cwh2bltr(pred_boxes)
    target_bltr = Algo.bb_cwh2bltr(target)
    iou = Algo.jaccard(pred_boxes, target.float())
    max_iou = iou.max(-1)[0]
    # conf_mask = max_iou.view(conf_mask.shape) <= sil_thresh
    conf_mask[max_iou.view(conf_mask.shape) > sil_thresh] = 0
    return conf_mask
    # db.printTensor(pred_boxes)
    # db.printTensor(target)
    # ious = bbox_ious(pred_boxes, target, x1y1x2y2=False)
    # db.printInfo(ious)

def bestAnchor(anchors, gw, gh, anchor_step, nA):
    best_iou = 0.0
    best_n = -1
    min_dist = float('inf')
    gt_box = [0, 0, gw, gh]
    for n in range(nA):
        aw = anchors[anchor_step*n]
        ah = anchors[anchor_step*n+1]
        anchor_box = [0, 0, aw, ah]
        iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
        if anchor_step == 4:
            ax = anchors[anchor_step*n+2]
            ay = anchors[anchor_step*n+3]
            dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
        if iou > best_iou:
            best_iou = iou
            best_n = n
        elif anchor_step==4 and iou == best_iou and dist < min_dist:
            best_iou = iou
            best_n = n
            min_dist = dist
    return best_n

def bestAnchor2(anchors, gw, gh, anchor_step, nA):
    assert anchor_step == 2, 'only support anchor_step 2'
    best_iou = 0.0
    best_n = -1
    min_dist = float('inf')
    anchors = torch.tensor(anchors)
    gt_box = torch.tensor([gw, gh], dtype=torch.float)
    anchors = anchors.view(-1, anchor_step)
    # db.printInfo(gt_box)
    # db.printInfo(anchors)
    min_wh = torch.min(gt_box, anchors)
    I = min_wh.prod(-1)
    U = gt_box.prod(-1) + anchors.prod(-1) - I
    IoU = I / U
    best_n = IoU.max(-1)[1]
    return best_n


@torch.no_grad()
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    t0 = time.time()

    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors) // num_anchors
    # print('anchor_step: ', anchor_step)
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW)
    tw         = torch.zeros(nB, nA, nH, nW)
    th         = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW)
    t1 = time.time()

    nAnchors = nA*nH*nW
    nPixels  = nH*nW

    conf_mask = genConfMask(conf_mask, pred_boxes, target, nAnchors, nB, nW, nH, sil_thresh)
    t2 = time.time()

    # conf_mask2 = genConfMask2(conf_mask, pred_boxes, target, nAnchors, nB, nW, nH, sil_thresh)
    # db.printInfo((conf_mask - conf_mask2).sum())

    if seen < 12800:
        if anchor_step == 4:
            tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
            ty = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    t3 = time.time()

    tA = 0
    tConv = 0

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        # pdb.set_trace()
        #
        # While it might not be faster to do the full loop, can speed up within the batch.
        for t in range(50):
            if target[b, t*5+1] == 0:
                break
            nGT = nGT + 1
            # best_iou = 0.0
            # best_n = -1
            # min_dist = 10000
            gx = target[b, t*5+1] * nW
            gy = target[b, t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b, t*5+3]*nW
            gh = target[b, t*5+4]*nH
            ta0 = time.time()
            # best_n = bestAnchor(anchors, gw, gh,  anchor_step, nA)
            best_n = bestAnchor2(anchors, gw, gh, anchor_step, nA)
            # db.printInfo(best_n- best_n2)
            ta1 = time.time()
            tA += ta1 - ta0


            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

            coord_mask[b, best_n, gj, gi] = 1
            cls_mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = object_scale
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj

            tw[b, best_n, gj, gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b, best_n, gj, gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b, best_n, gj, gi] = iou
            tcls[b, best_n, gj, gi] = target[b, t*5]
            tConv += (time.time() - ta1)
            if iou > 0.5:
                nCorrect = nCorrect + 1
    t4 = time.time()
    if False:
        print('-----------------------------------')
        print('        allocate   : %f' % (t1 - t0))
        print('         conf mask : %f' % (t2 - t1))
        print('         mod alloc : %f' % (t3 - t2))
        print('     build targets : %f' % (t4 - t3))
        print('     anchor total  : %f' % (tA ))
        print('       conv total  : %f' % (tConv ))
        print('             total : %f' % (t4 - t0))
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class RegionLossV2(nn.Module):
    """
    Yolo region loss + Softmax classification across meta-inputs
    """
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLossV2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        assert len(anchors)//num_anchors == len(anchors)/num_anchors, 'Modified here to not be a float'
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.mse = nn.MSELoss(reduction='sum')
        print('class_scale', self.class_scale)

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        # Get all classification prediction
        # pdb.set_trace()
        bs = target.size(0)
        cs = target.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls = output.view(output.size(0), nA, (5+nC), nH, nW)
        cls = cls.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda())).squeeze()
        cls = cls.view(bs, cs, nA*nC*nH*nW).transpose(1,2).contiguous().view(bs*nA*nC*nH*nW, cs)

        # Rearrange target and perform filtering operation
        target = target.view(-1, target.size(-1))
        # bef = target.size(0)
        output, target, inds = neg_filter(output, target, withids=True)
        counts, _ = np.histogram(inds, bins=bs, range=(0, bs*cs))
        # print("{}/{}".format(target.size(0), bef))

        t0 = time.time()
        nB = output.data.size(0)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = (output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW)).sigmoid()
        y    = (output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW)).sigmoid()
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = (output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)).sigmoid()
        # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
        # cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        # cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()
        with torch.no_grad():
            pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
            grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
            grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()

            anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
            anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
            anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
            anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

            pred_boxes[0] = x.data.view(-1) + grid_x
            pred_boxes[1] = y.data.view(-1) + grid_y
            pred_boxes[2] = torch.exp(w.data).view(-1) * anchor_w
            pred_boxes[3] = torch.exp(h.data).view(-1) * anchor_h
            pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
            t2 = time.time()

            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                                nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
            # Take care of class mask
            cls_num = torch.sum(cls_mask)
            idx_start = 0
            cls_mask_list = []
            tcls_list = []
            for i in range(len(counts)):
                if counts[i] == 0:
                    cur_mask = torch.zeros(nA, nH, nW)
                    cur_tcls = torch.zeros(nA, nH, nW)
                else:
                    cur_mask = torch.sum(cls_mask[idx_start:idx_start+counts[i]], dim=0)
                    cur_tcls = torch.sum(tcls[idx_start:idx_start+counts[i]], dim=0)
                cls_mask_list.append(cur_mask)
                tcls_list.append(cur_tcls)
                idx_start += counts[i]
            cls_mask = torch.stack(cls_mask_list)
            tcls = torch.stack(tcls_list)

            cls_mask = (cls_mask == 1)
            nProposals = int((conf > 0.25).float().sum().item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())


        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        # cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())
        cls        = cls[Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())].view(-1, cs)
        #
        # test this
        tcls = Variable(tcls.view(-1)[cls_mask.view(-1)].long().cuda())
        ClassificationLoss = nn.CrossEntropyLoss(reduction='sum')

        t3 = time.time()


        assert self.coord_scale == 1, 'Must have a 1 coord_scale'
        assert self.class_scale == 1, 'Must have a 1 class_scale'
        loss_x = self.mse(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.mse(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.mse(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.mse(h*coord_mask, th*coord_mask)/2.0
        loss_conf = self.mse(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = ClassificationLoss(cls, tcls)

        # # pdb.set_trace()
        # ids = [9,11,12,16]
        # new_cls, new_tcls = select_classes(cls, tcls, ids)
        # new_tcls = Variable(torch.from_numpy(new_tcls).long().cuda())
        # loss_cls_new = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(new_cls, new_tcls)
        # loss_cls_new *= 10
        # loss_cls += loss_cls_new

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        # print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        return loss


def select_classes(pred, tgt, ids):
    # convert tgt to numpy
    tgt = tgt.cpu().data.numpy()
    new_tgt = [(tgt == d) * i  for i, d in enumerate(ids)]
    new_tgt = np.max(np.stack(new_tgt), axis=0)
    idxes = np.argwhere(new_tgt > 0).squeeze()
    new_pred = pred[idxes]
    new_pred = new_pred[:, ids]
    new_tgt = new_tgt[idxes]
    return new_pred, new_tgt

# class RegionLoss(nn.Module):
#     def __init__(self, num_classes=0, anchors=[], num_anchors=1):
#         super(RegionLoss, self).__init__()
#         self.num_classes = num_classes
#         self.anchors = anchors
#         self.num_anchors = num_anchors
#         self.anchor_step = len(anchors)//num_anchors

#         assert len(anchors)//num_anchors == len(anchors)/num_anchors, 'Modified here to not be a float'
#         self.coord_scale = 1
#         self.noobject_scale = 1
#         self.object_scale = 5
#         self.class_scale = 1
#         self.thresh = 0.6
#         self.seen = 0

#     def forward(self, output, target):
#         # import pdb; pdb.set_trace()
#         #output : BxAs*(4+1+num_classes)*H*W

#         # if target.dim() == 3:
#         #     # target : B * n_cls * l
#         #     l = target.size(-1)
#         #     target = target.permute(1,0,2).contiguous().view(-1, l)
#         if target.dim() == 3:
#             target = target.view(-1, target.size(-1))
#         bef = target.size(0)
#         output, target = neg_filter(output, target)
#         # print("{}/{}".format(target.size(0), bef))

#         t0 = time.time()
#         nB = output.data.size(0)
#         nA = self.num_anchors
#         nC = self.num_classes
#         nH = output.data.size(2)
#         nW = output.data.size(3)

#         output   = output.view(nB, nA, (5+nC), nH, nW)
#         x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
#         y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
#         w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
#         h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
#         conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
#         # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
#         cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
#         cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

#         t1 = time.time()

#         pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
#         grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
#         grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
#         anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
#         anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
#         anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
#         anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
#         pred_boxes[0] = x.data + grid_x
#         pred_boxes[1] = y.data + grid_y
#         pred_boxes[2] = torch.exp(w.data) * anchor_w
#         pred_boxes[3] = torch.exp(h.data) * anchor_h
#         pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
#         t2 = time.time()

#         nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
#                                                                nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
#         cls_mask = (cls_mask == 1)
#         if cfg.metayolo:
#             tcls.zero_()
#         nProposals = int((conf > 0.25).float().sum().data[0])

#         tx    = Variable(tx.cuda())
#         ty    = Variable(ty.cuda())
#         tw    = Variable(tw.cuda())
#         th    = Variable(th.cuda())
#         tconf = Variable(tconf.cuda())
#         tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

#         coord_mask = Variable(coord_mask.cuda())
#         conf_mask  = Variable(conf_mask.cuda().sqrt())
#         cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
#         cls        = cls[cls_mask].view(-1, nC)

#         t3 = time.time()

#         loss_x = self.coord_scale * nn.MSELoss(reduction='sum')(x*coord_mask, tx*coord_mask)/2.0
#         loss_y = self.coord_scale * nn.MSELoss(reduction='sum')(y*coord_mask, ty*coord_mask)/2.0
#         loss_w = self.coord_scale * nn.MSELoss(reduction='sum')(w*coord_mask, tw*coord_mask)/2.0
#         loss_h = self.coord_scale * nn.MSELoss(reduction='sum')(h*coord_mask, th*coord_mask)/2.0
#         loss_conf = nn.MSELoss(reduction='sum')(conf*conf_mask, tconf*conf_mask)/2.0
#         loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)
#         loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
#         t4 = time.time()
#         if False:
#             print('-----------------------------------')
#             print('        activation : %f' % (t1 - t0))
#             print(' create pred_boxes : %f' % (t2 - t1))
#             print('     build targets : %f' % (t3 - t2))
#             print('       create loss : %f' % (t4 - t3))
#             print('             total : %f' % (t4 - t0))
#         print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
#         return loss
