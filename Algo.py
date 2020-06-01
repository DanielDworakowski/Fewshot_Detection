import torch
#
# Convert a cwh formatted bounding box to bltr.
def bb_cwh2bltr(box):
    box[..., :2].sub_(box[..., 2:] / 2.)
    box[..., 2:].add_(box[..., :2])
    return box
#
# Modified from:
# https://github.com/amdegroot/ssd.pytorch/blob/dd85aff4ce1b478f94b493df417d86f2168b7e7d/layers/box_utils.py
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[..., 2:4].unsqueeze(2).expand(-1, A, B, 2),
                        box_b[..., 2:4].unsqueeze(1).expand(-1, A, B, 2))
    min_xy = torch.max(box_a[..., :2].unsqueeze(2).expand(-1, A, B, 2),
                        box_b[..., :2].unsqueeze(1).expand(-1, A, B, 2))
    max_xy.sub_(min_xy).clamp_(min=0)
    max_xy[..., 0].mul_(max_xy[..., 1])
    return max_xy[..., 0]

#
# Jaccard overlap for rectangle between all.
# Modified from:
# https://github.com/amdegroot/ssd.pytorch/blob/dd85aff4ce1b478f94b493df417d86f2168b7e7d/layers/box_utils.py
def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    #
    # Area of a bounding box.
    def bb_area(box):
        return (box[..., 2] - box[..., 0]).mul_(box[..., 3] - box[..., 1])
    inter = intersect(box_a, box_b)
    area_a = bb_area(box_a).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = bb_area(box_b).unsqueeze(1).expand_as(inter)  # [A,B]
    union = (area_a + area_b).sub_(inter)
    return inter.div_(union)  # [A,B]
