import numpy as np

def iou(box, boxes, isMin = False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])  # Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

def nms(boxes, thresh=0.3, isMin = False):

    if boxes.shape[0] == 0:    #确定有框
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]  #按置信度大小排序 boxes[:,4]为置信度,按照置信度排序
    r_boxes = []                    #做完nms后保留下的框

    while _boxes.shape[0] > 1:  #有重叠框即至少有两个框,   .shape[0]0维轴上的个数
        a_box = _boxes[0]      #置信度最大的框
        b_boxes = _boxes[1:]   #切片 剩余的所有框

        r_boxes.append(a_box)

        index = np.where(iou(a_box, b_boxes,isMin) < thresh)  #做iou,NMS 判断  #重叠度iou<0.3 的框保留
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:   #只有一个框
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)   #框堆叠,stack()增加维度


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side
    return square_bbox

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
#
# if __name__ == '__main__':
#     # a = np.array([1,1,11,11])
#     # bs = np.array([[1,1,10,10],[11,11,20,20]])
#     # print(iou(a,bs))
#
#     bs = np.array([[1, 1, 10, 10, 40], [1, 1, 9, 9, 10], [9, 8, 13, 20, 15], [6, 11, 18, 17, 13]])
#     # print(bs[:,3].argsort())
#     print(nms(bs))