import numpy as np
#box = [X1,Y1,X2,Y2,C][box 坐标,置信度]

def iou(box,boxes,ismin = False):        #iou重叠度计算

    top_x = np.maximum(box[0],boxes[:,0]) #涉及广播规则,切片
    top_y = np.maximum(box[1],boxes[:,1])
    bottom_x = np.minimum(box[2],boxes[:,2])
    bottom_y = np.minimum(box[3],boxes[:,3])

    w = np.maximum(0,(bottom_x-top_x))
    h = np.maximum(0,(bottom_y-top_y)) #注意np.maximum()与np.max()的区别_
                                       #直观见test.  w,h 任何一个为零,则不想交,交集为零

    j_area = w *h #交集面积至少为零,无相交情况

    box_area = (box[2]-box[0])*(box[3]-box[1])                  #框面积
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])#框面积 张量

    if ismin == False:
        fm_area = box_area + boxes_area - j_area            #两框并集面积张量
        return j_area/fm_area                                #交集除并集 iou     包含关系太小属于负样本
    else:
        fm_area = np.minimum(box_area,boxes_area[:])
        return j_area/fm_area                                #交集除最小值 iou    包含关系太小属于正样本

def rect2squar(boxes):            #boxes为一图多个人脸多个框情况,boxes为二维张量,把框变成正方形框
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    side_len = np.maximum(w, h)
    cx = boxes[:, 0] + w / 2      #中心点x 坐标
    cy = boxes[:, 1] + h / 2      #中心点y 坐标

    x1 = cx - side_len / 2
    y1 = cy - side_len / 2
    x2 = cx + side_len / 2
    y2 = cy + side_len / 2
    return np.stack([x1, y1, x2, y2,boxes[:,4]], axis=1) #注意此时是轴为1上的拼接
    # np.concatenate([x1,y1,x2,y2],axis=0)的区别



