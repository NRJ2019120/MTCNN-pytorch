"""生成更精准样本框"""
label_file = r"/home/tensorflow01/oneday/celeba/Anno/list_bbox_celeba.txt"
self_label_file = r"/home/tensorflow01/oneday/celeba/little_bbox_celeba.txt"


file = open(self_label_file,"w")

for i,line in enumerate(open(label_file).readlines()):

    if i == 0:
        file.write("202599\n")
    if i == 1:
        file.write("image_id x1 y1 width height\n")
    if i > 1 and i < 202599+2:
        strs = line.split()
        # print(strs)
        print(line)
        filename = strs[0]
        x1 = int(strs[1])
        y1 = int(strs[2])
        w = int(strs[3])
        h = int(strs[4])

        # _x1 = int(x1 + w*0.08)
        # _y1 = int(y1 + h*0.08)
        # _w = int(w * 0.85)
        _h = int(h * 0.93)

        file.write(filename + "  {0}  {1}  {2}  {3}\n".format(x1,y1,w,_h))
        # im.save("{0}/{1}.jpg".format(self_img_path, i-1))
        # print(im.size)
        # imDraw = ImageDraw.Draw(im)
        # imDraw.rectangle((_x1,_y1,_x1+_w,_y1+_h),outline="red")
        # imDraw.rectangle((x1,y1,x1+w,y1+h),outline="green")
        # im.show()
file.close()