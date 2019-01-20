import os

sample_path_12 = r'/media/tensorflow01/myfile/landmarks_img_celeba/12'
sample_path_24 = r'/media/tensorflow01/myfile/landmarks_img_celeba/24'
sample_path_48 = r'/media/tensorflow01/myfile/landmarks_img_celeba/48'


def fun(sample_path):

    p_file = open(os.path.join(sample_path, "nonlandmarks_positive.txt"), "w")
    n_file = open(os.path.join(sample_path, "nonlandmarks_negative.txt"), "w")
    part_file = open(os.path.join(sample_path, "nonlandmarks_part.txt"), "w")

    for i,line in enumerate(open(os.path.join(sample_path, "positive.txt"),"r").readlines()):
        strs = line.strip().split()
        # print(strs)
        filename = strs[0]
        cls = strs[1]
        off1 = strs[2]
        off2 = strs[3]
        off3 = strs[4]
        off4 = strs[5]
        p_file.write("{0}  {1}  {2}  {3}  {4}  {5}\n".format(filename,cls,off1,off2,off3,off4))
    for i, line in enumerate(open(os.path.join(sample_path, "negative.txt"), "r").readlines()):
        strs = line.strip().split()
        # print(strs)
        filename = strs[0]
        cls = strs[1]
        off1 = strs[2]
        off2 = strs[3]
        off3 = strs[4]
        off4 = strs[5]
        n_file.write("{0}  {1}  {2}  {3}  {4}  {5}\n".format(filename, cls, off1, off2, off3, off4))
    for i, line in enumerate(open(os.path.join(sample_path, "part.txt"), "r").readlines()):
        strs = line.strip().split()
        # print(strs)
        filename = strs[0]
        cls = strs[1]
        off1 = strs[2]
        off2 = strs[3]
        off3 = strs[4]
        off4 = strs[5]
        part_file.write("{0}  {1}  {2}  {3}  {4}  {5}\n".format(filename, cls, off1, off2, off3, off4))

#
# if __name__ == '__main__':
#     fun(sample_path_48)
#     fun(sample_path_24)
#     fun(sample_path_12)

