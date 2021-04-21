


label_gt='/home/zhaoliu/carbrand-master/carbrand-master/dict/all_cls_2_label_dict'

gt_str = '/home/zhaoliu/carbrand-master/carbrand-master/alldata_new1/cls_2_label'

str_mid = '/home/zhaoliu/carbrand-master/carbrand-master/alldata_new1/mid_2_label'
mid_mgt = '/home/zhaoliu/carbrand-master/carbrand-master/dict/all_mid_2_label_dict'


label_2_gt = {}
gt_2_str ={}
str_2_mid = {}
mid_2_mgt ={}

def get_maps():

    for line in open(label_gt):
        index,index1 = line.strip().split()
        label_2_gt[index] = index1

    for line in open(gt_str):
        index,index1 = line.strip().split()
        gt_2_str[index1] = index

    for line in open(str_mid):
        index,index1 = line.strip().split()
        str_2_mid[index] = index1

    for line in open(mid_mgt):
        index,index1 = line.strip().split()
        mid_2_mgt[index1] = index

    return label_2_gt,gt_2_str,str_2_mid,mid_2_mgt


