import torch
import time
import sys
import json
import torch.distributed as dist
import numpy as np
from utils import AverageMeter, calculate_accuracy,calculate_union_accuracy,Logger

from torchvision.transforms import Compose, ToTensor, Resize, Normalize,ToPILImage
from pathlib import Path
from dataset.infer_dataloader import LmdbDataset_val
from dataset.maps import get_maps
from model.resnet import resnet18
from torch.utils.data._utils.collate import default_collate  # 注意版本


main_2_label_path = '/home/zhaoliu/car_data/alldata_new/all_main_2_label_new'
mid_2_label_path = '/home/zhaoliu/car_data/alldata_new/all_mid_2_label_new'
cls_2_label_path = '/home/zhaoliu/car_data/alldata_new/all_cls_2_label_new'

sub_2_super_path = '/home/zhaoliu/car_data/alldata_new/all_sub_2_super_new'
sub_2_fb_path = '/home/zhaoliu/car_data/alldata_new/all_sub_2_fb_new'


main_dict = {}
mid_dict = {}
cls_list = {}
sub2fb_dict = {}
sub2super_dict = {}
mid = {}

label_main_dict = {}
for line in open(main_2_label_path,'r'):
	main_cls,idx = line.strip().rsplit(' ', 1)
	main_dict[main_cls] = int(idx)    # 建立 主品牌-label（815） 的映射 
	label_main_dict[int(idx)] = main_cls  # 建立 label（815）- 主品牌 的映射 

for line in open(mid_2_label_path,'r'):
	mid_cls, idx = line.strip().rsplit(' ', 1)
	mid_dict[mid_cls] = int(idx)  # 子品牌-label(9186) 

for line in open(mid_2_label_path,'r'):
	mid_cls, idx = line.strip().rsplit(' ', 1)
	mid[str(idx)] = mid_cls  # label(9186) - 子品牌

for line in open(cls_2_label_path,'r'):
	cls, idx = line.strip().rsplit(' ', 1) 
	cls_list[int(idx)] = cls   # # label（27249）- 全品牌 

for line in open(sub_2_super_path,'r'):
	sub_cls, super_cls = line.strip().rsplit(' ', 1)
	sub2super_dict[int(sub_cls)]=int(super_cls)   # 所有品牌27249 - 车型21

for line in open(sub_2_fb_path,'r'):
	sub_cls, fb_cls = line.strip().rsplit(' ', 1)
	sub2fb_dict[int(sub_cls)] = int(fb_cls)     # 所有品牌27249 - 方向3


ori ={'0':'back','1':'front','2':'side'}
chexing = {
    "0": "轿车",
    "1": "面包车",
    "2": "皮卡",
    "3": "越野车/SUV",
    "4": "商务车/MPV",
    "5": "轻型客车",
    "6": "中型客车",
    "7": "大型客车",
    "8": "公交车",
    "9": "校车",
    "10": "微型货车",
    "11": "轻型货车",
    "12": "中型货车",
    "13": "大型货车",
    "14": "重型货车",
    "15": "集装箱车",
    "16": "三轮车",
    "17": "二轮车",
    "18": "人",
    "19": "非人非车",
    "20": "叉车"}
def collate_fn(batch):
    batch_clips, batch_targets,batch_keys = zip(*batch)


    return default_collate(batch_clips), default_collate(batch_targets), batch_keys



def find_badcase(outputs, targets,keys):
    n_correct_elems = 0
    bad_keys = []
    badcase = []

    outputs = outputs.numpy()
    outputs = np.argmax(outputs,1)
    outputs = outputs.tolist()      # 车型list

    assert len(outputs) == len(targets) == len(keys)

    for i in range(len(keys)):

        if str(outputs[i]) != str(targets[i]):
            bad_keys.append(keys[i])
            bad_item = str(outputs[i]) +' '+ str(targets[i]) # 预测错误的输出和对应的真实标签
            badcase.append(bad_item)

    return bad_keys,badcase



def get_test_utils(test_path,keys_path):
    
    transform = []
    # normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
    #                                  opt.no_std_norm)
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    resize = Resize(240)
    # transform.append(ToPILImage())
    transform.append(resize)

    transform.append(ToTensor())
    transform.append(normalize)
    transform = Compose(transform)
    
    test_data = LmdbDataset_val(test_path,transform,keys_path)

    test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=512,
                                             shuffle=False,
                                             num_workers=16, 
                                             collate_fn = collate_fn,                                 
                                             pin_memory=True)
                                             

    return test_loader

def test(model,data_loader,fullnpy_path,fulltxt_path,):


    model.eval()


    full_badcase_info = []
    mid_badcase_info = []
    full_badkeys = []

    with torch.no_grad():
        for i, (inputs, targets_full,keys) in enumerate(data_loader):

            targets = targets_full.numpy().tolist()
            outputs_full = model(inputs)

            full_bad_keys,full_badcase = find_badcase(outputs_full,targets,keys)


            full_badkeys.extend(full_bad_keys)   # 所有车型有错误的 keys


            for j in range(len(full_badcase)):
                info = full_badcase[j]
                pred_full = info.split(' ')[0]
                label_full = info.split(' ')[1]


                pred_full = cls_list[pred_full]   # 标签转换为文字
                label_full = cls_list[label_full]


                key = full_bad_keys[j].decode()   # 当前batch的
                item = '{}, pred_full: {} , label_full: {} '.format(key,pred_full,label_full)
                full_badcase_info.append(item)
            
            print('{}/{}'.format(i,len(data_loader)))
            # if i == 3:
            #     break
            

        np.save(fullnpy_path,full_badkeys)


        with open(fulltxt_path,'a') as f1:
            for item in full_badcase_info:

                f1.write(item+'\n')



def resume_model(pth_path, model):
    print('loading checkpoint {} model'.format(pth_path))
    checkpoint = torch.load(pth_path, map_location='cpu')
    # assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model

def main():

    pth_path = '/home/zhaoliu/car_brand/results/fuxian/save_15.pth'
    
    test_result='/home/zhaoliu/car_full/badcase/'
    test_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/val_lmdb'
    keys_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/new_val.npy'


    fullnpy_path = test_result + 'full_badkeys.npy'
    fulltxt_path = test_result + 'full_badcase_info.txt'

    model = resnet18(num_classes=27249)
    model = resume_model(pth_path,model)
    test_loader = get_test_utils(test_path,keys_path)

    print('数据加载完毕...')
    test(model,test_loader,fullnpy_path,fulltxt_path)

if __name__ == '__main__':
    main()