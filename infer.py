import json
import random
import os
import torchvision
import torch
import numpy as np
from torch import nn
import time
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,ToPILImage
from pathlib import Path

from model.inceptionv4 import Inceptionv4  
from model.resnet import resnet18
from utils import get_mean_std,Logger, worker_init_fn, get_lr,AverageMeter, calculate_accuracy,calculate_union_accuracy,find_badcase
# from utils import AverageMeter, calculate_accuracy
from dataset.infer_dataloader import LmdbDataset_val
from torch.utils.data._utils.collate import default_collate  # 注意版本



def collate_fn(batch):
    batch_clips, batch_targets,batch_keys = zip(*batch)

    # batch_keys = [key for key in batch_keys]

    return default_collate(batch_clips), default_collate(batch_targets), batch_keys

label_gt = '/home/zhaoliu/car_class/maps/label_2_gt.json'
gt_chexing = '/home/zhaoliu/car_class/maps/gt_2_chexing.json'
gt_ori = '/home/zhaoliu/carbrand-master/carbrand-master/alldata_new1/sub_2_fb'
gt_cls = '/home/zhaoliu/car_class/maps/gt_2_cls.json'
cls_mb = '/home/zhaoliu/carbrand-master/carbrand-master/alldata_new1/mid_2_label'
mb_tmb = '/home/zhaoliu/carbrand-master/carbrand-master/dict/all_mid_2_label_dict'



with open(label_gt,'r') as f1:
	label_2_gt = json.load(f1)

with open(gt_chexing,'r') as f2:
    gt_2_chexing = json.load(f2)


gt_2_ori={}
cls_2_mb={}
mb_2_tmb={}
for line in open(gt_ori):
	index,index1 = line.strip().split()
	gt_2_ori[index]=index1

for line in open(cls_mb):
    index,index1 = line.strip().split()
    cls_2_mb[index]=index1
with open(gt_cls,'r') as f3:
    gt_2_cls = json.load(f3)
for line in open(mb_tmb):
    index,index1 = line.strip().split()
    mb_2_tmb[index1]=index


def get_test_utils(test_path,test_result,keys_path):
    
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
                                             

    test_logger = Logger(test_result / 'val.log',
                            ['acc'])
                                            #  worker_init_fn=worker_init_fn)
    test_batch_logger = Logger(
            test_result / 'test.log',
            ['batch', 'acc'])


    return test_loader,test_batch_logger,test_logger



def test(model,test_loader,test_batch_logger,test_logger,badcase_npy):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    accuracies_chexing = AverageMeter()
    accuracies_ori = AverageMeter()

    accuracies = AverageMeter()
    print('----------开始测试----------')

    badcase_keys=[]
    with torch.no_grad():
        for i, (inputs, targets,keys) in enumerate(test_loader):
            data_time.update(time.time() - end_time)

            targets = targets.numpy().tolist()
            targets = [label_2_gt[str(target)] for target in targets]


            labels_chexing = [int(gt_2_chexing[str(target)]) for target in targets]
            labels_ori = [int(gt_2_ori[str(target)]) for target in targets]


            labels_chexing = torch.tensor(labels_chexing)
            labels_ori = torch.tensor(labels_ori)
            out_chexing,out_ori = model(inputs)

            # acc = calculate_accuracy(out_chexing, labels_chexing)
            acc_chexing = calculate_accuracy(out_chexing, labels_chexing)
            acc_ori = calculate_accuracy(out_ori,labels_ori)
            acc = calculate_union_accuracy((out_chexing,out_ori),(labels_chexing,labels_ori))
            badcase = find_badcase((out_chexing,out_ori),(labels_chexing,labels_ori),keys)

            badcase_keys.extend(badcase)

        
            accuracies_chexing.update(acc_chexing, inputs.size(0))
            accuracies_ori.update(acc_ori, inputs.size(0))
			#
			# accuracies_midbrand.update(acc_midbrand, inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # test_batch_logger.log({
			# 	'batch': i + 1,
			# 	'acc': accuracies.val,
			# })

            print('test iter: {}/{}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Acc_chexing {acc_chexing.val:.3f} ({acc_chexing.avg:.3f})\t'
                      'Acc_ori {acc_ori.val:.3f} ({acc_ori.avg:.3f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i + 1,
                    len(test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    acc_chexing = accuracies_chexing,
                    acc_ori= accuracies_ori,
                    acc=accuracies))
            np.save(badcase_npy,badcase_keys)

    


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

    pth_path = '/home/zhaoliu/car_class+ori/results/resnet_18_newval/save_27.pth'
    
    test_result=Path('/home/zhaoliu/car_class+ori/test_result')
    test_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/val_lmdb'
    keys_path = '/home/zhaoliu/car_data/训练数据/4.9新加测试集/new_val.npy'
    # test_path = '/mnt/disk/zhaoliu_data/carlogo/lmdb/carlogo_train_new/car_256_all_lmdb'
    # keys_path = '/home/zhaoliu/car_data/val.npy'

    badcase_npy = '/home/zhaoliu/car_class+ori/test_result/badcase_val_aug.npy'
    model = resnet18(num_classes=21)
    model = resume_model(pth_path,model)
    test_loader,test_batch_logger,test_logger = get_test_utils(test_path,test_result,keys_path)
    print('数据加载完毕...')
    test(model,test_loader,test_batch_logger,test_logger,badcase_npy)

if __name__ == '__main__':
    main()
