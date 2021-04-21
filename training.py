import time
import os
import sys
import json
import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy,calculate_union_accuracy,get_lr
from torch.utils.data.dataloader import default_collate

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


def train_epoch(epoch,
				data_loader1,
				data_loader2,
				model,
				criterion,
				optimizer,
				device,
				current_lr,
				epoch_logger,
				batch_logger,
				is_master_node,
				tb_writer=None,
				distributed=False):

	print('train at epoch {}'.format(epoch))

	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()


	losses = AverageMeter()


	#

	accuracies = AverageMeter()

	end_time = time.time()
	# data_loader = data_loader1+data_loader2
	# for i,(inputs,targets) in enumerate(data_loader):
	# for i,(data1,data2) in enumerate(zip(data_loader1,data_loader2)):
	dataloader_iterator = iter(data_loader2)
	for i, data1 in enumerate(data_loader1):

		try:	
			data2 = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(data_loader2)
			data2 = next(dataloader_iterator)
		

		data_time.update(time.time() - end_time)

		inputs1,targets1 = data1
		inputs2,targets2 = data2
		inputs = torch.cat((inputs1,inputs2),0)
		
		targets = torch.cat((targets1,targets2),0)




		inputs = inputs.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad()
		outputs= model(inputs)

		loss = criterion(outputs, targets)
		acc = calculate_accuracy(outputs, targets)

	
		losses.update(loss.item(), inputs.size(0))
		accuracies.update(acc, inputs.size(0))

		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end_time)
		end_time = time.time()
		itera = (epoch - 1) * int(len(data_loader1)) + (i + 1)
		batch_lr=get_lr(optimizer)
		if is_master_node:
			if tb_writer is not None:
				

				tb_writer.add_scalar('train_iter/loss_iter', losses.val, itera)
				tb_writer.add_scalar('train_iter/acc_iter', accuracies.val, itera)
				tb_writer.add_scalar('train_iter/lr_iter', batch_lr, itera)


		if batch_logger is not None:
				batch_logger.log({
				'epoch': epoch,
				'batch': i + 1,
				'iter':itera,
				'loss': losses.val,
				'acc': accuracies.val,
				'lr': current_lr
			})
		
		local_rank = 0
		if is_master_node:
			print('Train Epoch: [{0}][{1}/{2}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
				'RANK {rank}'.format(epoch,
									   i + 1,
									   len(data_loader1),
									   batch_time=batch_time,
									   data_time=data_time,
									   loss=losses,
									   acc=accuracies,
									   rank=local_rank))

		if distributed:
			loss_sum = torch.tensor([losses.sum],
								dtype=torch.float32,
								device=device)
			loss_count = torch.tensor([losses.count],
								  dtype=torch.float32,
								  device=device)
			acc_sum = torch.tensor([accuracies.sum],
							   dtype=torch.float32,
							   device=device)
			acc_count = torch.tensor([accuracies.count],
								 dtype=torch.float32,
								 device=device)

			dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
			dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
			dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
			dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

			losses.avg = loss_sum.item() / loss_count.item()
			accuracies.avg = acc_sum.item() / acc_count.item()

	if epoch_logger is not None:
		epoch_logger.log({
			'epoch': epoch,
			'loss': losses.avg,
			'acc': accuracies.avg,
			'lr': current_lr,
			'rank': local_rank
		})
	if is_master_node:
		if tb_writer is not None:
			tb_writer.add_scalar('train/loss', losses.avg, epoch)
			tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
			tb_writer.add_scalar('train/lr', current_lr, epoch)
