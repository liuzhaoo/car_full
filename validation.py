import torch
import time
import sys
import json
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy,calculate_union_accuracy
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

def val_epoch(epoch,
			  data_loader,
			  model,
			  criterion,
			  device,
			  logger,
			  is_master_node,
			  tb_writer=None,
			  distributed=False):
	print('validation at epoch {}'.format(epoch))

	model.eval()

	batch_time = AverageMeter()
	data_time = AverageMeter()

	# losses_chexing = AverageMeter()
	# losses_ori = AverageMeter()
	losses = AverageMeter()
	#
	# losses_midbrand = AverageMeter()

	accuracies_chexing = AverageMeter()
	accuracies_ori = AverageMeter()
	#
	# accuracies_midbrand = AverageMeter()
	accuracies = AverageMeter()

	end_time = time.time()

	with torch.no_grad():
		for i, (inputs, targets) in enumerate(data_loader):
			data_time.update(time.time() - end_time)

	
			inputs = inputs.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)


			outputs= model(inputs)

			loss = criterion(outputs, targets)
			acc = calculate_accuracy(outputs, targets)


			losses.update(loss.item(), inputs.size(0))
			accuracies.update(acc, inputs.size(0))

			batch_time.update(time.time() - end_time)
			end_time = time.time()
			itera = (epoch - 1) * len(data_loader) + (i + 1)

			if is_master_node:
				print('Val Epoch: [{0}][{1}/{2}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
					epoch,
					i + 1,
					len(data_loader),
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					acc=accuracies
					))

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

	if logger is not None:
		logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
	if is_master_node:
		if tb_writer is not None:
			tb_writer.add_scalar('val/loss', losses.avg, epoch)
			tb_writer.add_scalar('val/acc', accuracies.avg, epoch)
			
	return losses.avg
