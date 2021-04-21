import caffe
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,ToPILImage
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from caffe.proto import caffe_pb2
import lmdb
# import torch.multiprocessing as mp
import json
# from proto import tensor_pb2
# from proto import utils
from opts import parse_opts
from pathlib import Path

from multiprocessing import Pool


class T:
    def __init__(self, a):
        self.a = a
        # 修改方式1：将对象转换为列表
        # self.b = [i for i in range(5)]

    # 修改方式2：定义属性方法获取生成器对象
    @property
    def b(self):
        return (self.a for self.a in range(5))

    def add(self, n):
        for i in self.b:
            print(i + n)

    def run(self):
        p = Pool()
        p.map(self.add, self.a)

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_mean_std(value_scale):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]


    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean,std


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.train_data1 = opt.root_path +'/'+ opt.train_data1
        opt.train_data2 = opt.root_path + '/' + opt.train_data2
        opt.val_data = opt.root_path + '/' + opt.val_data
        opt.test_data = opt.root_path + '/' + opt.test_data
        # opt.result_path = opt.root_path + '/' + opt.result_path
    opt.mean, opt.std = get_mean_std(opt.value_scale)
    opt.arch = '{}-{}'.format(opt.model, opt.mission)
    opt.begin_epoch = 1
    # opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3

    if opt.distributed:
        # opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"]) # 
        opt.dist_rank = 0  # 单个机器

        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt

class LmdbDataset_train(Dataset):
    def __init__(self,lmdb_path,optimizer,keys_path):
        # super().__init__()
        self.optimizer = optimizer

        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        # self.env = lmdb.open(lmdb_path)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']-1
            # print(self.length)
            # self.keys = [key for key, _ in txn.cursor()]
            keys = np.load(keys_path)
            self.keys = keys.tolist()

        
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            serialized_str = txn.get(self.keys[index])
        datum=caffe_pb2.Datum()
        datum.ParseFromString(serialized_str)

        size=datum.width*datum.height

        pixles1=datum.data[0:size]
        pixles2=datum.data[size:2*size]
        pixles3=datum.data[2*size:3*size]

        image1=Image.frombytes('L', (datum.width, datum.height), pixles1)
        image2=Image.frombytes('L', (datum.width, datum.height), pixles2)
        image3=Image.frombytes('L', (datum.width, datum.height), pixles3)

        img=Image.merge("RGB",(image3,image2,image1))

        img =self.optimizer(img)

        label=datum.label
        return img, label

    def __len__(self):
        return self.length


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)

def main_worker(index, opt):
    print('1')


    transform = []
    
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    resize = Resize(opt.size)

    transform.append(resize)
    transform.append(ToTensor())
    transform.append(normalize)
    
    transform = Compose(transform)
    
    training_data = LmdbDataset_train(opt.train_data1,transform,opt.keys_path_train)
    dataloader = DataLoader(training_data, batch_size=256, shuffle=True,
                                     num_workers=16)
    for i,(inputs,targets) in enumerate(dataloader):

	

        print(inputs.shape,targets.shape)
   

if __name__ == '__main__':

    import multiprocessing as mp
    opt = get_opt()
    # torch.multiprocessing.set_start_method('spawn')
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')


    opt.ngpus_per_node = 4

    opt.world_size = opt.ngpus_per_node * opt.world_size
    # main_worker = list(main_worker)
    # mp.fork(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    # main_worker(-1,opt)

    # a = [main_worker for main_worker in range(4)]
    # s = T(a)
    # s.run()
    # import os
    # from multiprocessing import Pool
    # # print('Parent process %s.' % os.getpid())
    # p = Pool(4)
    # for i in range(5):
    #     p.apply_async(main_worker, args=(opt,))
    # # p.close()
    # p.join()
    num_processes = 4

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=main_worker, args=(num_processes,opt,))
        p.start()
        processes.append(p)
    for p in processes:
      p.join()
      print(p)
