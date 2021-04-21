import caffe
import torch

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from caffe.proto import caffe_pb2
import lmdb
# from proto import tensor_pb2
# from proto import utils


class LmdbDataset_train(Dataset):
    def __init__(self,lmdb_path,optimizer,keys_path):
        # super().__init__()
        self.optimizer = optimizer

        self.datum=caffe_pb2.Datum()
        self.lmdb_path = lmdb_path
        keys = np.load(keys_path)
        self.keys = keys.tolist()
        self.length = len(self.keys)




    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin(buffers=True,write=False)
   
    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        serialized_str = self.txn.get(self.keys[index])
        
        self.datum.ParseFromString(serialized_str)

        size=self.datum.width*self.datum.height

        pixles1=self.datum.data[0:size]
        pixles2=self.datum.data[size:2*size]
        pixles3=self.datum.data[2*size:3*size]

        image1=Image.frombytes('L', (self.datum.width, self.datum.height), pixles1)
        image2=Image.frombytes('L', (self.datum.width, self.datum.height), pixles2)
        image3=Image.frombytes('L', (self.datum.width, self.datum.height), pixles3)

        img=Image.merge("RGB",(image3,image2,image1))

        img =self.optimizer(img)

        label=self.datum.label
        return img, label

    def __len__(self):
        return self.length



