import os.path as osp
import numpy as np

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 num_meshfree=None,
                 seq_len=None,
                 mesh_graph=None,
                 mesh_graph_torso=None,
                 heart_torso=None,
                 train=True,
                 subset=1,
                 label_type='size'):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.heart_torso = heart_torso
        filename = '_ir_ic_' + str(num_meshfree) + '.mat'
        # print(label_type)
        if train:
            filename = 'training' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['cor']
            dataset = matFiles['params_t']
            label_aha = matFiles['label_t']
            
        else:
            filename = 'testing' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['cor']
            dataset = matFiles['params_e']
            label_aha = matFiles['label_e']

        #N = int(N / 10)

        # if label_type == 'size':
        #     self.label = torch.from_numpy(label_size[0:N]).float()
        # else:
        #     self.label = torch.from_numpy(label_aha[0:N]).float()
        dataset = dataset.reshape(num_meshfree, -1, seq_len)

        N = dataset.shape[1]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[:, sub_index, :]
            index = np.arange(dataset.shape[1])
        
        label_aha = label_aha.astype(int)
        # self.label = torch.from_numpy(label_aha[0:N]).float()
        if self.heart_torso == 0:
            self.label = torch.from_numpy(label_aha[index])
            self.graph = mesh_graph
            self.datax = torch.from_numpy(dataset[:, index]).float()
            self.corMfree = corMfree
            print('final data size: {}'.format(self.datax.shape[1]))
        elif self.heart_torso == 1:
            self.label = torch.from_numpy(label_aha[index])
            self.heart = mesh_graph
            self.torso = mesh_graph_torso
            self.data_heart = torch.from_numpy(dataset[0:-120, index]).float()
            self.data_torso = torch.from_numpy(dataset[-120:, index]).float()
            self.heart_cor = corMfree[0:-120, 0:3]
            self.torso_cor = corMfree[-120:, 0:3]
            print('heart data size: {}'.format(self.data_heart.shape[1]))
            print('torso data size: {}'.format(self.data_torso.shape[1]))
        elif self.heart_torso == 2:
            self.label = torch.from_numpy(label_aha[index])
            self.heart = mesh_graph
            self.torso = mesh_graph_torso
            self.data_heart = torch.from_numpy(dataset[0:-120, index]).float() * 1e-2
            self.data_torso = torch.from_numpy(dataset[-120:, index]).float() * 1e-2
            self.heart_cor = corMfree[0:-120, 0:3]
            self.torso_cor = corMfree[-120:, 0:3]
            print('heart data size: {}'.format(self.data_heart.shape[1]))
            print('torso data size: {}'.format(self.data_torso.shape[1]))
        elif self.heart_torso == 3:
            self.label = torch.from_numpy(label_aha[index])
            self.heart = mesh_graph
            self.torso = mesh_graph_torso
            self.data_heart = torch.from_numpy(dataset[0:-120, index]).float() * 1e-4
            self.data_torso = torch.from_numpy(dataset[-120:, index]).float() * 1e-4
            self.heart_cor = corMfree[0:-120, 0:3]
            self.torso_cor = corMfree[-120:, 0:3]
            print('heart data size: {}'.format(self.data_heart.shape[1]))
            print('torso data size: {}'.format(self.data_torso.shape[1]))

    def getCorMfree(self):
        if self.heart_torso == 0:
            return self.corMfree
        else:
            return (self.heart_cor, self.torso_cor)

    def __len__(self):
        if self.heart_torso == 0:
            return (self.datax.shape[1])
        else:
            return (self.data_heart.shape[1])

    def __getitem__(self, idx):
        if self.heart_torso == 0:
            x = self.datax[:, [idx]]  # torch.tensor(dataset[:,[i]],dtype=torch.float)
            y = self.label[[idx]]  # torch.tensor(label_aha[[i]],dtype=torch.float)

            sample = Data(
                x=x,
                y=y
            )
            return sample
        elif self.heart_torso == 1 or self.heart_torso == 2 or self.heart_torso == 3:
            sample = Data(
                x=self.data_torso[:, [idx]],
                y=self.data_heart[:, [idx]],
                pos=self.label[[idx]]
            )
            return sample


class HeartEmptyGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. The features and target values are 
    set to zeros in given graph.
    Not suitable for training.
    """

    def __init__(self,
                 mesh_graph,
                 label_type=None):
        self.graph = mesh_graph
        dim = self.graph.pos.shape[0]
        self.datax = np.zeros((dim, 201))
        self.label = np.zeros((201))
        #print((self.datax.shape[1]))

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.datax[:, [idx]]).float()  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = torch.from_numpy(self.label[[idx]]).float()  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        # print(sample)
        return sample


class HeartNetSubset(InMemoryDataset):
    url = 'NO URL'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 mfree=None,
                 seq_len=None):
        self.mfree = mfree
        self.seq_len = seq_len
        super(HeartNetSubset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'testing_ir_ic_{}.mat'.format(self.mfree)

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download testing_ir_ic_???? from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        # extract data from the matlab files
        #path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        filename = 'testing_ir_ic_' + str(self.mfree) + '.mat'
        path = osp.join(self.raw_dir, filename)

        matFiles = scipy.io.loadmat(path,squeeze_me=True,struct_as_record=False)
        corMfree = matFiles['cor']
        corMfree = corMfree[:, 0:3]
        dataset = matFiles['params_e']
        label_aha = matFiles['label_e']
        label_aha = label_aha.astype(int)

        dataset = dataset.reshape(self.mfree, -1, self.seq_len)
        num_nodes, tot_data, _ = dataset.shape
        data_list = []
        for i in range(tot_data):
            pos = torch.from_numpy(corMfree).float() #torch.tensor(corMfree,dtype=torch.float)
            x = torch.from_numpy(dataset[:,[i]]).float()#torch.tensor(dataset[:,[i]],dtype=torch.float)
            y = torch.from_numpy(label_aha[[i]]).float()#torch.tensor(label_aha[[i]],dtype=torch.float)
            data = Data(pos = pos, x = x, y = y) # geometry, features, and label
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)            
            data_list.append(data)
        
        np.random.seed(45)
        rnd_idx  = np.random.permutation(tot_data)

        if tot_data > 1:
            train_split, test_split = rnd_idx[:int(0.8 *tot_data)], rnd_idx[int(0.8 *tot_data):]
            data_list_train = [data_list[i] for i in train_split]
            data_list_test = [data_list[i] for i in test_split]
        else:
            data_list_train = data_list
            data_list_test = data_list
        
        torch.save(self.collate(data_list_train), self.processed_paths[0])
        torch.save(self.collate(data_list_test), self.processed_paths[1])
