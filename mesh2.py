import copy
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io

import torch_geometric.transforms as T
from torch_geometric.data import Data
from data_utils import HeartNetSubset
from torch_geometric.nn.pool import *
from torch_geometric.utils import normalized_cut
from torch_cluster import nearest


class GraphPyramid():
    """Construct a graph for a given heart along with a graph hierarchy.
    For graph construction: Nodes are converted to vertices, edges are added between every node
    and it K nearest neighbor (criteria can be modified) and edge attributes between any two vertices
    is the normalized differences of Cartesian coordinates if an edge exists between the nodes
    , i.e., normalized [x1-x2, y1-y2, z1-z2] and 0 otherwise.
    
    For graph hierarchy, graph clustering method is used.
    
    Args:
        heart: name of the cardiac anatomy on which to construct the  graph and its hierarchy
        K: K in KNN for defining edge connectivity in the graph
    """

    def __init__(self, heart='case3', structure='EC', mfree=1230, seq_len=201, K=6):
        """
        """
        self.path_in = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'training', heart)
        # import ipdb; ipdb.set_trace()
        self.path_structure = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'structure', structure)
        self.pre_transform = T.KNNGraph(k=K)
        self.transform = T.Cartesian(cat=False)
        self.filename = osp.join(self.path_in, 'raw', heart)
        self.mfree = mfree
        self.seq_len = seq_len

    def normalized_cut_2d(self, edge_index, pos):
        """ calculate the normalized cut 2d 
        """
        row, col = edge_index
        if pos.size(1) == 3:
            edge_attr = torch.norm(pos[row] - pos[col], dim=1)
        else:
            u = pos[row] - pos[col]
            edge_attr = torch.norm(u[:, 0:3], dim=1)
        return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

    def save_graph(self, h_g, h_g1, h_g2, h_g3, h_g4, h_P10, h_P21, h_P32, h_P43,
                    t_g, t_g1, t_g2, t_g3, t_P10, t_P21, t_P32, H_inv, P):
        """save the graphs and the pooling matrices in a file
        """
        with open(self.filename + '_graclus_hier' + '.pickle', 'wb') as f:
            pickle.dump(h_g, f)
            pickle.dump(h_g1, f)
            pickle.dump(h_g2, f)
            pickle.dump(h_g3, f)
            pickle.dump(h_g4, f)
            # pickle.dump(h_g5, f)
            # pickle.dump(h_g6, f)

            pickle.dump(h_P10, f)
            pickle.dump(h_P21, f)
            pickle.dump(h_P32, f)
            pickle.dump(h_P43, f)
            # pickle.dump(h_P54, f)
            # pickle.dump(h_P65, f)

            pickle.dump(t_g, f)
            pickle.dump(t_g1, f)
            pickle.dump(t_g2, f)
            pickle.dump(t_g3, f)

            pickle.dump(t_P10, f)
            pickle.dump(t_P21, f)
            pickle.dump(t_P32, f)

            pickle.dump(H_inv, f)
            pickle.dump(P, f)

    def load_graph(self):
        """load the graphs and pooling matrices; used to test existing files
        """
        with open(filename + '.pickle', 'rb') as f:
            g = pickle.load(f)
            g1 = pickle.load(f)
            g2 = pickle.load(f)
            g3 = pickle.load(f)
            g4 = pickle.load(f)
            g5 = pickle.load(f)
            g6 = pickle.load(f)

            P10 = pickle.load(f)
            P21 = pickle.load(f)
            P32 = pickle.load(f)
            P43 = pickle.load(f)
            P54 = pickle.load(f)
            P65 = pickle.load(f)

        P01 = P10 / P10.sum(axis=0)
        P12 = P21 / P21.sum(axis=0)
        P23 = P32 / P32.sum(axis=0)
        P34 = P43 / P43.sum(axis=0)
        P45 = P54 / P54.sum(axis=0)
        P56 = P65 / P65.sum(axis=0)
        return g, g1, g2, g3, g4, g5, g6, P10, P21, P32, P01, P12, P23
    
    def cluster_mesh(self, g, cluster, cor, edge_index):
        m = len(cor)
        n = len(cluster)
        P = np.zeros((n, m))
        for i in range(n):
            j = cluster[i] - 1
            P[i, j] = 1
        Pn = P / P.sum(axis=0)
        PnT = torch.from_numpy(np.transpose(Pn)).float()

        m, _, s = g.x.shape
        x = g.x.view(m, s)
        x = torch.mm(PnT, x)
        x = x.view(-1, 1, s)
        
        edge_index = torch.tensor(edge_index)
        cor = torch.tensor(cor).float()
        g_coarse = Data(x=x, y=g.y, pos=cor, edge_index=edge_index)
        g_coarse = self.transform(g_coarse)
        return P, g_coarse


    def clus_heart(self, d, method='graclus'):
        """Use graph clustering method to make a hierarchy of coarser-finer graphs
        
        Args:
            method: graph clustering method to use (options: graclus or voxel)
            d: a instance of Data class (a graph object)
        
        Output:
            P: transformation matrix from coarser to finer scale
            d_coarser: graph for the coarser scale
        """
        # clustering
        if (method == 'graclus'):
            weight = self.normalized_cut_2d(d.edge_index, d.pos)
            cluster = graclus(d.edge_index, weight, d.x.size(0))
        elif (method == 'voxel'):
            cluster = voxel_grid(d.pos, torch.tensor(np.zeros(d.pos.shape[0])), size=10)
        else:
            print('this clustering method has not been implemented')

        # get clusters assignments with consequitive numbers
        cluster, perm = self.consecutive_cluster(cluster)
        unique_cluster = np.unique(cluster)
        n, m = cluster.shape[0], unique_cluster.shape[0]  # num nodes, num clusters

        # transformaiton matrix that consists of num_nodes X num_clusters
        P = np.zeros((n, m))
        # P_{ij} = 1 if ith node in the original cluster was merged to jth node in coarser scale
        for j in range(m):
            i = np.where(cluster == int(unique_cluster[j]))
            P[i, j] = 1
        Pn = P / P.sum(axis=0)  # column normalize P
        PnT = torch.from_numpy(np.transpose(Pn)).float()  # PnT tranpose
        # the coarser scale features =  Pn^T*features
        # this is done for verification purpose only
        m, _, s = d.x.shape
        x = d.x.view(m, s)
        x = torch.mm(PnT, x)  # downsampled features
        pos = torch.mm(PnT, d.pos)  # downsampled coordinates (vertices)
        x = x.view(-1, 1, s)

        # convert into a new object of data class (graphical format)
        d_coarser = Data(x=x, pos=pos, y=d.y)
        d_coarser = self.pre_transform(d_coarser)
        d_coarser = self.transform(d_coarser)
        return P, d_coarser

    def declus_heart(self, gn_coarse, gn, Pr):
        """ Test the up-pooling matrix. Obtain finer scale features by patching operation.
        
        Args:
            gn: finer scale graph
            gn_coarse: coarser scale graph
            Pr: gn.features = Pr*gn_coarse.features  (obtain finer scale features)
        """
        m, _, s = gn_coarse.x.shape
        x = gn_coarse.x.view(m, s)
        x = torch.mm(torch.from_numpy(Pr).float(), x)
        x = x.view(-1, 1, s)
        pos = gn.pos
        edge_index = gn.edge_index
        edge_attr = gn.edge_attr
        d_finer = Data(edge_attr=edge_attr, edge_index=edge_index,
                       x=x, pos=pos, y=gn_coarse.y)
        return d_finer

    def consecutive_cluster(self, src):
        """
        Args:
            src: cluster
        """
        unique, inv = torch.unique(src, sorted=True, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        return inv, perm

    def scatter_plots(self, data, name='_', colorby=0):
        """visualize and save the graph data

        """
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        # x, y, z coordinates
        x = data.pos[:, 0]
        y = data.pos[:, 1]
        z = data.pos[:, 2]

        # color code in the figure
        if (colorby == 0):  # by features on the nodes
            features = copy.deepcopy(data.x)
            features = np.squeeze(features.numpy())
            min_x, max_x = np.min(features, axis=1), np.max(features, axis=1)
            features = features.transpose()
            features = (features - min_x) / (max_x - min_x)
            features = features.transpose()
            features = features[:, 20]
            im = ax.scatter(x, y, z, s=30, c=features, cmap=plt.get_cmap('jet'), vmin=0.07, vmax=0.55)
        else:  # by labels on the nodes
            label = data.y
            if (len(label) > 1):
                im = ax.scatter(x, y, z, s=30, c=label, cmap=plt.get_cmap('jet'))
            else:
                im = ax.scatter(x, y, z, cmap=plt.get_cmap('jet'))
        plt.axis('off')
        fig.tight_layout()
        fig.savefig(self.filename + name + '_' + str(len(x)) + '.png', dpi=600,
                    bbox_inches='tight', transparent=True)
        plt.close()
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()
    
    def heart_torso(self, heart, torso):
        heart = copy.deepcopy(heart)
        torso = copy.deepcopy(torso)

        row = np.arange(heart.pos.shape[0])
        col = np.arange(torso.pos.shape[0])
        row_len = row.shape[0]
        col_len = col.shape[0]
        col += row_len
        edge_index = []
        for i in row:
            for j in col:
                edge_index.append([i, j])
        edge_index = np.array(edge_index)
        edge_index = edge_index.transpose()
        combine_x = torch.cat((heart.x, torso.x), 0)
        combine_pos = torch.cat((heart.pos, torso.pos), 0)
        H_inv = Data(x=combine_x, y=heart.y, pos=combine_pos)
        H_inv.edge_index = torch.tensor(edge_index)
        H_inv = self.transform(H_inv)
        
        P = np.zeros((row_len, col_len, 12))
        
        return H_inv, P
    
    def face2edge(self, face):
        edge_index = []
        for triangle in face:
            a, b, c = triangle
            if [a, b] not in edge_index:
                edge_index.append([a, b])
            if [b, a] not in edge_index:
                edge_index.append([b, a])
            if [a, c] not in edge_index:
                edge_index.append([a, c])
            if [c, a] not in edge_index:
                edge_index.append([c, a])
            if [b, c] not in edge_index:
                edge_index.append([b, c])
            if [c, b] not in edge_index:
                edge_index.append([c, b])
        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = np.array(edge_index).transpose()
        edge_index = edge_index - 1
        return edge_index.astype(np.int64)
    
    def save_connection(self, g, name, face=0):
        edge_index = g.edge_index
        pos = g.pos

        edge_index = edge_index.numpy()
        edge_index = edge_index.tolist()

        pos = pos.numpy()
        pos = pos.tolist()
        file_name = self.filename + '{}.mat'.format(name)
        scipy.io.savemat(file_name, {'pos': pos, 'edge_index': edge_index, 'face': face})

    def make_graph(self, heart_name, K=6):
        """Main function for constructing the graph and its hierarchy
        """

        # Create a graph on a subset of datapoints with pre-transform and transform properties 
        train_dataset = HeartNetSubset(self.path_in, True, pre_transform=self.pre_transform,
                                       transform=self.transform, mfree=self.mfree, seq_len=self.seq_len)
        # one instance of the graph class
        testdata = train_dataset[6]
        heart = Data(x=testdata.x[0:-120, :], y=testdata.y, pos=testdata.pos[0:-120, :])
        torso = Data(x=testdata.x[-120:, :], y=testdata.y, pos=testdata.pos[-120:, :])

        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_0.mat'.format(heart_name)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        edge_index = self.face2edge(face)
        heart.edge_index = torch.tensor(edge_index)
        herat = self.transform(heart)
        torso = self.pre_transform(torso)
        torso = self.transform(torso)

        # begin creating a graph hierarchy (downpooling operation)
        h_g = copy.deepcopy(heart)  # graph at the meshfree nodes level
        # self.save_connection(h_g, name='h0', face=face)  # plot the graph

        # h_P1, h_g1 = copy.deepcopy(self.clus_heart(h_g))
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_1.mat'.format(heart_name)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        cor = matFiles['cor']
        cluster = matFiles['cluster']
        edge_index = self.face2edge(face)
        h_P1, h_g1 = copy.deepcopy(self.cluster_mesh(h_g, cluster, cor, edge_index))
        # self.save_connection(h_g1, name='h1', face=face)

        # h_P2, h_g2 = copy.deepcopy(self.clus_heart(h_g1))
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_2.mat'.format(heart_name)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        cor = matFiles['cor']
        cluster = matFiles['cluster']
        edge_index = self.face2edge(face)
        h_P2, h_g2 = copy.deepcopy(self.cluster_mesh(h_g1, cluster, cor, edge_index))
        # self.save_connection(h_g2, name='h2', face=face)
        
        # h_P3, h_g3 = copy.deepcopy(self.clus_heart(h_g2))
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_3.mat'.format(heart_name)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        cor = matFiles['cor']
        cluster = matFiles['cluster']
        edge_index = self.face2edge(face)
        h_P3, h_g3 = copy.deepcopy(self.cluster_mesh(h_g2, cluster, cor, edge_index))
        # self.save_connection(h_g3, name='h3', face=face)
        
        # h_P4, h_g4 = copy.deepcopy(self.clus_heart(h_g3))
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_4.mat'.format(heart_name)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        cor = matFiles['cor']
        cluster = matFiles['cluster']
        edge_index = self.face2edge(face)
        h_P4, h_g4 = copy.deepcopy(self.cluster_mesh(h_g3, cluster, cor, edge_index))
        # self.save_connection(h_g4, name='h4', face=face)

        t_g = copy.deepcopy(torso)  # graph at the meshfree nodes level
        # self.save_connection(t_g, name='t0')  # plot the graph
        t_P1, t_g1 = copy.deepcopy(self.clus_heart(t_g))
        # self.save_connection(t_g1, name='t1')
        t_P2, t_g2 = copy.deepcopy(self.clus_heart(t_g1))
        # self.save_connection(t_g2, name='t2')
        t_P3, t_g3 = copy.deepcopy(self.clus_heart(t_g2))
        # self.save_connection(t_g3, name='t3')


        H_inv, P = self.heart_torso(h_g4, t_g3)
        self.save_graph(h_g, h_g1, h_g2, h_g3, h_g4, h_P1, h_P2, h_P3, h_P4,\
                        t_g, t_g1, t_g2, t_g3, t_P1, t_P2, t_P3, H_inv, P)
