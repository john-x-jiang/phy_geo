import pickle
import copy
import numpy as np
import scipy.io
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch_geometric.data import DataLoader
from data_utils import HeartEmptyGraphDataset
from Spline import SplineSample
from nn_modules import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphTorsoHeart(nn.Module):
    def __init__(self, hparams, training=True):
        super().__init__()
        self.nf = hparams.nf
        self.ns = hparams.ns
        self.batch_size = hparams.batch_size if training else 1
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim
        self.latent_seq = hparams.latent_seq

        self.conv1 = st_gcn(self.nf[0], self.nf[2], self.ns[0], self.ns[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = st_gcn(self.nf[2], self.nf[3], self.ns[2], self.ns[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = st_gcn(self.nf[3], self.nf[4], self.ns[3], self.ns[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[-1], 1)
        self.fce21 = nn.Conv2d(self.nf[-1], self.latent_dim, 1)
        # self.fce22 = nn.Conv2d(self.nf[-1], self.latent_dim, 1)

        self.trans = SplineSample(self.latent_dim, self.latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        self.fcd3 = nn.Conv2d(self.latent_dim, self.nf[-1], 1)
        self.fcd4 = nn.Conv2d(self.nf[-1], self.nf[5], 1)

        self.deconv4 = st_gcn(self.nf[5], self.nf[3], self.ns[4], self.ns[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = st_gcn(self.nf[3], self.nf[2], self.ns[3], self.ns[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = st_gcn(self.nf[2], self.nf[1], self.ns[2], self.ns[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = st_gcn(self.nf[1], self.nf[0], self.ns[1], self.ns[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        # self.bg4 = dict()
        # self.bg5 = dict()
        # self.bg6 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
        # self.P54 = dict()
        # self.P65 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()
        self.tg3 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.h_L = dict()
        self.t_L = dict()
        self.H = dict()

    def set_graphs(self, gParams, heart_name):
        self.bg[heart_name] = gParams["bg"]
        self.bg1[heart_name] = gParams["bg1"]
        self.bg2[heart_name] = gParams["bg2"]
        self.bg3[heart_name] = gParams["bg3"]
        # self.bg4[heart_name] = gParams["bg4"]
        # self.bg5[heart_name] = gParams["bg5"]
        # self.bg6[heart_name] = gParams["bg6"]
        
        self.P10[heart_name] = gParams["P10"]
        self.P21[heart_name] = gParams["P21"]
        self.P32[heart_name] = gParams["P32"]
        self.P43[heart_name] = gParams["P43"]
        # self.P54[heart_name] = gParams["P54"]
        # self.P65[heart_name] = gParams["P65"]

        self.tg[heart_name] = gParams["t_bg"]
        self.tg1[heart_name] = gParams["t_bg1"]
        self.tg2[heart_name] = gParams["t_bg2"]
        self.tg3[heart_name] = gParams["t_bg3"]

        self.t_P01[heart_name] = gParams["t_P01"]
        self.t_P12[heart_name] = gParams["t_P12"]
        self.t_P23[heart_name] = gParams["t_P23"]

        self.H_inv[heart_name] = gParams["H"]
        self.P[heart_name] = gParams["P"]

    def set_physics(self, h_L, t_L, H, heart_name):
        self.h_L[heart_name] = h_L
        self.t_L[heart_name] = t_L
        self.H[heart_name] = H
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(self.batch_size, -1, self.nf[0], self.seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[2] * self.ns[2])
        x = torch.matmul(self.t_P01[heart_name], x)  # bs X 648 X f[1]
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.ns[2]), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[3] * self.ns[3])
        x = torch.matmul(self.t_P12[heart_name], x)  # bs X 347 X f[2]
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.ns[3]), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[4] * self.ns[4])
        x = torch.matmul(self.t_P23[heart_name], x)  # bs X 184 X f[3]
        x = x.view(self.batch_size, -1, self.nf[4], self.ns[4])

        # latent
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)

        mu = self.fce21(x)
        # logvar = self.fce22(x)
        return mu
    
    # def reparameterize(self, mu, logvar):
    #     """ reparameterization; draw a random sample from the p(z|x)
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)
    
    def inverse(self, z, heart_name):
        x = z.view(self.batch_size, self.latent_dim, -1, self.latent_seq)

        x = x.permute(0, 2, 1, 3).contiguous()
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        
        num_heart, num_torso = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(self.batch_size, num_heart, self.latent_dim, self.latent_seq).to(device)
        x_bin = torch.cat((x_bin, x), 1)
        
        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(self.batch_size, num_heart + num_torso, self.latent_seq, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.latent_seq, self.batch_size, -1, self.latent_dim)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_torso, :, :]
        x_bin = x_bin.permute(0, 2, 1, 3).contiguous()
        return x_bin
    
    def decode(self, z, heart_name):
        """ graph  convolutional decoder
        """
        x = F.elu(self.fcd3(z), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(self.batch_size, -1, self.nf[5] * self.ns[4])
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[5], self.ns[4]), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)  # (bs*184) X f[3]

        x = x.view(self.batch_size, -1, self.nf[3] * self.ns[3])
        x = torch.matmul(self.P32[heart_name], x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.ns[3]), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)  # (bs*351) X f[2]

        x = x.view(self.batch_size, -1, self.nf[2] * self.ns[2])
        x = torch.matmul(self.P21[heart_name], x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.ns[2]), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)  # (bs*646) X f[1]

        x = x.view(self.batch_size, -1, self.nf[1] * self.ns[1])
        x = torch.matmul(self.P10[heart_name], x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[1], self.ns[1]), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)  # (bs*1230) X f[0]

        x = x.view(-1, self.nf[0], self.seq_len)
        return x

    def physics(self, phi_t, phi_h, heart_name):
        phi_t = phi_t.view(self.batch_size, -1, self.seq_len)
        phi_h = phi_h.view(self.batch_size, -1, self.seq_len)
        # laplacian
        l_h = torch.matmul(self.h_L[heart_name], phi_h)
        phi_t_ = torch.matmul(self.H[heart_name], phi_h)
        return phi_t_, l_h
    
    def forward(self, phi_t, heart_name):
        mu = self.encode(phi_t, heart_name)
        # z = self.reparameterize(mu, logvar)
        z = self.inverse(mu, heart_name)
        phi_h = self.decode(z, heart_name)
        phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
        return phi_h, phi_t_, l_h, torch.zeros_like(mu), torch.zeros_like(mu), torch.zeros_like(mu), torch.zeros_like(mu)


class Graph_ODE_RNN(nn.Module):
    def __init__(self, hparams, training=True):
        super().__init__()
        self.nf = hparams.nf
        self.batch_size = hparams.batch_size if training else 1
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim
        self.ode_func_type = hparams.ode_func_type
        self.num_layers = hparams.num_layers
        self.method = hparams.method
        self.rtol = hparams.rtol
        self.atol = hparams.atol
        self.cell_type = hparams.cell_type
        self.decoder_out = hparams.decoder_out

        self.conv1 = gcn(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = gcn(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = gcn(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[6], 1)
        self.fce2 = nn.Conv2d(self.nf[6], self.latent_dim, 1)

        self.trans = SplineSample(self.latent_dim, self.latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        self.ode_rnn = ODERNN(input_dim=self.latent_dim, hidden_dim=self.latent_dim, kernel_size=3, dim=3, norm=False, 
                            ode_func_type=self.ode_func_type, num_layers=self.num_layers, method=self.method, 
                            rtol=self.rtol, atol=self.atol, cell_type=self.cell_type)

        self.fcd3 = nn.Conv2d(self.latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = gcn(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = gcn(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = gcn(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = gcn(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()
        # self.bg5 = dict()
        # self.bg6 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
        # self.P54 = dict()
        # self.P65 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()
        self.tg3 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.h_L = dict()
        self.t_L = dict()
        self.H = dict()

    def set_graphs(self, gParams, heart_name):
        self.bg[heart_name] = gParams["bg"]
        self.bg1[heart_name] = gParams["bg1"]
        self.bg2[heart_name] = gParams["bg2"]
        self.bg3[heart_name] = gParams["bg3"]
        self.bg4[heart_name] = gParams["bg4"]
        # self.bg5[heart_name] = gParams["bg5"]
        # self.bg6[heart_name] = gParams["bg6"]
        
        self.P10[heart_name] = gParams["P10"]
        self.P21[heart_name] = gParams["P21"]
        self.P32[heart_name] = gParams["P32"]
        self.P43[heart_name] = gParams["P43"]
        # self.P54[heart_name] = gParams["P54"]
        # self.P65[heart_name] = gParams["P65"]

        self.tg[heart_name] = gParams["t_bg"]
        self.tg1[heart_name] = gParams["t_bg1"]
        self.tg2[heart_name] = gParams["t_bg2"]
        self.tg3[heart_name] = gParams["t_bg3"]

        self.t_P01[heart_name] = gParams["t_P01"]
        self.t_P12[heart_name] = gParams["t_P12"]
        self.t_P23[heart_name] = gParams["t_P23"]

        self.H_inv[heart_name] = gParams["H"]
        self.P[heart_name] = gParams["P"]

    def set_physics(self, h_L, t_L, H, heart_name):
        self.h_L[heart_name] = h_L
        self.t_L[heart_name] = t_L
        self.H[heart_name] = H
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(self.batch_size, -1, self.nf[0], self.seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.t_P01[heart_name], x)  # bs X 648 X f[1]
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.seq_len), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.t_P12[heart_name], x)  # bs X 347 X f[2]
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.seq_len), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.t_P23[heart_name], x)  # bs X 184 X f[3]
        # x = x.view(self.batch_size, -1, self.nf[4], self.seq_len)

        # latent
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[4], self.seq_len), self.tg3[heart_name].edge_index, self.tg3[heart_name].edge_attr
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def inverse(self, x, heart_name):
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_right, num_left = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(self.batch_size, num_right, self.latent_dim, self.seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(self.batch_size, num_right + num_left, self.seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.seq_len, self.batch_size, -1, self.latent_dim)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def ode_model(self, x, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        h, h_ = self.ode_rnn(x, edge_index, edge_attr)
        return h, h_
    
    def decode(self, x, heart_name):
        """ graph  convolutional decoder
        """
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        pseudo_batch = self.batch_size if not self.decoder_out else 2 * self.batch_size

        x = x.view(pseudo_batch, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[4], self.seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)  # (bs*184) X f[3]

        x = x.view(pseudo_batch, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.P32[heart_name], x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[3], self.seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)  # (bs*351) X f[2]

        x = x.view(pseudo_batch, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.P21[heart_name], x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[2], self.seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)  # (bs*646) X f[1]

        x = x.view(pseudo_batch, -1, self.nf[1] * self.seq_len)
        x = torch.matmul(self.P10[heart_name], x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[1], self.seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)  # (bs*1230) X f[0]

        x = x.view(pseudo_batch, -1, self.seq_len)
        return x

    def physics(self, phi_t, phi_h, heart_name):
        phi_t = phi_t.view(self.batch_size, -1, self.seq_len)
        # phi_h = phi_h.view(self.batch_size, -1, self.seq_len)
        # laplacian
        l_h = torch.matmul(self.h_L[heart_name], phi_h)
        phi_t_ = torch.matmul(self.H[heart_name], phi_h)
        return phi_t_, l_h
    
    def forward(self, phi_t, heart_name):
        mu = self.encode(phi_t, heart_name)
        z = self.inverse(mu, heart_name)
        h, h_ = self.ode_model(z, heart_name)
        
        if self.decoder_out:
            h_concat = torch.cat([h, h_], dim=0)
            phi = self.decode(h_concat, heart_name)
            phi_h, phi_h_ = phi[0:self.batch_size, :, :], phi[self.batch_size:2 * self.batch_size, :, :]
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(mu), torch.zeros_like(mu), phi_h, phi_h_
        else:
            phi_h = self.decode(h, heart_name)
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(mu), torch.zeros_like(mu), h, h_


class Graph_ODE_RNN_Embedding(nn.Module):
    def __init__(self, hparams, training=True):
        super().__init__()
        self.nf = hparams.nf
        self.batch_size = hparams.batch_size if training else 1
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim
        self.ode_func_type = hparams.ode_func_type
        self.num_layers = hparams.num_layers
        self.method = hparams.method
        self.rtol = hparams.rtol
        self.atol = hparams.atol
        self.cell_type = hparams.cell_type
        self.decoder_out = hparams.decoder_out

        self.conv1 = gcn(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = gcn(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = gcn(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.trans_enc = SplineSample(self.nf[4], self.nf[5], dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)

        self.fce1 = nn.Conv2d(self.nf[5], self.nf[7], 1)
        self.fce2 = nn.Conv2d(self.nf[7], self.latent_dim, 1)
        
        self.ode_rnn = ODERNN(input_dim=self.latent_dim, hidden_dim=self.latent_dim, kernel_size=3, dim=3, norm=False, 
                            ode_func_type=self.ode_func_type, num_layers=self.num_layers, method=self.method, 
                            rtol=self.rtol, atol=self.atol, cell_type=self.cell_type)

        self.fcd3 = nn.Conv2d(self.latent_dim, self.nf[7], 1)
        self.fcd4 = nn.Conv2d(self.nf[7], self.nf[6], 1)

        self.trans_dec = SplineSample(self.nf[6], self.nf[4], dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        self.deconv4 = gcn(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = gcn(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = gcn(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = gcn(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()
        # self.bg5 = dict()
        # self.bg6 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
        # self.P54 = dict()
        # self.P65 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()
        self.tg3 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_enc = dict()
        self.P_enc = dict()
        
        self.H_dec = dict()
        self.P_dec = dict()

        self.h_L = dict()
        self.t_L = dict()
        self.H = dict()

    def set_graphs(self, gParams, heart_name):
        self.bg[heart_name] = gParams["bg"]
        self.bg1[heart_name] = gParams["bg1"]
        self.bg2[heart_name] = gParams["bg2"]
        self.bg3[heart_name] = gParams["bg3"]
        self.bg4[heart_name] = gParams["bg4"]
        # self.bg5[heart_name] = gParams["bg5"]
        # self.bg6[heart_name] = gParams["bg6"]
        
        self.P10[heart_name] = gParams["P10"]
        self.P21[heart_name] = gParams["P21"]
        self.P32[heart_name] = gParams["P32"]
        self.P43[heart_name] = gParams["P43"]
        # self.P54[heart_name] = gParams["P54"]
        # self.P65[heart_name] = gParams["P65"]

        self.tg[heart_name] = gParams["t_bg"]
        self.tg1[heart_name] = gParams["t_bg1"]
        self.tg2[heart_name] = gParams["t_bg2"]
        self.tg3[heart_name] = gParams["t_bg3"]

        self.t_P01[heart_name] = gParams["t_P01"]
        self.t_P12[heart_name] = gParams["t_P12"]
        self.t_P23[heart_name] = gParams["t_P23"]

        self.H_enc[heart_name] = gParams["H_enc"]
        self.P_enc[heart_name] = gParams["P_enc"]
        
        self.H_dec[heart_name] = gParams["H_dec"]
        self.P_dec[heart_name] = gParams["P_dec"]

    def set_physics(self, h_L, t_L, H, heart_name):
        self.h_L[heart_name] = h_L
        self.t_L[heart_name] = t_L
        self.H[heart_name] = H
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(self.batch_size, -1, self.nf[0], self.seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.t_P01[heart_name], x)  # bs X 648 X f[1]
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.seq_len), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.t_P12[heart_name], x)  # bs X 347 X f[2]
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.seq_len), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.t_P23[heart_name], x)  # bs X 184 X f[3]
        x = x.view(self.batch_size, -1, self.nf[4], self.seq_len)

        # layer 4
        x = self.inverse(x, heart_name)

        # latent
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def inverse(self, x, heart_name, is_encode=True):
        N, V, C, T = x.shape
        if is_encode:
            batch_size = self.batch_size
            edge_index, edge_attr = self.H_enc[heart_name].edge_index, self.H_enc[heart_name].edge_attr
            num_right, num_left = self.P_enc[heart_name].shape[0], self.P_enc[heart_name].shape[1]
        else:
            batch_size = self.batch_size if not self.decoder_out else 2 * self.batch_size
            edge_index, edge_attr = self.H_dec[heart_name].edge_index, self.H_dec[heart_name].edge_attr
            num_right, num_left = self.P_dec[heart_name].shape[0], self.P_dec[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_right, C, self.seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, C)
        edge_index, edge_attr = expand(batch_size, num_right + num_left, self.seq_len, edge_index, edge_attr)

        if is_encode:
            x_bin = self.trans_enc(x_bin, edge_index, edge_attr)
        else:
            x_bin = self.trans_dec(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.seq_len, batch_size, num_right + num_left, -1)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def ode_model(self, x, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        h, h_ = self.ode_rnn(x, edge_index, edge_attr)
        return h, h_
    
    def decode(self, x, heart_name):
        """ graph  convolutional decoder
        """
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        pseudo_batch = self.batch_size if not self.decoder_out else 2 * self.batch_size

        x = x.view(pseudo_batch, -1, self.nf[6], self.seq_len)
        x = self.inverse(x, heart_name, is_encode=False)

        x = x.view(pseudo_batch, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[4], self.seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)  # (bs*184) X f[3]

        x = x.view(pseudo_batch, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.P32[heart_name], x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[3], self.seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)  # (bs*351) X f[2]

        x = x.view(pseudo_batch, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.P21[heart_name], x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[2], self.seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)  # (bs*646) X f[1]

        x = x.view(pseudo_batch, -1, self.nf[1] * self.seq_len)
        x = torch.matmul(self.P10[heart_name], x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[1], self.seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)  # (bs*1230) X f[0]

        x = x.view(pseudo_batch, -1, self.seq_len)
        return x

    def physics(self, phi_t, phi_h, heart_name):
        phi_t = phi_t.view(self.batch_size, -1, self.seq_len)
        # phi_h = phi_h.view(self.batch_size, -1, self.seq_len)
        # laplacian
        l_h = torch.matmul(self.h_L[heart_name], phi_h)
        phi_t_ = torch.matmul(self.H[heart_name], phi_h)
        return phi_t_, l_h
    
    def forward(self, phi_t, heart_name):
        z = self.encode(phi_t, heart_name)
        h, h_ = self.ode_model(z, heart_name)
        
        if self.decoder_out:
            h_concat = torch.cat([h, h_], dim=0)
            phi = self.decode(h_concat, heart_name)
            phi_h, phi_h_ = phi[0:self.batch_size, :, :], phi[self.batch_size:2 * self.batch_size, :, :]
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(z), torch.zeros_like(z), phi_h, phi_h_
        else:
            phi_h = self.decode(h, heart_name)
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(z), torch.zeros_like(z), h, h_


class Graph_Filter(nn.Module):
    def __init__(self, hparams, training=True):
        super().__init__()
        self.nf = hparams.nf
        self.net_arch = hparams.net_arch
        self.batch_size = hparams.batch_size if training else 1
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim
        self.ode_func_type = hparams.ode_func_type
        self.num_layers = hparams.num_layers
        self.method = hparams.method
        self.rtol = hparams.rtol
        self.atol = hparams.atol
        self.cell_type = hparams.cell_type
        self.decoder_out = hparams.decoder_out

        self.conv1 = gcn(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = gcn(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = gcn(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[6], 1)
        self.fce2 = nn.Conv2d(self.nf[6], self.latent_dim, 1)

        self.trans = SplineSample(self.latent_dim, self.latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)

        if self.net_arch == "filter":
            self.ode_rnn = GFilter(input_dim=self.latent_dim, hidden_dim=self.latent_dim, kernel_size=3, dim=3,
                                   norm=False, ode_func_type=self.ode_func_type, num_layers=self.num_layers,
                                   method=self.method, rtol=self.rtol, atol=self.atol, cell_type=self.cell_type)
        elif self.net_arch == "flatten_filter":
            self.ode_rnn = GFlattenFilter(input_dim=self.latent_dim, hidden_dim=self.latent_dim, kernel_size=3, dim=3,
                                          norm=False, ode_func_type=self.ode_func_type, num_layers=self.num_layers,
                                          method=self.method, rtol=self.rtol, atol=self.atol, cell_type=self.cell_type)

        self.fcd3 = nn.Conv2d(self.latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = gcn(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = gcn(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = gcn(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = gcn(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()
        # self.bg5 = dict()
        # self.bg6 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
        # self.P54 = dict()
        # self.P65 = dict()

        self.tg = dict()
        self.tg1 = dict()
        self.tg2 = dict()
        self.tg3 = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

        self.H_inv = dict()
        self.P = dict()

        self.h_L = dict()
        self.t_L = dict()
        self.H = dict()

    def set_graphs(self, gParams, heart_name):
        self.bg[heart_name] = gParams["bg"]
        self.bg1[heart_name] = gParams["bg1"]
        self.bg2[heart_name] = gParams["bg2"]
        self.bg3[heart_name] = gParams["bg3"]
        self.bg4[heart_name] = gParams["bg4"]
        # self.bg5[heart_name] = gParams["bg5"]
        # self.bg6[heart_name] = gParams["bg6"]
        
        self.P10[heart_name] = gParams["P10"]
        self.P21[heart_name] = gParams["P21"]
        self.P32[heart_name] = gParams["P32"]
        self.P43[heart_name] = gParams["P43"]
        # self.P54[heart_name] = gParams["P54"]
        # self.P65[heart_name] = gParams["P65"]

        self.tg[heart_name] = gParams["t_bg"]
        self.tg1[heart_name] = gParams["t_bg1"]
        self.tg2[heart_name] = gParams["t_bg2"]
        self.tg3[heart_name] = gParams["t_bg3"]

        self.t_P01[heart_name] = gParams["t_P01"]
        self.t_P12[heart_name] = gParams["t_P12"]
        self.t_P23[heart_name] = gParams["t_P23"]

        self.H_inv[heart_name] = gParams["H"]
        self.P[heart_name] = gParams["P"]

    def set_physics(self, h_L, t_L, H, heart_name):
        self.h_L[heart_name] = h_L
        self.t_L[heart_name] = t_L
        self.H[heart_name] = H
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(self.batch_size, -1, self.nf[0], self.seq_len), self.tg[heart_name].edge_index, self.tg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.t_P01[heart_name], x)  # bs X 648 X f[1]
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.seq_len), self.tg1[heart_name].edge_index, self.tg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.t_P12[heart_name], x)  # bs X 347 X f[2]
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.seq_len), self.tg2[heart_name].edge_index, self.tg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.t_P23[heart_name], x)  # bs X 184 X f[3]
        # x = x.view(self.batch_size, -1, self.nf[4], self.seq_len)

        # latent
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[4], self.seq_len), self.tg3[heart_name].edge_index, self.tg3[heart_name].edge_attr
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def inverse(self, x, heart_name):
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_right, num_left = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(self.batch_size, num_right, self.latent_dim, self.seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)

        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(self.batch_size, num_right + num_left, self.seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.seq_len, self.batch_size, -1, self.latent_dim)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_left, :, :]
        return x_bin
    
    def ode_model(self, x, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        h, h_ = self.ode_rnn(x, edge_index, edge_attr)
        return h, h_
    
    def decode(self, x, heart_name):
        """ graph  convolutional decoder
        """
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        pseudo_batch = self.batch_size if not self.decoder_out else 2 * self.batch_size

        x = x.view(pseudo_batch, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[4], self.seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)  # (bs*184) X f[3]

        x = x.view(pseudo_batch, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.P32[heart_name], x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[3], self.seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)  # (bs*351) X f[2]

        x = x.view(pseudo_batch, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.P21[heart_name], x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[2], self.seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)  # (bs*646) X f[1]

        x = x.view(pseudo_batch, -1, self.nf[1] * self.seq_len)
        x = torch.matmul(self.P10[heart_name], x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(pseudo_batch, -1, self.nf[1], self.seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)  # (bs*1230) X f[0]

        x = x.view(pseudo_batch, -1, self.seq_len)
        return x

    def physics(self, phi_t, phi_h, heart_name):
        phi_t = phi_t.view(self.batch_size, -1, self.seq_len)
        # phi_h = phi_h.view(self.batch_size, -1, self.seq_len)
        # laplacian
        l_h = torch.matmul(self.h_L[heart_name], phi_h)
        phi_t_ = torch.matmul(self.H[heart_name], phi_h)
        return phi_t_, l_h
    
    def forward(self, phi_t, heart_name):
        mu = self.encode(phi_t, heart_name)
        z = self.inverse(mu, heart_name)
        h, h_ = self.ode_model(z, heart_name)
        
        if self.decoder_out:
            h_concat = torch.cat([h, h_], dim=0)
            phi = self.decode(h_concat, heart_name)
            phi_h, phi_h_ = phi[0:self.batch_size, :, :], phi[self.batch_size:2 * self.batch_size, :, :]
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(mu), torch.zeros_like(mu), phi_h, phi_h_
        else:
            phi_h = self.decode(h, heart_name)
            phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
            return phi_h, phi_t_, l_h, torch.zeros_like(mu), torch.zeros_like(mu), h, h_


def loss_stgcnn_mixed(recon_x, x, recon_y, y, l_h, mu, logvar, h, h_, phy_mode, smooth, hidden, *args):
    """ VAE Loss: Reconstruction + KL divergence losses summed over all elements and batch
    """
    batch_size = args[0]
    seq_len = args[1]
    epoch = args[2]
    anneal = args[3]

    if anneal:
        if epoch < 40:
            step_param = 0
        elif epoch < 400:
            step_param = epoch / 400
        else:
            step_param = 1
    else:
        step_param = 0

    if phy_mode == 0:
        r1, r2 = 1, 0
    elif phy_mode == 1:
        r1, r2 = 0, 1
    else:
        r1, r2 = 1, 1

    shape1 = np.prod(x.shape) / (batch_size * seq_len)
    shape2 = np.prod(y.shape) / (batch_size * seq_len)
    shape3 = np.prod(mu.shape) / (batch_size * 20)

    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * anneal * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    PHY = F.mse_loss(recon_y, y, reduction='sum')
    l_h = l_h.view(-1, seq_len)
    SMOOTH = F.mse_loss(l_h, torch.zeros_like(l_h), reduction='sum')
    HID = F.mse_loss(h, h_, reduction='sum')

    # BCE = BCE / shape1
    # KLD = KLD / shape3
    # PHY = PHY / shape2
    # SMOOTH = SMOOTH / shape1

    TOTAL = r1 * BCE + step_param * KLD + r2 * PHY + smooth * SMOOTH + hidden * HID

    return TOTAL, BCE, KLD, PHY, SMOOTH, HID


def loss_variation(recon_x, x, mu, logvar, *args):
    batch_size = args[0]
    seq_len = args[1]
    epoch = args[2]
    anneal = args[3]

    shape1 = np.prod(x.shape)

    BCE = F.mse_loss(recon_x, x, reduction='sum')
    BCE /= shape1

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= shape1

    return BCE + anneal * KLD, BCE, KLD


def load_graph(filename, heart_torso=0, graph_method=None):
    with open(filename + '.pickle', 'rb') as f:
        g = pickle.load(f)
        g1 = pickle.load(f)
        g2 = pickle.load(f)
        g3 = pickle.load(f)
        g4 = pickle.load(f)
        # g5 = pickle.load(f)
        # g6 = pickle.load(f)

        P10 = pickle.load(f)
        P21 = pickle.load(f)
        P32 = pickle.load(f)
        P43 = pickle.load(f)
        # P54 = pickle.load(f)
        # P65 = pickle.load(f)

        if heart_torso == 1 or heart_torso == 2 or heart_torso == 3:
            t_g = pickle.load(f)
            t_g1 = pickle.load(f)
            t_g2 = pickle.load(f)
            t_g3 = pickle.load(f)

            t_P10 = pickle.load(f)
            t_P21 = pickle.load(f)
            t_P32 = pickle.load(f)

            if graph_method == 'graclus_hier':
                Hs = pickle.load(f)
                Ps = pickle.load(f)
            elif graph_method == 'embedding':
                H_enc = pickle.load(f)
                H_dec = pickle.load(f)
                Hs = [H_enc, H_dec]

                P_enc = pickle.load(f)
                P_dec = pickle.load(f)
            else:
                raise NotImplementedError

    if heart_torso == 0:
        P01 = P10 / P10.sum(axis=0)
        P12 = P21 / P21.sum(axis=0)
        P23 = P32 / P32.sum(axis=0)
        P34 = P43 / P43.sum(axis=0)
        # P45 = P54 / P54.sum(axis=0)
        # P56 = P65 / P65.sum(axis=0)

        P01 = torch.from_numpy(np.transpose(P01)).float()
        P12 = torch.from_numpy(np.transpose(P12)).float()
        P23 = torch.from_numpy(np.transpose(P23)).float()
        P34 = torch.from_numpy(np.transpose(P34)).float()
        # P45 = torch.from_numpy(np.transpose(P45)).float()
        # P56 = torch.from_numpy(np.transpose(P56)).float()

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()
        # P54 = torch.from_numpy(P54).float()
        # P65 = torch.from_numpy(P65).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34
    elif heart_torso == 1 or heart_torso == 2 or heart_torso == 3:
        t_P01 = t_P10 / t_P10.sum(axis=0)
        t_P12 = t_P21 / t_P21.sum(axis=0)
        t_P23 = t_P32 / t_P32.sum(axis=0)

        t_P01 = torch.from_numpy(np.transpose(t_P01)).float()
        t_P12 = torch.from_numpy(np.transpose(t_P12)).float()
        t_P23 = torch.from_numpy(np.transpose(t_P23)).float()

        if graph_method == 'graclus_hier':
            Ps = torch.from_numpy(Ps).float()
        elif graph_method == 'embedding':
            P_enc = torch.from_numpy(P_enc).float()
            P_dec = torch.from_numpy(P_dec).float()
            Ps = [P_enc, P_dec]

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()
        # P54 = torch.from_numpy(P54).float()
        # P65 = torch.from_numpy(P65).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43,\
            t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps


def get_graphparams(filename, device, batch_size, heart_torso=0, graph_method=None):
    if heart_torso == 0:
        g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34 = \
            load_graph(filename, heart_torso, graph_method)
    else:
        g, g1, g2, g3, g4, P10, P21, P32, P43,\
        t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps = load_graph(filename, heart_torso, graph_method)

    num_nodes = [g.pos.shape[0], g1.pos.shape[0], g2.pos.shape[0], g3.pos.shape[0],
                 g4.pos.shape[0]#, g5.pos.shape[0], g6.pos.shape[0]
                 ]
    print(g)
    print(g1)
    print(g2)
    print(g3)
    print('P21 requires_grad:', P21.requires_grad)
    print('number of nodes:', num_nodes)

    g_dataset = HeartEmptyGraphDataset(mesh_graph=g)
    g_loader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg = next(iter(g_loader))

    g1_dataset = HeartEmptyGraphDataset(mesh_graph=g1)
    g1_loader = DataLoader(g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg1 = next(iter(g1_loader))

    g2_dataset = HeartEmptyGraphDataset(mesh_graph=g2)
    g2_loader = DataLoader(g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg2 = next(iter(g2_loader))

    g3_dataset = HeartEmptyGraphDataset(mesh_graph=g3)
    g3_loader = DataLoader(g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg3 = next(iter(g3_loader))

    g4_dataset = HeartEmptyGraphDataset(mesh_graph=g4)
    g4_loader = DataLoader(g4_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg4 = next(iter(g4_loader))

    # g5_dataset = HeartEmptyGraphDataset(mesh_graph=g5)
    # g5_loader = DataLoader(g5_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # bg5 = next(iter(g5_loader))

    # g6_dataset = HeartEmptyGraphDataset(mesh_graph=g6)
    # g6_loader = DataLoader(g6_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # bg6 = next(iter(g6_loader))

    P10 = P10.to(device)
    P21 = P21.to(device)
    P32 = P32.to(device)
    P43 = P43.to(device)
    # P54 = P54.to(device)
    # P65 = P65.to(device)

    bg1 = bg1.to(device)
    bg2 = bg2.to(device)
    bg3 = bg3.to(device)
    bg4 = bg4.to(device)
    # bg5 = bg5.to(device)
    # bg6 = bg6.to(device)
    bg = bg.to(device)

    if heart_torso == 0:
        P01 = P01.to(device)
        P12 = P12.to(device)
        P23 = P23.to(device)
        P34 = P34.to(device)
        # P45 = P45.to(device)
        # P56 = P56.to(device)

        P1n = np.ones((num_nodes[1], 1))
        Pn1 = P1n / P1n.sum(axis=0)
        Pn1 = torch.from_numpy(np.transpose(Pn1)).float()
        P1n = torch.from_numpy(P1n).float()
        P1n = P1n.to(device)
        Pn1 = Pn1.to(device)

        graphparams = {"bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4, #"bg5": bg5, "bg6": bg6,
                    "P01": P01, "P12": P12, "P23": P23, "P34": P34, #"P45": P45, "P56": P56,
                    "P10": P10, "P21": P21, "P32": P32, "P43": P43, #"P54": P54, "P65": P65,
                    "P1n": P1n, "Pn1": Pn1, "num_nodes": num_nodes, "g": g, "bg": bg}
    elif heart_torso == 1 or heart_torso == 2 or heart_torso == 3:
        t_num_nodes = [t_g.pos.shape[0], t_g1.pos.shape[0], t_g2.pos.shape[0], t_g3.pos.shape[0]]
        print(t_g)
        print(t_g1)
        print(t_g2)
        print('t_P12 requires_grad:', t_P12.requires_grad)
        print('number of nodes on torso:', t_num_nodes)
        t_g_dataset = HeartEmptyGraphDataset(mesh_graph=t_g)
        t_g_loader = DataLoader(t_g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg = next(iter(t_g_loader))

        t_g1_dataset = HeartEmptyGraphDataset(mesh_graph=t_g1)
        t_g1_loader = DataLoader(t_g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg1 = next(iter(t_g1_loader))

        t_g2_dataset = HeartEmptyGraphDataset(mesh_graph=t_g2)
        t_g2_loader = DataLoader(t_g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg2 = next(iter(t_g2_loader))

        t_g3_dataset = HeartEmptyGraphDataset(mesh_graph=t_g3)
        t_g3_loader = DataLoader(t_g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg3 = next(iter(t_g3_loader))

        t_P01 = t_P01.to(device)
        t_P12 = t_P12.to(device)
        t_P23 = t_P23.to(device)
        # t_P10 = t_P10.to(device)
        # t_P21 = t_P21.to(device)
        # t_P32 = t_P32.to(device)

        t_bg1 = t_bg1.to(device)
        t_bg2 = t_bg2.to(device)
        t_bg3 = t_bg3.to(device)
        t_bg = t_bg.to(device)

        if graph_method == 'graclus_hier':
            H_dataset = HeartEmptyGraphDataset(mesh_graph=Hs)
            H_loader = DataLoader(H_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_b = next(iter(H_loader))

            H_b = H_b.to(device)
            Ps = Ps.to(device)

            graphparams = {
                "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4, #"bg5": bg5, "bg6": bg6,
                "P10": P10, "P21": P21, "P32": P32, "P43": P43, #"P54": P54, "P65": P65,
                "num_nodes": num_nodes, "g": g, "bg": bg,
                "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                "H": H_b, "P": Ps
            }
        elif graph_method == 'embedding':
            H_enc, H_dec = Hs[0], Hs[1]
            H_enc_dataset = HeartEmptyGraphDataset(mesh_graph=H_enc)
            H_enc_loader = DataLoader(H_enc_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_enc_b = next(iter(H_enc_loader))

            H_dec_dataset = HeartEmptyGraphDataset(mesh_graph=H_dec)
            H_dec_loader = DataLoader(H_dec_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_dec_b = next(iter(H_dec_loader))

            H_enc_b = H_enc_b.to(device)
            H_dec_b = H_dec_b.to(device)
            
            P_enc, P_dec = Ps[0], Ps[1]
            P_enc = P_enc.to(device)
            P_dec = P_dec.to(device)

            graphparams = {
                "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4, #"bg5": bg5, "bg6": bg6,
                "P10": P10, "P21": P21, "P32": P32, "P43": P43, #"P54": P54, "P65": P65,
                "num_nodes": num_nodes, "g": g, "bg": bg,
                "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                "H_enc": H_enc_b, "P_enc": P_enc, "H_dec": H_dec_b, "P_dec": P_dec
            }
        else:
            raise NotImplementedError

    return graphparams


def get_physics(phy_dir, heart_name, device):
    mat_files = scipy.io.loadmat(os.path.join(phy_dir, heart_name, 'h_L.mat'), squeeze_me=True, struct_as_record=False)
    h_L = mat_files['h_L']

    mat_files = scipy.io.loadmat(os.path.join(phy_dir, heart_name, 'H.mat'), squeeze_me=True, struct_as_record=False)
    H = mat_files['H']

    h_L = torch.from_numpy(h_L).float().to(device)
    print('Load heart Laplacian: {} x {}'.format(h_L.shape[0], h_L.shape[1]))

    H = torch.from_numpy(H).float().to(device)
    print('Load H matrix: {} x {}'.format(H.shape[0], H.shape[1]))
    return h_L, None, H
