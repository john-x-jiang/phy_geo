import pickle
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch_geometric.data import DataLoader
from data_utils import HeartEmptyGraphDataset
from Spline import SplineSample


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TorsoHeart(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.in_dim = hparams.in_dim
        self.out_dim = hparams.out_dim
        self.latent_dim = hparams.latent_dim
        self.mid_dim_i = hparams.mid_dim_i
        self.mid_dim_o = hparams.mid_dim_o
        self.v_mid = hparams.v_mid
        self.v_latent = hparams.v_latent

        # encoder
        self.fc1 = nn.LSTM(self.in_dim, self.mid_dim_i)
        self.fc21 = nn.LSTM(self.mid_dim_i, self.latent_dim)
        self.fc22 = nn.LSTM(self.mid_dim_i, self.latent_dim)
        self.lin1 =nn.Linear(self.latent_dim * self.seq_len, self.v_mid)
        self.lin2 =nn.Linear(self.v_mid, self.v_latent)

        # decoder
        self.lin3 = nn.Linear(self.v_latent, self.v_mid)
        self.lin4 = nn.Linear(self.v_mid, self.latent_dim * self.seq_len)
        self.fc3 = nn.LSTM(self.latent_dim, self.mid_dim_o)
        self.fc41 = nn.LSTM(self.mid_dim_o, self.out_dim)
        self.fc42 = nn.LSTM(self.mid_dim_o, self.out_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x, heart_name):
        x = x.view(self.batch_size, -1, self.seq_len)
        x = x.permute(2, 0, 1).contiguous()
        _, B, _ = x.shape
        out, hidden = self.fc1(x)
        h1 = self.relu(out)
        out21, hidden21 = self.fc21(h1)
        outMean = out21.permute(1, 2, 0).contiguous().view(B,-1)
        outMean = self.relu(self.lin1(outMean))
        outMean = self.relu(self.lin2(outMean))
        out22, hidden22 = self.fc22(h1)
        outVar = out22.permute(1, 2, 0).contiguous().view(B, -1)
        outVar = self.relu(self.lin1(outVar))
        outVar = self.relu(self.lin2(outVar))
        return outMean, outVar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z, heart_name):
        B, _ = z.shape
        z1 = self.relu(self.lin3(z))
        z2 = self.relu(self.lin4(z1))
        z = z2.view(B, self.latent_dim,-1).permute(2, 0, 1)

        out3, hidden3 = self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1 = self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        out1 = out1.permute(1, 2, 0).contiguous()
        out2 = out2.permute(1, 2, 0).contiguous()
        return out1, out2
    
    def forward(self, x, heart_name):
        mu, logvar = self.encode(x, heart_name)
        z = self.reparameterize(mu, logvar)
        mu_theta, logvar_theta = self.decode(z, heart_name)
        return (mu_theta, logvar_theta), mu, logvar


class GraphHeart(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.nf = hparams.nf
        self.ns = hparams.ns
        self.batch_size = hparams.batch_size
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim
        self.latent_seq = hparams.latent_seq

        self.conv1 = st_gcn(self.nf[0], self.nf[1], self.ns[0], self.ns[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = st_gcn(self.nf[1], self.nf[2], self.ns[1], self.ns[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = st_gcn(self.nf[2], self.nf[3], self.ns[2], self.ns[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = st_gcn(self.nf[3], self.nf[4], self.ns[3], self.ns[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[-1], 1)
        self.fce21 = nn.Conv2d(self.nf[-1], self.latent_dim, 1)
        self.fce22 = nn.Conv2d(self.nf[-1], self.latent_dim, 1)
        
        self.fcd3 = nn.Conv2d(self.latent_dim, self.nf[-1], 1)
        self.fcd4 = nn.Conv2d(self.nf[-1], self.nf[4], 1)

        self.deconv4 = st_gcn(self.nf[4], self.nf[3], self.ns[4], self.ns[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
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

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()

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

        self.P01[heart_name] = gParams["P01"]
        self.P12[heart_name] = gParams["P12"]
        self.P23[heart_name] = gParams["P23"]
        self.P34[heart_name] = gParams["P34"]
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(self.batch_size, -1, self.nf[0], self.seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr  # (1230*bs) X f[0]
        x = self.conv1(x, edge_index, edge_attr)  # (1230*bs) X f[1]
        x = x.view(self.batch_size, -1, self.nf[1] * self.ns[1])
        x = torch.matmul(self.P01[heart_name], x)  # bs X 648 X f[1]
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[1], self.ns[1]), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)  # 648*bs X f[2]
        x = x.view(self.batch_size, -1, self.nf[2] * self.ns[2])
        x = torch.matmul(self.P12[heart_name], x)  # bs X 347 X f[2]
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.ns[2]), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[3] * self.ns[3])
        x = torch.matmul(self.P23[heart_name], x)  # bs X 184 X f[3]

        # layer 4
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.ns[3]), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)  # 347*bs X f[3]
        x = x.view(self.batch_size, -1, self.nf[4] * self.ns[4])
        x = torch.matmul(self.P34[heart_name], x)  # bs X 184 X f[3]
        x = x.view(self.batch_size, -1, self.nf[4], self.ns[4])

        # latent
        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)

        mu = self.fce21(x)
        logvar = self.fce22(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode(self, z, heart_name):
        """ graph  convolutional decoder
        """
        x = F.elu(self.fcd3(z), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(self.batch_size, -1, self.nf[4] * self.ns[4])
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[4], self.ns[4]), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
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
    
    def forward(self, data, heart_name):
        # erase all heart signal
        mu, logvar = self.encode(data, heart_name)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, heart_name), mu, logvar


class GraphTorsoHeart(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.nf = hparams.nf
        self.ns = hparams.ns
        self.batch_size = hparams.batch_size
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

        self.H_inv = dict()
        self.P = dict()

        self.t_P01 = dict()
        self.t_P12 = dict()
        self.t_P23 = dict()

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

        self.H_inv[heart_name] = gParams["H_inv"]
        self.P[heart_name] = gParams["P"]
    
    def encode(self, data, heart_name):
        """ graph convolutional encoder
        """
        # layer 1 (graph setup, conv, nonlinear, pool)
        # x = data * 1e3
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

        x = z.permute(0, 2, 1, 3).contiguous()
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        
        num_heart = self.P43[heart_name].shape[1]
        num_torso = self.t_P23[heart_name].shape[0]
        
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
    
    def forward(self, data, heart_name):
        # erase all heart signal
        mu = self.encode(data, heart_name)
        # z = self.reparameterize(mu, logvar)
        z = self.inverse(mu, heart_name)
        return self.decode(z, heart_name), torch.zeros_like(mu), torch.zeros_like(mu)


class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_seq,
                 out_seq,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        self.gcn = SplineSample(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0], norm=False)

        if process == 'e':
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        elif process == 'd':
            self.tcn = nn.Sequential(
                nn.ConvTranspose2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        else:
            raise NotImplementedError
        
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )

    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.gcn(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = self.tcn(x)
        return x.permute(0, 3, 2, 1).contiguous()


def expand(batch_size, num_nodes, T, edge_index, edge_attr, sample_rate=None):
    # edge_attr = edge_attr.repeat(T, 1)
    num_edges = int(edge_index.shape[1] / batch_size)
    edge_index = edge_index[:, 0:num_edges]
    edge_attr = edge_attr[0:num_edges, :]
    

    sample_number = int(sample_rate * num_edges) if sample_rate is not None else num_edges
    selected_edges = torch.zeros(edge_index.shape[0], batch_size * T * sample_number).to(device)
    selected_attrs = torch.zeros(batch_size * T * sample_number, edge_attr.shape[1]).to(device)

    for i in range(batch_size * T):
        chunk = edge_index + num_nodes * i
        if sample_rate is not None:
            index = np.random.choice(num_edges, sample_number, replace=False)
            index = np.sort(index)
        else:
            index = np.arange(num_edges)
        selected_edges[:, sample_number * i:sample_number * (i + 1)] = chunk[:, index]
        selected_attrs[sample_number * i:sample_number * (i + 1), :] = edge_attr[index, :]

    selected_edges = selected_edges.long()
    return selected_edges, selected_attrs


def loss_stgcnn(recon_x, x, mu, logvar, *args):
    """ VAE Loss: Reconstruction + KL divergence losses summed over all elements and batch
    """
    batch_size = args[0]
    seq_len = args[1]
    epoch = args[2]
    anneal = args[3]

    # if anneal:
    #     if epoch < 50:
    #         anneal_param = 0
    #     elif epoch < 500:
    #         anneal_param = epoch / 500
    #     else:
    #         anneal_param = 1
    # else:
    #     anneal_param = 1
    shape1 = np.prod(x.shape)
    shape2 = np.prod(mu.shape)

    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * anneal * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    BCE = BCE / shape1
    KLD = KLD / shape2

    return BCE + KLD, BCE, KLD


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


def loss_bottleneck(mu_theta, logvar_theta, x, mu, logvar, *args):
    batch_size = args[0]
    seq_len = args[1]
    epoch = args[2]
    anneal = args[3]

    if anneal:
        if epoch < 50:
            anneal_param = 0
        elif epoch < 500:
            anneal_param = epoch / 500
        else:
            anneal_param = 1
    else:
        anneal_param = 1

    shape1 = np.prod(x.shape)

    diffSq = (x - mu_theta).pow(2)
    precis = torch.exp(-logvar_theta)

    BCE = 0.5 * torch.sum(logvar_theta + torch.mul(diffSq, precis))
    BCE /= shape1

    KLD = -0.5 * anneal_param * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= shape1

    return BCE + anneal * KLD, BCE, KLD


def load_graph(filename, heart_torso=0):
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

        if heart_torso == 1 or heart_torso == 2:
            t_g = pickle.load(f)
            t_g1 = pickle.load(f)
            t_g2 = pickle.load(f)
            t_g3 = pickle.load(f)

            t_P10 = pickle.load(f)
            t_P21 = pickle.load(f)
            t_P32 = pickle.load(f)

            H_inv = pickle.load(f)
            P = pickle.load(f)

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
    elif heart_torso == 1 or heart_torso == 2:
        t_P01 = t_P10 / t_P10.sum(axis=0)
        t_P12 = t_P21 / t_P21.sum(axis=0)
        t_P23 = t_P32 / t_P32.sum(axis=0)

        t_P01 = torch.from_numpy(np.transpose(t_P01)).float()
        t_P12 = torch.from_numpy(np.transpose(t_P12)).float()
        t_P23 = torch.from_numpy(np.transpose(t_P23)).float()

        P = torch.from_numpy(P).float()

        # t_P10 = torch.from_numpy(t_P10).float()
        # t_P21 = torch.from_numpy(t_P21).float()
        # t_P32 = torch.from_numpy(t_P32).float()

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()
        # P54 = torch.from_numpy(P54).float()
        # P65 = torch.from_numpy(P65).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43,\
            t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, H_inv, P


def get_graphparams(filename, device, batch_size, heart_torso=0):
    if heart_torso == 0:
        g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34 = \
            load_graph(filename, heart_torso)
    else:
        g, g1, g2, g3, g4, P10, P21, P32, P43,\
        t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, H_inv, P = load_graph(filename, heart_torso)

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
    elif heart_torso == 1 or heart_torso == 2:
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

        H_inv_dataset = HeartEmptyGraphDataset(mesh_graph=H_inv)
        H_inv_loader = DataLoader(H_inv_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        H_inv_b = next(iter(H_inv_loader))

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

        H_inv_b = H_inv_b.to(device)
        P = P.to(device)

        graphparams = {
            "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4, #"bg5": bg5, "bg6": bg6,
            # "P01": P01, "P12": P12, "P23": P23, "P34": P34, "P45": P45, "P56": P56,
            "P10": P10, "P21": P21, "P32": P32, "P43": P43, #"P54": P54, "P65": P65,
            "num_nodes": num_nodes, "g": g, "bg": bg,
            "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
            "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
            # "t_P10": t_P10, "t_P21": t_P21, "t_P32": t_P32,
            "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
            "H_inv": H_inv_b, "P": P
        }

    return graphparams
