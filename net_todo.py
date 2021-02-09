class Graph_LODE(nn.Module):
    def __init__(self, hparams, training=True):
        super().__init__()
        self.graph_method = hparams.graph_method
        self.nf = hparams.nf
        self.batch_size = hparams.batch_size if training else 1
        self.seq_len = hparams.seq_len
        self.latent_dim = hparams.latent_dim

        self.conv1 = gcn(self.nf[0], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = gcn(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = gcn(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[6], 1)
        self.fce2 = nn.Conv2d(self.nf[6], self.latent_dim, 1)
        self.gru = ReverseGRU(input_dim=self.latent_dim, hidden_dim=self.latent_dim, kernel_size=3, dim=3, norm=False)
        self.fc_mu = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.fc_logvar = nn.Conv1d(self.latent_dim, self.latent_dim, 1)

        self.trans = SplineSample(self.latent_dim, self.latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)
        
        self.ode_layer = ODE_func_lin(self.latent_dim, 2 * self.latent_dim, num_layers=1)
        self.ode_solver = ODE_block(self.ode_layer, 'conv', method='rk4', adjoint=True)

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

        if self.graph_method == 'graclus_hier':
            self.H_inv = dict()
            self.P = dict()
        elif self.graph_method == 'embedding':
            self.H_enc = dict()
            self.P_enc = dict()
            
            self.H_dec = dict()
            self.P_dec = dict()
        else:
            raise NotImplementedError

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

        if self.graph_method == 'graclus_hier':
            self.H_inv[heart_name] = gParams["H"]
            self.P[heart_name] = gParams["P"]
        elif self.graph_method == 'embedding':
            self.H_enc[heart_name] = gParams["H_enc"]
            self.P_enc[heart_name] = gParams["P_enc"]
            
            self.H_dec[heart_name] = gParams["H_dec"]
            self.P_dec[heart_name] = gParams["P_dec"]
        else:
            raise NotImplementedError

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
        x = F.elu(self.fce2(x), inplace=True)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.gru(x, edge_index, edge_attr)
        x = x.view(self.batch_size, -1, self.latent_dim)
        x = x.permute(0, 2, 1).contiguous()

        mu = torch.tanh(self.fc_mu(x))
        logvar = torch.tanh(self.fc_logvar(x))

        mu = mu.permute(0, 2, 1).contiguous()
        logvar = logvar.permute(0, 2, 1).contiguous()
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """ reparameterization; draw a random sample from the p(z|x)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def inverse(self, x, heart_name):
        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        
        num_heart = self.P43[heart_name].shape[1]
        num_torso = self.t_P23[heart_name].shape[0]
        
        x_bin = torch.zeros(self.batch_size, num_heart, self.latent_dim).to(device)
        x_bin = torch.cat((x_bin, x), 1)
        
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(self.batch_size, num_heart + num_torso, 1, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.batch_size, -1, self.latent_dim)
        
        x_bin = x_bin[:, 0:-num_torso, :]
        return x_bin
    
    def decode(self, x, heart_name):
        """ graph  convolutional decoder
        """
        # edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        x = self.ode_solver(x, self.seq_len, steps=self.seq_len)

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(self.batch_size, -1, self.nf[4] * self.seq_len)
        x = torch.matmul(self.P43[heart_name], x)  # bs X 184 X f[4]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[4], self.seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)  # (bs*184) X f[3]

        x = x.view(self.batch_size, -1, self.nf[3] * self.seq_len)
        x = torch.matmul(self.P32[heart_name], x)  # bs X 351 X f[3]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[3], self.seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)  # (bs*351) X f[2]

        x = x.view(self.batch_size, -1, self.nf[2] * self.seq_len)
        x = torch.matmul(self.P21[heart_name], x)  # bs X 646 X f[2]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[2], self.seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)  # (bs*646) X f[1]

        x = x.view(self.batch_size, -1, self.nf[1] * self.seq_len)
        x = torch.matmul(self.P10[heart_name], x)  # bs X 1230 X f[1]
        x, edge_index, edge_attr = \
            x.view(self.batch_size, -1, self.nf[1], self.seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
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
        mu, logvar = self.encode(phi_t, heart_name)
        z = self.reparameterize(mu, logvar)
        z = self.inverse(z, heart_name)
        phi_h = self.decode(z, heart_name)
        phi_t_, l_h = self.physics(phi_t, phi_h, heart_name)
        return phi_h, phi_t_, l_h, mu, logvar, torch.zeros_like(mu), torch.zeros_like(mu)