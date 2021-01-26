import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from Spline import SplineSample
import torchdiffeq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
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

        self.glayer = SplineSample(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0], norm=False)
        
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
        x = F.elu(self.glayer(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class GCGRUCell(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        self.xr = SplineSample(in_channels=self.input_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
        
        self.hr = SplineSample(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
        
        self.xz = SplineSample(in_channels=self.input_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
        
        self.hz = SplineSample(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
        
        self.xn = SplineSample(in_channels=self.input_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
        
        self.hn = SplineSample(in_channels=self.hidden_dim,
                                out_channels=self.hidden_dim,
                                dim=dim,
                                kernel_size=self.kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=self.sample_rate)
    
    def forward(self, x, hidden, edge_index, edge_attr):
        r = torch.sigmoid(self.xr(x, edge_index, edge_attr) + self.hr(hidden, edge_index, edge_attr))
        z = torch.sigmoid(self.xz(x, edge_index, edge_attr) + self.hz(hidden, edge_index, edge_attr))
        n = torch.tanh(self.xn(x, edge_index, edge_attr) + r * self.hr(hidden, edge_index, edge_attr))
        h_new = (1 - z) * n + z * hidden
        return h_new
    
    def init_hidden(self, graph_size):
        return torch.zeros(graph_size, self.hidden_dim, device=device)


class ReverseGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dim, is_open_spline=True,
                 degree=1, norm=True, root_weight=True, bias=True, sample_rate=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        self.init_layer = Init(self.hidden_dim, 2 * self.hidden_dim)
        
        self.odefunc = ODE_func_lin(self.hidden_dim, 2 * self.hidden_dim, num_layers=1)
        self.gde_solver = GDE_block(self.odefunc, method='rk4', adjoint=True)

        self.gru_layer = GCGRUCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dim=dim,
            is_open_spline=is_open_spline,
            degree=degree,
            norm=norm,
            root_weight=root_weight,
            bias=bias,
            sample_rate=self.sample_rate
        )
    
    def forward(self, x, edge_index, edge_attr):
        x = x.permute(3, 0, 1, 2).contiguous()
        T, N, V, C = x.size()

        last_h = self.init_layer(x[-1, :, :, :])

        x = x.view(T, N * V, C)
        edge_index, edge_attr = expand(N, V, 1, edge_index, edge_attr)

        for t in reversed(range(T)):
            last_h = last_h.view(N, V, -1)
            last_h = self.gde_solver(last_h, 1, steps=1)
            last_h = last_h.view(N * V, -1)

            h = self.gru_layer(
                    x=x[t, :, :],
                    hidden=last_h,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
            last_h = h
        
        return last_h
    
    # def __init_hidden(self, graph_size):
    #     init_states = self.gru_layer.init_hidden(graph_size)
    #     return init_states


class Init(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super().__init__()
        self.g1 = nn.Conv1d(in_channel, hidden_channel, 1)
        self.g2 = nn.Conv1d(hidden_channel, in_channel, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = F.elu(self.g1(x))
        x = torch.tanh(self.g2(x))
        x = x.permute(0, 2, 1).contiguous()
        return x


class ODE_func_lin(nn.Module):
    def __init__(self, in_channel, hidden_channel, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.in_layer = nn.Conv1d(in_channel, hidden_channel, 1)
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Conv1d(hidden_channel, hidden_channel, 1))
        self.layers = nn.Sequential(*layers)
        self.out_layer = nn.Conv1d(hidden_channel, in_channel, 1)
    
    def forward(self, t, x):
        x = x.permute(0, 2, 1).contiguous()
        x = F.elu(self.in_layer(x))
        x = F.elu(self.layers(x))
        x = torch.tanh(self.out_layer(x))
        x = x.permute(0, 2, 1).contiguous()
        return x


class ODE_func_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, dim, kernel_size, is_open_spline=True,
                 degree=1, norm=True, root_weight=True, bias=True, sample_rate=None):
        super().__init__()
        self.g1 = SplineSample(in_channels=in_channels,
                                out_channels=out_channels,
                                dim=dim,
                                kernel_size=kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=sample_rate)
        self.g2 = SplineSample(in_channels=in_channels,
                                out_channels=out_channels,
                                dim=dim,
                                kernel_size=kernel_size,
                                is_open_spline=is_open_spline,
                                degree=degree,
                                norm=norm,
                                root_weight=root_weight,
                                bias=bias,
                                sample_rate=sample_rate)
        self.g1 = nn.Conv1d(in_channels, 2 * out_channels, 1)
        self.g2 = nn.Conv1d(2 * out_channels, out_channels, 1)
        
        self.edge_index = None
        self.edge_attr = None
        
    def update_graph(self, edge_index, edge_attr):
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, t, x):
        x = x.view(-1, C)
        x = F.elu(self.g1(x, self.edge_index, self.edge_attr))
        x = F.elu(self.g2(x, self.edge_index, self.edge_attr))

        return x


class GDE_block(nn.Module):
    def __init__(self, odefunc, method, rtol=1e-5, atol=1e-7, adjoint=True):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint = adjoint
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, T, steps=1):
        if steps == 1:
            self.integration_time = torch.Tensor([0, 1]).float().to(device)
        else:
            self.integration_time = torch.linspace(0, T-1, steps=T).to(device)

        if type(x) == tuple:
            (x, edge_index, edge_attr) = x
        N, V, C = x.shape
        # edge_index, edge_attr = expand(N, V, 1, edge_index, edge_attr)
        # self.odefunc.update_graph(edge_index, edge_attr)

        x = x.contiguous()

        if self.adjoint:
            x = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            x = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        if steps == 1:
            x = x[-1, :, :]
        else:
            x = x.view(T, N, V, C)
            x = x.permute(1, 3, 2, 0).contiguous()
        return x


class ODERNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dim, is_open_spline=True,
                 degree=1, norm=True, root_weight=True, bias=True, sample_rate=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        self.odefunc = ODE_func_lin(self.hidden_dim, 2 * self.hidden_dim, num_layers=1)
        self.gde_solver = GDE_block(self.odefunc, method='rk4', adjoint=True)

        self.gru_layer = GCGRUCell(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            dim=dim,
            is_open_spline=is_open_spline,
            degree=degree,
            norm=norm,
            root_weight=root_weight,
            bias=bias,
            sample_rate=self.sample_rate
        )

    def forward(self, x, edge_index, edge_attr):
        gru_out = []
        ode_out = []

        x = x.permute(3, 0, 1, 2).contiguous()
        T, N, V, C = x.shape
        edge_index, edge_attr = expand(N, V, 1, edge_index, edge_attr)

        last_h = x[0]
        gru_out.append(last_h.view(1, N, V, C))
        ode_out.append(last_h.view(1, N, V, C))
        x = x.view(T, N * V, C)

        for t in range(1, T):
            last_h = last_h.view(N, V, -1)
            last_h = self.gde_solver(last_h, 1, steps=1)
            last_h = last_h.view(N * V, -1)
            ode_out.append(last_h.view(1, N, V, C))

            h = self.gru_layer(
                x=x[t, :, :],
                hidden=last_h,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            last_h = h
            gru_out.append(h.view(1, N, V, C))

        gru_out = torch.cat(gru_out, dim=0)
        ode_out = torch.cat(ode_out, dim=0)
        gru_out = gru_out.permute(1, 2, 3, 0).contiguous()
        ode_out = ode_out.permute(1, 2, 3, 0).contiguous()
        return gru_out, ode_out


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
