import torch
import numpy as np
import numbers
import itertools
from torch import nn
from torch_geometric.nn.inits import uniform
# from torch_spline_conv import SplineBasis, SplineWeighting
from torch_spline_conv import spline_basis, spline_weighting


def repeat(src, length):
    if isinstance(src, numbers.Number):
        src = list(itertools.repeat(src, length))
    return src


def node_degree(index, num_nodes=None, dtype=None, device=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


class Spline(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.norm = norm

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, pseudo):
        if edge_index.numel() == 0:
            out = torch.mm(x, self.root)
            out = out + self.bias
            return out

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        row, col = edge_index
        n, m_out = x.size(0), self.weight.size(2)

        # Weight each node.
        basis, weight_index = spline_basis(pseudo, self._buffers['kernel_size'],
                                                self._buffers['is_open_spline'], self.degree)
        weight_index = weight_index.detach()
        out = spline_weighting(x[col], self.weight, basis, weight_index)

        # Convert e x m_out to n x m_out features.
        row_expand = row.unsqueeze(-1).expand_as(out)
        out = x.new_zeros((n, m_out)).scatter_add_(0, row_expand, out)

        # Normalize out by node degree (if wished).
        if self.norm:
            deg = node_degree(row, n, out.dtype, out.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SplineSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degree = degree
        self.norm = norm
        self.sample_rate = sample_rate

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer('kernel_size', kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer('is_open_spline', is_open_spline)

        K = kernel_size.prod().item()
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    
    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.root)
        uniform(size, self.bias)
    
    def sample(self, edge_index, pseudo):
        num_edge = edge_index.shape[1]
        index = np.random.choice(num_edge, int(self.sample_rate * num_edge), replace=False)
        index = np.sort(index)
        edge_index_sample = edge_index[:, index]
        pseudo_sample = pseudo[index]
        return edge_index_sample, pseudo_sample

    def forward(self, x, edge_index, pseudo):
        if edge_index.numel() == 0:
            out = torch.mm(x, self.root)
            out = out + self.bias
            return out

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        if self.sample_rate is not None:
            edge_index, pseudo = self.sample(edge_index, pseudo)

        row, col = edge_index
        n, m_out = x.size(0), self.weight.size(2)

        # Weight each node.
        basis, weight_index = spline_basis(pseudo, self._buffers['kernel_size'],
                                                self._buffers['is_open_spline'], self.degree)
        weight_index = weight_index.detach()
        out = spline_weighting(x[col], self.weight, basis, weight_index)

        # Convert e x m_out to n x m_out features.
        row_expand = row.unsqueeze(-1).expand_as(out)
        out = x.new_zeros((n, m_out)).scatter_add_(0, row_expand, out)

        # Normalize out by node degree (if wished).
        if self.norm:
            deg = node_degree(row, n, out.dtype, out.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        # Add bias (if wished).
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
