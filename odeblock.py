"""
@file odeblock.py
@author Ryan Missel

Holds the code for the GDE Function and ODE Block used as part of the ODE-RNN of the latent space
"""
import dgl
import torch
import torch.nn as nn
import torchdiffeq


class GDEFunc(nn.Module):
    def __init__(self, gnn: nn.Module):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0

    def set_graph(self, g: dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g

    def forward(self, t, x):
        self.nfe += 1
        x = self.gnn(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, method: str = 'dopri5', rtol: float = 1e-3, atol: float = 1e-4,
                 adjoint: bool = True):
        """ Standard ODEBlock class. Can handle all types of ODE functions
            :method:str = {'euler', 'rk4', 'dopri5', 'adams'}
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.atol, self.rtol = atol, rtol

    def forward(self, x: torch.Tensor, T: int = 1):
        self.integration_time = torch.tensor([0, T]).float()
        self.integration_time = self.integration_time.type_as(x)

        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        return out[-1]

    def forward_batched(self, x: torch.Tensor, nn: int, indices: list, timestamps: set):
        """ Modified forward for ODE batches with different integration times """
        timestamps = torch.Tensor(list(timestamps))
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, timestamps,
                                             rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(self.odefunc, x, timestamps,
                                     rtol=self.rtol, atol=self.atol, method=self.method)

        out = self._build_batch(out, nn, indices).reshape(x.shape)
        return out

    def _build_batch(self, odeout, nn, indices):
        b_out = []
        for i in range(len(indices)):
            b_out.append(odeout[indices[i], i * nn:(i + 1) * nn])
        return torch.cat(b_out).to(odeout.device)

    def trajectory(self, x: torch.Tensor, T: int, num_points: int):
        self.integration_time = torch.linspace(0, T, num_points)
        self.integration_time = self.integration_time.type_as(x)
        out = torchdiffeq.odeint(self.odefunc, x, self.integration_time,
                                 rtol=self.rtol, atol=self.atol, method=self.method)
        return out
