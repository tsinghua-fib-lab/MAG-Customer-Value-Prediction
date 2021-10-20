import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from inits import *
from utils import *

import sys


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None):

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha.view(self.heads, -1), edge_index_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class gated_influence(torch.nn.Module):
    def __init__(self, ego_size, influence_size, hidden_size, output_size):
        super(gated_influence, self).__init__()
        self.ego_size = ego_size
        self.influence_size = influence_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.w_er= Parameter(torch.Tensor(self.ego_size, self.hidden_size))
        self.w_ir= Parameter(torch.Tensor(self.influence_size, self.hidden_size))
        self.w_ez= Parameter(torch.Tensor(self.ego_size, self.hidden_size))
        self.w_iz= Parameter(torch.Tensor(self.influence_size, self.hidden_size))
        self.w_en= Parameter(torch.Tensor(self.ego_size, self.hidden_size))
        self.w_in= Parameter(torch.Tensor(self.influence_size, self.hidden_size))
        
        self.b_er= Parameter(torch.Tensor(self.hidden_size))
        self.b_ir= Parameter(torch.Tensor(self.hidden_size))
        self.b_ez= Parameter(torch.Tensor(self.hidden_size))
        self.b_iz= Parameter(torch.Tensor(self.hidden_size))
        self.b_en= Parameter(torch.Tensor(self.hidden_size))
        self.b_in= Parameter(torch.Tensor(self.hidden_size))
        

        self.reset_parameters()
    def reset_parameters(self):
        glorot(self.w_er)
        glorot(self.w_ir)
        glorot(self.w_ez)
        glorot(self.w_iz)
        glorot(self.w_en)
        glorot(self.w_in)

        zeros(self.b_er)
        zeros(self.b_ir)
        zeros(self.b_ez)
        zeros(self.b_iz)
        zeros(self.b_en)
        zeros(self.b_in)
    
    def forward(self, X_ego, X_inf):
        r = F.sigmoid(torch.matmul(X_ego, self.w_er) + torch.matmul(X_inf, self.w_ir) + self.b_er + self.b_ir)
        z = F.sigmoid(torch.matmul(X_ego, self.w_ez) + torch.matmul(X_inf, self.w_iz) + self.b_ez + self.b_iz)
        n = torch.tanh(torch.matmul(X_inf, self.w_in) + self.b_in + torch.mul(r, (torch.matmul(X_ego, self.w_en)+self.b_en)))
        #z = F.sigmoid(torch.matmul(X_ego, self.w_ez) + self.b_ez )
        y = torch.mul((1-z),n) + torch.mul(z, X_ego)
        return y
    
    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                             self.ego_size,
                                             self.influence_size, 
                                             self.hidden_size) 
        