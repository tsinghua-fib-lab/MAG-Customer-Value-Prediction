import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
import numpy as np
import sys
from layers import *
from torch_geometric.nn import Node2Vec


class GCN_motif_gru(torch.nn.Module):
    def __init__(self,args):
        super(GCN_motif_gru, self).__init__()

        self.args = args
        self.num_features = args.num_features
        self.n_hidden = args.n_hidden
        self.dropout_ratio = args.dropout_ratio
        self.n_output = args.enh
        self.n_gcn = args.gnh
        
        self.n_demo = 85
        self.n_purchase = 16


        
        self.embedding1 = torch.nn.Linear(self.n_demo, self.n_output)
        self.embedding2 = torch.nn.Linear(self.n_purchase, self.n_output)
    

        self.embedding3 = torch.nn.Linear(2 * self.n_output, self.n_gcn)

        #gef
        self.gef = gated_influence(self.n_gcn,self.n_gcn,1,self.n_gcn)

        #tri:motif
        self.tri0 = GCNConv(self.n_gcn, self.n_gcn)
        self.tri1 = GCNConv(self.n_gcn, self.n_gcn)

        #attention
        self.att1 = torch.nn.Linear(self.n_gcn, self.n_gcn)
        self.attn_weight = Parameter(torch.Tensor(self.n_gcn, 1))
        glorot(self.attn_weight)

        self.lin1 = torch.nn.Linear(self.n_gcn, self.n_hidden)
        self.lin3 = torch.nn.Linear(self.n_hidden, int(self.n_hidden/2))
        self.lin2 = torch.nn.Linear(int(self.n_hidden/4), 1)
        self.lin4 = torch.nn.Linear(int(self.n_hidden/2), int(self.n_hidden/4))


    def attetion_layer(self, x):
        return torch.matmul(torch.tanh(self.att1(x)),self.attn_weight)
    def forward(self, data):
        x, edge_index, batch_ids = data.x, data.edge_index, data.batch_ids

        edge_index_tri_0, edge_weight_tri_0  = data.motif_triangle_0, data.motif_triangle_weight_0
        edge_index_tri_1, edge_weight_tri_1  = data.motif_triangle_1, data.motif_triangle_weight_1
        edge_index_tri_2, edge_weight_tri_2  = data.motif_triangle_2, data.motif_triangle_weight_2
        edge_index_tri_3, edge_weight_tri_3  = data.motif_triangle_3, data.motif_triangle_weight_3
        edge_index_tri_4, edge_weight_tri_4  = data.motif_triangle_4, data.motif_triangle_weight_4
        edge_index_tri_5, edge_weight_tri_5  = data.motif_triangle_5, data.motif_triangle_weight_5
        edge_index_tri_6, edge_weight_tri_6  = data.motif_triangle_6, data.motif_triangle_weight_6
        edge_index_tri_7, edge_weight_tri_7  = data.motif_triangle_7, data.motif_triangle_weight_7
        edge_index_tri_8, edge_weight_tri_8  = data.motif_triangle_8, data.motif_triangle_weight_8
        edge_index_tri_9, edge_weight_tri_9  = data.motif_triangle_9, data.motif_triangle_weight_9
        edge_index_tri_10, edge_weight_tri_10  = data.motif_triangle_10, data.motif_triangle_weight_10
        edge_index_tri_11, edge_weight_tri_11  = data.motif_triangle_11, data.motif_triangle_weight_11
        edge_index_tri_12, edge_weight_tri_12  = data.motif_triangle_12, data.motif_triangle_weight_12

        
        x1 = F.relu(self.embedding1(x[:, 0: 85]))
        x2 = F.relu(self.embedding2(x[:, 85:101]))


        x = torch.cat([x1, x2], 1)
        x = F.relu(self.embedding3(x))

        x_norm = F.relu(self.tri0(x, edge_index))
        x_norm = F.relu(self.tri1(x_norm, edge_index))
        x_norm = x_norm[batch_ids,:]
 
        x_tri_0 =  F.relu(self.tri0(x, edge_index_tri_0))
        x_tri_0 = F.relu(self.tri1(x_tri_0, edge_index_tri_0))
        x_tri_0 = x_tri_0[batch_ids,:]

        x_tri_1 =  F.relu(self.tri0(x, edge_index_tri_1))
        x_tri_1 = F.relu(self.tri1(x_tri_1, edge_index_tri_1))
        x_tri_1 = x_tri_1[batch_ids,:]

        x_tri_2 =  F.relu(self.tri0(x, edge_index_tri_2))
        x_tri_2 = F.relu(self.tri1(x_tri_2, edge_index_tri_2))
        x_tri_2 = x_tri_2[batch_ids,:]

        x_tri_3 =  F.relu(self.tri0(x, edge_index_tri_3))
        x_tri_3 = F.relu(self.tri1(x_tri_3, edge_index_tri_3))
        x_tri_3 = x_tri_3[batch_ids,:]

        x_tri_4 =  F.relu(self.tri0(x, edge_index_tri_4))
        x_tri_4 = F.relu(self.tri1(x_tri_4, edge_index_tri_4))
        x_tri_4 = x_tri_4[batch_ids,:]

        x_tri_5 =  F.relu(self.tri0(x, edge_index_tri_5))
        x_tri_5 = F.relu(self.tri1(x_tri_5, edge_index_tri_5))
        x_tri_5 = x_tri_5[batch_ids,:]

        x_tri_6 =  F.relu(self.tri0(x, edge_index_tri_6))
        x_tri_6 = F.relu(self.tri1(x_tri_6, edge_index_tri_6))
        x_tri_6 = x_tri_6[batch_ids,:]

        x_tri_7 =  F.relu(self.tri0(x, edge_index_tri_7))
        x_tri_7 = F.relu(self.tri1(x_tri_7, edge_index_tri_7))
        x_tri_7 = x_tri_7[batch_ids,:]
        

        x_tri_8 =  F.relu(self.tri0(x, edge_index_tri_8))
        x_tri_8 = F.relu(self.tri1(x_tri_8, edge_index_tri_8))
        x_tri_8 = x_tri_8[batch_ids,:]

        x_tri_9 =  F.relu(self.tri0(x, edge_index_tri_9))
        x_tri_9 = F.relu(self.tri1(x_tri_9, edge_index_tri_9))
        x_tri_9 = x_tri_9[batch_ids,:]

        x_tri_10 =  F.relu(self.tri0(x, edge_index_tri_10))
        x_tri_10 = F.relu(self.tri1(x_tri_10, edge_index_tri_10))
        x_tri_10 = x_tri_10[batch_ids,:]
        
        x_tri_11=  F.relu(self.tri0(x, edge_index_tri_11))
        x_tri_11 = F.relu(self.tri1(x_tri_11, edge_index_tri_11))
        x_tri_11 = x_tri_11[batch_ids,:]

        x_tri_12=  F.relu(self.tri0(x, edge_index_tri_12))
        x_tri_12 = F.relu(self.tri1(x_tri_12, edge_index_tri_12))
        x_tri_12 = x_tri_12[batch_ids,:]

        atten_co = torch.cat([self.attetion_layer(x_norm) ,self.attetion_layer(x_tri_0),self.attetion_layer(x_tri_1), self.attetion_layer(x_tri_2), self.attetion_layer(x_tri_3), self.attetion_layer(x_tri_4), \
            self.attetion_layer(x_tri_5), self.attetion_layer(x_tri_6), self.attetion_layer(x_tri_7), self.attetion_layer(x_tri_8), self.attetion_layer(x_tri_9), self.attetion_layer(x_tri_10), \
            self.attetion_layer(x_tri_11), self.attetion_layer(x_tri_12)], 1)

        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
 
        x_graph = torch.stack([x_norm, x_tri_0, x_tri_1, x_tri_2, x_tri_3, x_tri_4, x_tri_5, \
            x_tri_6, x_tri_7, x_tri_8, x_tri_9, x_tri_10, x_tri_11, x_tri_12], dim=1)
        x_graph = atten_co * x_graph


        x_graph = torch.sum(x_graph, dim=1)
        x = x[batch_ids,:]
        x = self.gef(x, x_graph)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin2(x)).squeeze()

        
        return x




