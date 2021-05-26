from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential, ReLU, Linear, Dropout, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import weight_norm

from torch_geometric.data import Data, Batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool, GCN2Conv
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, SGConv,TransformerConv
# ResGatedGraphConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, SAGEConv

from performer_pytorch import Performer

import math


def get_pe(pos, dim):
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(dim)
    pe[0::2] = torch.sin(pos * div_term)
    pe[1::2] = torch.cos(pos * div_term)
    return pe 


def add_weight_decay(pars, l2_value):
    decay, no_decay = [], []
    for name, param in pars:
        if len(param.shape) == 1 or name.endswith(".bias") or torch.tensor(param.shape).sum()<3: 
            no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

class FeatureTransform(torch.nn.Module):
    def __init__(self, num_features, nn1 , nn2 = None):
        super().__init__()      
        lin1 = nn.Linear(num_features,nn1)
        lin1.bias.data.fill_(0.)  
        bn1 = BatchNorm1d(nn1)
        self.encoder = Sequential(lin1, ReLU(),bn1)
        self.dropout = Dropout(0.2)
        
    def forward(self, x):
#         self.dropout(x)
        z = self.encoder(x)
        return z

class TrEnc(nn.Module):
    def __init__(self, ndim, n_out):
        super().__init__()
        self.transformer_encoder = Performer(dim = n_out, depth = 1, heads = 8, dim_head = n_out//8)

    def forward(self, x):
        out = self.transformer_encoder(x.unsqueeze(0))
        out = out.squeeze()   
        return out


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.lins = torch.nn.ModuleList()
#         for _ in range(K + 1):
#             self.lins.append(Linear(dataset.num_node_features, 1024))
#         self.lin = Linear((K + 1) * 1024, dataset.num_classes)

#     def forward(self, xs):
#         hs = []
#         for x, lin in zip(xs, self.lins):
#             h = lin(x).relu()
#             h = F.dropout(h, p=0.5, training=self.training)
#             hs.append(h)
#         h = torch.cat(hs, dim=-1)
#         h = self.lin(h)
#         return h.log_softmax(dim=-1)

class GraphNet(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
#         nn1 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv1 = GINConv(nn1)
#         self.bn1 = torch.nn.BatchNorm1d(dim)
#         nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
#         self.conv2 = GINConv(nn2)
#         self.bn2 = torch.nn.BatchNorm1d(dim)

        # self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,dropout=0.1)

        self.conv_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(arch['nlayers_gnn']):
            if arch['conv_type'] == 'sage':
                self.conv_layers.append(GCNConv(arch['nd'], arch['nd'], 
                        aggr=arch['node_agg_type']))
            elif arch['conv_type'] == 'sg':
                self.conv_layers.append(SGConv(arch['nd'], arch['nd'], 
                        aggr=arch['node_agg_type'],K=2, add_self_loops =False))
            elif arch['conv_type'] == 'resgate':
                self.conv_layers.append(TransformerConv(arch['nd'], arch['nd']))
            elif arch['conv_type'] == 'gin':
                nn1 = Linear(arch['nd'], arch['nd'])
                self.conv_layers.append(GINConv(nn1, 
                        aggr=arch['node_agg_type']))    
            elif arch['conv_type'] == 'gat':
                self.conv_layers.append(GATConv(arch['nd'], arch['nd'], heads = 8, concat =False,
                        aggr=arch['node_agg_type'])) 
                
            self.batch_norms.append(BatchNorm(arch['nd']))
#             elif arch['conv_type'] == 'pna':
#                 aggregators = ['mean', 'min', 'max', 'std']
#                 scalers = ['identity', 'amplification', 'attenuation']
#                 self.conv_layers.append(PNAConv(in_channels=arch['nd'], out_channels=arch['nd'], 
#                         aggregators=aggregators, scalers=scalers, deg=deg,
#                            towers=5, pre_layers=1, post_layers=1)             
        # self.gnn = nn.Sequential(*conv_layers)

    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
#         for conv in self.conv_layers:
#                 x = F.relu(conv(x, edge_index))
        for conv, batch_norm in zip(self.conv_layers, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        return x

class GraphAgg(torch.nn.Module):
    def __init__(self, dim_out, agg_type):
        super().__init__()
        self.lin_g_agg = weight_norm(nn.Linear(dim_out, 1))
        self.lin2 = weight_norm(nn.Linear(dim_out, dim_out))
        self.agg = agg_type
        self.dim_out = dim_out

    def forward(self, x, mask):
        
        z = self.lin_g_agg(x).squeeze(2)
        print(z.shape, mask.shape)

        z[~mask] = float('-inf')
        rz = torch.sigmoid(z)
        rz = x * rz.unsqueeze(2)    
        n_nonzero = mask.sum(dim=1)

        if self.agg == 'mean':
            rz = []
            for dim in range(x.size(0)):
                rz.append(x[dim][mask[dim]].sum(dim=0) / mask[dim].sum() )
            rz = torch.stack(rz, dim =0)

        elif self.agg == 'mean_gate':
            rz = rz.sum(dim=1) / n_nonzero.unsqueeze(1)
        elif self.agg == 'sum_gate':
            rz = rz.sum(dim=1)
        elif self.agg == 'max':
            rz = []
            for dim in range(x.size(0)):
                rz.append(x[dim][mask[dim]].max(dim=0)[0] )
            rz = torch.stack(rz, dim =0)
        elif self.agg == 'softmax':
            rz = (x * (z/math.sqrt(self.dim_out))\
                    .softmax(dim=1).unsqueeze(2)).sum(dim=1)
        return rz
    

class Vects2Edge(nn.Module):
    def __init__(self, ndim, p0 = None, drop_rate = 0):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.lin = weight_norm(nn.Linear(ndim, 1, bias=True))
#         if p0 is not None:
#             b0 = torch.log(p0/(1.-p0)) 
#             self.lin.bias.data.fill_(b0 + self.lin.bias.data[0])
        self.lin0_ve= weight_norm(nn.Linear(ndim, 1))
        self.lin0_ve.bias.data.fill_(0)
#         self.lin0.weight.data.fill_(0)
        
    def forward(self,  v_new, v_old):
#         z =  torch.abs(v_new-v_old) \
#                   .mean(axis=1).unsqueeze(1)
        z =  torch.abs(v_new-v_old) 
        z = self.lin0_ve(z)    
#         print(z.shape,z)
        return z.squeeze()


class ClassDiff(nn.Module):
    def __init__(self, nd, f_type, w0 = None, b0 = 0):
        super().__init__()
        # self.dropout = nn.Dropout(drop_rate)
        self.fin_type = f_type
        self.pre_transform  = Sequential(Linear(nd, nd), ReLU())

        self.lin_fin_n  = weight_norm(nn.Linear(nd, 1))
        self.lin_fin_n.bias.data.fill_(b0)
        if w0 is not None:
            self.lin_fin_n.weight.data.fill_(w0)
            
        self.lin_fin_n_d = weight_norm(nn.Linear(nd, 1))
        self.lin_fin_n_d.bias.data.fill_(b0)
        if w0 is not None:
            self.lin_fin_n_d.weight.data.fill_(w0)
            
        self.lin_fin_0 = weight_norm(nn.Linear(2, 1))
        self.lin_fin_0.bias.data.fill_(b0)
        if w0 is not None:
            self.lin_fin_0.weight.data.fill_(w0)
            

        
    def forward(self,  v1, v2):
        # print(v1.shape, ba1.shape)
        if self.fin_type == 'cos':
            v1 = F.normalize(self.pre_transform(v1), dim=len(v1.shape)-1)
            v2 = F.normalize(self.pre_transform(v2), dim=len(v2.shape)-1)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
        elif self.fin_type == 'mm_sig':
            v1 = self.pre_transform(v1)
            v2 = self.pre_transform(v2)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
            z = torch.sigmoid(self.lin_fin_0(z))
        elif self.fin_type == 'mm_cos':
            v1 = F.normalize(F.relu(v1), dim=len(v1.shape)-1)
            v2 = F.normalize(F.relu(v2), dim=len(v2.shape)-1)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
        elif self.fin_type == 'diff_mlp':
            # print('v1', v1.shape, 'v2', v2.shape)
            z =  torch.abs(v2-v1) 
            # print(z.shape, torch.repeat_interleave)
            # z =  torch.cat((z, ba1.unsqueeze(1),ba2.unsqueeze(1)), dim = 1)
            z = self.lin_fin_n(z)
            # print(z.shape)
            # .view(-1,1)
            z = torch.sigmoid(z)
        elif self.fin_type == 'diff_sum':
            z =  torch.abs(v1-v2) 
            z = self.lin_fin_0(z.sum(dim=len(z.shape)-1).view(-1,1))
            z = torch.sigmoid(z)

        elif self.fin_type == 'het':
            z =  torch.abs(v1-v2) 
            nm = z.sum(dim=len(z.shape)-1).unsqueeze(len(z.shape)-1)
            # .view(-1,1)
            z = self.lin_fin_0(torch.cat((nm,1./(nm+1e-12)), dim=(len(nm.shape)-1)))
            z = torch.sigmoid(z)

        elif self.fin_type == 'stack':
            z = self.lin_fin_n(v1) + self.lin_fin_n_d(v2)
            z = torch.sigmoid(z).view(-1,1)
            
        return z.squeeze()




class EdgeDecision(nn.Module):
    def __init__(self, nd, f_type, w0 = None, b0 = 0):
        super().__init__()
        # self.dropout = nn.Dropout(drop_rate)
        self.fin_type = f_type
        self.gr_transform  = Sequential(Linear(nd, nd), ReLU())
        self.v_transform  = Sequential(Linear(nd, nd), ReLU())

        self.lin_fin_n  = weight_norm(nn.Linear(nd, 1))
        self.lin_fin_n.bias.data.fill_(b0)
        if w0 is not None:
            self.lin_fin_n.weight.data.fill_(w0)

        self.lin_fin_0 = weight_norm(nn.Linear(1, 1))
        self.lin_fin_0.bias.data.fill_(b0)
        if w0 is not None:
            self.lin_fin_0.weight.data.fill_(w0)
        
    def forward(self,  v_gr, v_in, v_new):
        rz = self.gr_transform(v_gr) + self.v_transform(v_in)

        if self.fin_type == 'cos':
            v1 = F.normalize(self.pre_transform(v1), dim=len(v1.shape)-1)
            v2 = F.normalize(self.pre_transform(v2), dim=len(v2.shape)-1)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
        elif self.fin_type == 'mm_sig':
            v1 = self.pre_transform(v1)
            v2 = self.pre_transform(v2)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
            z = torch.sigmoid(self.lin_fin_0(z))
        elif self.fin_type == 'mm_cos':
            v1 = F.normalize(F.relu(v1), dim=len(v1.shape)-1)
            v2 = F.normalize(F.relu(v2), dim=len(v2.shape)-1)
            z = v1 * v2
            z = z.sum(dim=len(z.shape)-1).view(-1,1)
        elif self.fin_type == 'diff_mlp':
            z =  torch.abs(v2-v1) 
            z = self.lin_fin_n(z).view(-1,1)
            z = torch.sigmoid(z)
        elif self.fin_type == 'diff_sum':
            z =  torch.abs(v1-v2) 
            z = self.lin_fin_0(z.sum(dim=len(z.shape)-1).view(-1,1))
            z = torch.sigmoid(z)
        return z.squeeze()




class NodeExpand(nn.Module):
    def __init__(self, nd, w0 = None, b0 = 0):
        super().__init__()

        self.lin = weight_norm(nn.Linear(nd, 1))
        self.lin.bias.data.fill_(b0)
        if w0 is not None:
            self.lin.weight.data.fill_(w0)
        
    def forward(self, v):
        return torch.sigmoid(self.lin(v)).squeeze()

class Vects2Choice(nn.Module):
    def __init__(self, ndim, drop_rate = 0):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)     
        self.lin0_vc= nn.Linear(1, 1)
        self.lin0_vc.bias.data.fill_(0)
        self.lin0_vc.weight.data.fill_(0)
        # proverit inicializaciyu !!!!!!!!!!!!!!!!!!1       
#         b0 = torch.log(p0/(1.-p0)) 
#         self.lin0.bias.data.fill_(b0 + self.lin0.bias.data[0])
        
    def forward(self,  v_graph, v_choice):        
#         print(torch.abs(v_choice-v_graph).shape)  
        z =  torch.mm(v_graph, v_choice.T).view(-1,1)
        z = self.lin0_vc(z)
#         print(z.shape, z)
#         print(z)
        return z.squeeze()
        
