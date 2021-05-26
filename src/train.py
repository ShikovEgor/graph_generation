import numpy as np
import pandas as pd
import os.path as osp
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from torch.nn import Sequential, ReLU, Linear, Dropout, BatchNorm1d
from torch.nn.utils import weight_norm

import networkx as nx

from src.models import TrEnc, GraphNet, Vects2Edge, Vects2Choice, GraphAgg,add_weight_decay,FeatureTransform
from src.models import  ClassDiff

from src.radam import RAdam
from src.data_loading import data_gen, get_cora

def cal_edge_loss(pred, lbs):
    # print(pred)
    predd = torch.stack((1-pred, pred)).T
    predd = torch.clamp(torch.log(predd+1e-20),max= 0)
    eps = 0.05
    with torch.no_grad():
        one_hot = torch.zeros_like(predd).scatter(1, lbs.view(-1, 1).to(dtype=torch.long), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps

    ls = F.kl_div(predd, one_hot, reduction='mean')
    return ls

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.log(x)
        # print(logprobs.shape, target.shape)
        nll_loss = -logprobs.gather(dim=-1, index=target.to(dtype = torch.long))
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ModelFull(torch.nn.Module):
    def __init__(self, num_features, device, arch):
        super().__init__()
        
        enc_layers = []
        enc_layers.append(FeatureTransform(num_features, arch['nd']))
        if arch['add_trans']:
            enc_layers.append(TrEnc(arch['nd'], arch['nd']))
        self.encoder = nn.Sequential(*enc_layers)
        self.graph_net = GraphNet(arch)        
        self.v2_edge = ClassDiff(arch['nd'], arch['fin_type_e'], 0)
        self.gr_transform  = Sequential(weight_norm(Linear(arch['nd'], arch['nd'])), ReLU())
        self.v_transform  = Sequential(weight_norm(Linear(arch['nd'], arch['nd'])), ReLU())
        
        self.v2_choice = ClassDiff(arch['nd'], arch['fin_type_c'], 0, 0)
        self.gagg = GraphAgg(arch['nd'], arch['agg_type'])     
        self.device = device
        self.nd = arch['nd']
        self.ls = LabelSmoothing()
        self.label_smooth = arch['lb_smth']

    def iterate(self, data, inp, b_size): 
        encoded = self.encoder(data.x.to(device = self.device))
        return self.decode(data, inp, encoded, b_size)
        
    def decode(self, dat, inp, x_enc, b_size):
        gr, v_1, v_0, e_1, e_0, v_in_gr = inp
        gr, v_1, v_0, e_1, e_0, v_in_gr = gr.to(device = self.device), \
                                    v_1.to(device = self.device), \
                                    v_0.to(device = self.device), \
                                    e_1.to(device = self.device), \
                                    e_0.to(device = self.device), \
                                    v_in_gr.to(device = self.device)
        
#         print(v_1.shape, e_1.shape, v_in_gr.shape, gr.shape)
        
        decoded = self.graph_net( x_enc[torch.cat(b_size*[torch.arange(dat.num_nodes, \
                                                                        device = self.device)]),:], gr)
        
        
        graph_emb = self.gagg(decoded.view(-1,dat.num_nodes, self.nd), v_in_gr)  

        # Select new node  !!!!!!!!!!!
        pred_v =  self.v2_choice(graph_emb.unsqueeze(1), x_enc)
        pred_v = pred_v[torch.cat((v_0,v_1))]
        targ_v = torch.cat((torch.zeros(v_0.shape[0], device = self.device),
                            torch.ones(v_1.shape[0], device = self.device))) 
        # print(pred_v)
        # loss_c = cal_edge_loss(pred_v, targ_v)

        if self.label_smooth:
            loss_c = cal_edge_loss(pred_v, targ_v)
        else:
            loss_c = F.binary_cross_entropy(pred_v, targ_v) #binary_cross_entropy_with_logits

        e_new, e_old = torch.cat((e_0,e_1), dim=1)     
        
         # Select edges  !!!!!!!!!!!
        context = (self.gr_transform(graph_emb).unsqueeze(1)  + \
                   self.v_transform(decoded.view(-1,dat.num_nodes, self.nd))).view(-1,self.nd)
        
#         pred_e = self.v2_edge(x_enc[e_new % self.num_nodes], decoded[e_old])
        
        pred_e = self.v2_edge(x_enc[e_new % dat.num_nodes], context[e_old])
        # print(pred_e)
        targ_e = torch.cat((torch.zeros(e_0.shape[1], device =  self.device),
                            torch.ones(e_1.shape[1], device =  self.device)))
        if self.label_smooth:
            loss_e = cal_edge_loss(pred_e, targ_e)
        else:
            loss_e = F.binary_cross_entropy(pred_e, targ_e)    
        # print(cal_edge_loss(pred_e, targ_e))
#         print('*******************************')
#         print(loss_e)

        loss = loss_e + loss_c

        pred_v = torch.round(pred_v)
        return pred_v.eq(targ_v).sum().item(), pred_v.shape[0],\
            pred_e, targ_e, v_in_gr.sum()*v_1.shape[0], \
            loss

def create_model(num_features, arch, device):
    dc = ModelFull(num_features, device, arch)
    dc = dc.to(device = device)
    prs = add_weight_decay(dc.named_parameters(), arch['weight_decay'])
    optimizer = Adam(prs, lr=arch['lr'])
    # optimizer = RAdam(prs, lr=arch['lr'])
    return dc, optimizer


def train(mod, optimizer, data, adjacency, s_node0, bsz):
    mod.train()

    perm = torch.randperm(len(s_node0))[:bsz]
    s_node = s_node0[perm] 

    l_val = 0   
    for inp in data_gen(data, adjacency, s_node):
        optimizer.zero_grad() 
        _, _, _, _, _, loss = mod.iterate(data, inp, len(s_node))   
           
        l_val =  loss.item() 
        loss.backward()
        optimizer.step()
#         print(l_val) 
        
@torch.no_grad()
def test(mod, data, adjacency, s_node):
    mod.eval()
        
    acc_v,num_v, acc_e,num_e = [],[],[],[]
    
    for inp in data_gen(data, adjacency, s_node, shuffle=False):
        tp_v, n_v, pred_e, targ_e, n_e_sum, _ = mod.iterate(data, inp, len(s_node))
#         print(targ_e.shape)
        pred_e = torch.round(pred_e)
        tp_e = pred_e.eq(targ_e).sum().item()
        n_e = pred_e.shape[0]
         
        acc_v.append(tp_v)
        acc_e.append(tp_e)
        num_v.append(n_v)
        num_e.append(n_e)
    return  np.array(acc_v)/(np.array(num_v)+1e-9), \
            np.array(acc_e)/(np.array(num_e)+1e-9)


