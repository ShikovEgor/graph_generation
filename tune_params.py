import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import networkx as nx

import optuna

from src.models import TrEnc, GraphNet, Vects2Edge, Vects2Choice, GraphAgg,add_weight_decay,FeatureTransform
from src.radam import RAdam
from src.data_loading import data_gen, get_cora
from src.utils import create_nx_graph


class ModelFull(torch.nn.Module):
    def __init__(self, data, num_nodes, num_features, device, arch):
        super().__init__()
        
        enc_layers = []
        enc_layers.append(FeatureTransform(num_features, arch['nd']))
        if arch['add_trans']:
            enc_layers.append(TrEnc(num_features, arch['nd']))
        self.encoder = nn.Sequential(*enc_layers)
        
        self.graph_net = GraphNet(arch)
        self.v2_edge = Vects2Edge(arch['nd'])
        self.v2_choice = Vects2Choice(arch['nd'])
        self.gagg = GraphAgg(arch['nd'], arch['agg_type'])
        self.num_nodes = num_nodes
        self.device = device
        self.nd = arch['nd']
        self.data = data

    def iterate(self, inp, b_size): 
        encoded = self.encoder(self.data.x.to(device = self.device))
        return self.decode(inp, encoded, b_size)
        
    def decode(self, inp, x_enc, b_size):
        gr, v_1, v_0, e_1, e_0, v_in_gr = inp
        gr, v_1, v_0, e_1, e_0, v_in_gr = gr.to(device = self.device), \
                                    v_1.to(device = self.device), \
                                    v_0.to(device = self.device), \
                                    e_1.to(device = self.device), \
                                    e_0.to(device = self.device), \
                                    v_in_gr.to(device = self.device)
        
        decoded = self.graph_net( x_enc[torch.cat(b_size*[torch.arange(self.num_nodes, \
                                                                        device = self.device)]),:], gr)
        graph_emb = self.gagg(decoded.view(-1,self.num_nodes, self.nd), v_in_gr)     
        pred_v =  self.v2_choice(graph_emb, x_enc)  

        pred_v = pred_v[torch.cat((v_0,v_1))]
        targ_v = torch.cat((torch.zeros(v_0.shape[0], device = self.device),
                            torch.ones(v_1.shape[0], device = self.device))) 

        loss_c = F.binary_cross_entropy_with_logits(pred_v, targ_v) 

        e_new, e_old = torch.cat((e_0,e_1), dim=1)        
        pred_e = self.v2_edge(x_enc[e_new % self.num_nodes], decoded[e_old]) 
        targ_e = torch.cat((torch.zeros(e_0.shape[1], device = self.device),
                            torch.ones(e_1.shape[1], device = self.device)))
        loss_e = F.binary_cross_entropy_with_logits(pred_e, targ_e)    

#         print(cal_edge_loss(pred_e, targ_e))
#         print('*******************************')
#         print(loss_e)

        loss = loss_e + loss_c

        pred_v = torch.round(torch.sigmoid(pred_v))
        return pred_v.eq(targ_v).sum().item(), pred_v.shape[0],\
            pred_e, targ_e, v_in_gr.sum()*v_1.shape[0], \
            loss
       

def create_model(dataset, arch, device):
    dc = ModelFull(dataset, dataset.num_nodes,dataset.num_features, device, arch)
    dc = dc.to(device = device)
    prs = add_weight_decay(dc.named_parameters(), arch['weight_decay'])
    optimizer = RAdam(prs, lr=arch['lr'])
    return dc, optimizer

def train(mod, optimizer, data, adjacency, s_node):
    mod.train()
        
    l_val = 0   
    for inp in data_gen(data, adjacency, s_node):
        optimizer.zero_grad() 
        _, _, _, _, _, loss = mod.iterate(inp, len(s_node))   
           
        l_val =  loss.item() 
        loss.backward()
        optimizer.step()
#         print(l_val) 
        
@torch.no_grad()
def test(mod, data, adjacency, s_node):
    mod.eval()
        
    acc_v,num_v, acc_e,num_e = [],[],[],[]
    
    for inp in data_gen(data, adjacency, s_node, shuffle=False):
        tp_v, n_v, pred_e, targ_e, n_e_sum, _ = mod.iterate(inp, len(s_node))
#         print(targ_e.shape)
        pred_e = torch.round(torch.sigmoid(pred_e))
        tp_e = pred_e.eq(targ_e).sum().item()
        n_e = pred_e.shape[0]
         
        acc_v.append(tp_v)
        acc_e.append(tp_e)
        num_v.append(n_v)
        num_e.append(n_e)
    return  np.array(acc_v)/(np.array(num_v)+1e-9), \
            np.array(acc_e)/(np.array(num_e)+1e-9)


class DoTrial(): 
    def __init__(self, data, adjacency, device, start_points, test_point):
        self.data = data
        self.adjacency = adjacency
        self.device = device
        self.start_points = start_points
        self.test_point = test_point

    def __call__(self, trial):
        print('new trial')
        
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        hidden =  trial.suggest_categorical('nd', [64, 128, 256])
        nlayers_gnn = trial.suggest_int('nlayers_gnn', 2, 4)
        dropout = trial.suggest_uniform('dropout', 0, 0.5)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        add_trans =  trial.suggest_categorical('add_trans', [True, False])  
        agg_type =  trial.suggest_categorical('agg_type', ['mean', 'gate', 'softmax'])
        graph_sys_type =  trial.suggest_categorical('sys_type', ['static', 'dysat', 'my'])
        conv_type =  trial.suggest_categorical('conv_type', ['static', 'dysat', 'my'])
        node_agg_type =  trial.suggest_categorical('node_agg_type', ['mean', 'max', 'add'])


        model, optimizer = create_model(self.data, trial.params, self.device)

        num_epochs = 200
        patience = 0
        best_score  = 0
        for epoch in range(1, num_epochs + 1):
            train(model, optimizer, self.data, self.adjacency, self.start_points)
            acc_v, acc_e = test(model, self.data, self.adjacency, self.test_point)
            # print(acc_v.mean(), acc_e.mean())
            score = acc_v.mean() + acc_e.mean()
            
            if score > best_score:
                best_score = score
                patience = 0
            else:
                patience +=1

            if (patience > 20):
                print('patience lost')
                break    

        return best_score

def main():
    device = torch.device('cuda')

    data, data0 = get_cora()

    B_SIZE = 64
    adjacency = torch.full((data.num_nodes, data.num_nodes), False, dtype= torch.bool)
    adjacency[data.edge_index[0],data.edge_index[1]] = True

    degrees = torch.unique(data.edge_index[0,:], return_counts = True)[1]
    # start_points = torch.linspace(0, data.num_nodes*(batch_size-1), 
    #                               batch_size, dtype=torch.long)

    start_nodes = torch.argsort(degrees, descending=True)
    start_points = start_nodes[1:(B_SIZE+1)]
    test_point = start_nodes[:1]

    data.x = data.x.to(dtype=torch.float)


    do_trial = DoTrial(data, adjacency, device, start_points, test_point)
    study = optuna.create_study(direction='maximize', storage='sqlite:///gen1.db')
    study.optimize(do_trial, n_trials=1000)

if __name__ == "__main__":
    main()
