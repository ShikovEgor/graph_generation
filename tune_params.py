import numpy as np
import pandas as pd
import os.path as osp
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Sequential, ReLU, Linear, Dropout, BatchNorm1d
from torch.nn.utils import weight_norm

import networkx as nx

import optuna

from src.models import TrEnc, GraphNet, Vects2Edge, Vects2Choice, GraphAgg,add_weight_decay,FeatureTransform
from src.models import  ClassDiff

from src.radam import RAdam
from src.data_loading import data_gen, get_cora
from src.utils import create_nx_graph
from src.train import ModelFull, create_model, train, test
from src.data_loading import get_dataset, get_start

B_SIZE = 4
class DoTrial(): 
    def __init__(self, data_train, data_test, adjacency, device, start_points, test_point):
        self.data_train = data_train
        self.data_test = data_test
        self.adjacency = adjacency
        self.device = device
        self.start_points = start_points
        self.test_point = test_point  
        
    def __call__(self, trial):        
#         lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
#         hidden =  trial.suggest_categorical('nd', [32, 64, 128, 256])
        
        nlayers_gnn = trial.suggest_int('nlayers_gnn', 2, 4)
        # arch['nlayers_gnn'] = 4
#         weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        
#         add_trans =  trial.suggest_categorical('add_trans', [True, False])  

#         smth_loss =  trial.suggest_categorical('lb_smth', [True, False])  
        # arch['add_trans'] = True
#         agg_type =  trial.suggest_categorical('agg_type', ['mean', 'gate', 'softmax'])
#         f_type_e =  trial.suggest_categorical('fin_type_e', ['cos', 'mm_sig', 'mm_cos', 'diff_mlp', 'diff_sum', 'stack'])
#         f_type_c =  trial.suggest_categorical('fin_type_c', ['cos', 'mm_sig', 'mm_cos', 'diff_mlp', 'diff_sum', 'stack'])
#         conv_type =  trial.suggest_categorical('conv_type', ['sage','sg','resgate','gin','gat'])
#         node_agg_type =  trial.suggest_categorical('node_agg_type', ['mean', 'max'])
        

        agg_type =  trial.suggest_categorical('agg_type', ['mean', 'sum_gate', 'mean_gate','max', 'softmax'])
        f_type_e =  trial.suggest_categorical('fin_type_e', ['cos', 'mm_sig', 'diff_mlp', 'diff_sum', 'stack'])
        f_type_c =  trial.suggest_categorical('fin_type_c', ['cos', 'mm_sig', 'diff_mlp', 'diff_sum', 'stack'])
        conv_type =  trial.suggest_categorical('conv_type', ['sage','sg','resgate','gin','gat'])
#         node_agg_type =  trial.suggest_categorical('node_agg_type', ['mean', 'max'])
        
        

        arch = {}
        for k,v in trial.params.items():
            arch[k] = v  
        arch['add_trans'] = False
        arch['lb_smth'] = True
        arch['node_agg_type'] = 'max'
        arch['lr'] = 0.005
        arch['nd'] = 256
        arch['weight_decay'] = 0.0007
        
        
#         print(type(trial.params))
        # print( arch)
        try:
            model, optimizer = create_model(self.data_train.num_features, arch, self.device)
            num_epochs = 200
            patience = 0
            best_score  = 0
            for epoch in range(1, num_epochs + 1):
                train(model, optimizer, self.data_train, self.adjacency, self.start_points, B_SIZE)
                acc_v, acc_e = test(model, self.data_test, self.adjacency, self.test_point)

                # print(acc_v.mean(), acc_e.mean())
                score = acc_v.mean() + acc_e.mean()

                if score > best_score:
                    best_score = score
                    patience = 0
                else:
                    patience +=1

                if (patience > 30):
                    print('patience lost')
                    break    
        except Exception as e: 
            raise optuna.TrialPruned()
            
        print(best_score)
        return best_score

def main():
    device = torch.device('cuda')

    d_train, d_test, adj, data0 = get_dataset('cora', '/data/egor/graph_generation/graph_generation/data')

    start_train = get_start(d_train, B_SIZE*3)
    start_test = get_start(d_test, 1)[:1]


    do_trial = DoTrial(d_train, d_test , adj, device, start_train, start_test)
    study = optuna.create_study(direction='maximize', storage='sqlite:///gen8.db')
    study.optimize(do_trial, n_trials=1500)

if __name__ == "__main__":
    main()
