import os
os.environ['https_proxy'] = ''
os.environ['http_proxy'] = ''

import numpy as np
import pandas as pd
import os.path as osp
import pathlib
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx


def tt_split_edges(dat):
    torch.manual_seed(0)
    train_test_ratio = 0.2

    row, col = dat.edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    e_mask = torch.FloatTensor(row.shape[0]).uniform_() >train_test_ratio
    dat.train_test_mask = torch.cat((e_mask, e_mask))
    dat.edge_index = torch.cat( (torch.stack((row, col ), dim=1).T,
                              torch.stack((col,row), dim=1).T), dim=1)
    return dat

def sample_bfs(nodes, num_hops, edge_index, num_nodes=None):
    edge_array = []
    graph_mask = []
    col, row = edge_index
    
    curr_mask = torch.full((num_nodes,), False,  dtype=torch.bool)
    visited_mask = torch.full((num_nodes,), False,  dtype=torch.bool)
    
    for ind, node_idx in enumerate(nodes):
        edges = []
        vmask = []
        curr_mask.fill_(False)
        visited_mask.fill_(False)
        visited_mask[node_idx] = True
        curr_mask[node_idx] = True
        for _ in range(num_hops):   
            edge_mask = curr_mask[row]  & ~visited_mask[col]
            curr_nodes = col[edge_mask]

            curr_mask.fill_(False)    
            curr_mask[curr_nodes] = True 
            
            vmask.append(visited_mask)
            visited_mask = visited_mask | curr_mask  # add new to visited

            edges.append(edge_index[:,edge_mask] + num_nodes*ind)    
        edge_array.append(edges)
        graph_mask.append(torch.stack(vmask))
    return edge_array, torch.stack(graph_mask)

def get_cora():
    d_name = 'Cora'
    path = osp.join(pathlib.Path().absolute(), '/data/egor/graph_gen/data', d_name)
    dataset = Planetoid(path, d_name, transform=T.NormalizeFeatures())
    gg = to_networkx(dataset[0], node_attrs=['x','y']).to_undirected()
    
    CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
    CG_big = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)[0]
    return from_networkx(CG_big), dataset[0]

def data_gen(data, adj, start_nodes, shuffle=True):
    n_hops = 7 
    batch_size = len(start_nodes)
    
    all_v = torch.arange(batch_size * data.num_nodes)

    edges_array, vm = sample_bfs(nodes=start_nodes, num_hops=n_hops, 
           edge_index=data.edge_index,
           num_nodes = data.num_nodes)
    
    if shuffle:
        binds = []
        ind_q = deque(range(1, n_hops))
        for _ in range(batch_size):
            binds.append(list(ind_q))
            ind_q.rotate()
        binds = np.array(binds)
    else:
        binds = np.array([list(range(1, n_hops))]*batch_size)
        
    for i in range(n_hops-1):
        gr = [torch.cat(edges_array[j][:binds[j,i]], dim=1) for j in range(batch_size)]
        gr = torch.cat(gr, dim=1)
        e_1 = [edges_array[j][binds[j,i]] for j in range(batch_size)]
        e_1 = torch.cat(e_1, dim=1)

        v_in_graph = vm[torch.arange(batch_size),binds[:,i],:]
        v_1, _ = e_1
        v_1 = torch.unique(v_1) 

        v_mask = ~v_in_graph.view((-1,))
        v_mask[v_1] = False
        v_0 = all_v[v_mask]
        i_0 = torch.randperm(v_0.shape[0])
        v_0 = v_0[i_0][:v_1.shape[0]]


        cc = v_1[torch.randint(len(v_1), (e_1.shape[1],))]  
        v_graph = all_v[v_in_graph.view((-1,))]
        rr = v_graph[torch.randint(len(v_graph), (e_1.shape[1],))]  
        e_0 = torch.stack((cc,rr))
        y_e_0 = adj[cc % data.num_nodes,rr % data.num_nodes].to(dtype=torch.float)

        yield torch.cat((gr, torch.flip(gr, dims=(0,))), dim=1), v_1, v_0, e_1, e_0, v_in_graph
