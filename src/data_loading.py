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

from torch_geometric.data import Data, Batch, ClusterData
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import to_networkx, from_networkx, subgraph


import networkx as nx

def get_connected_comp(dat):
    gg = to_networkx(dat, node_attrs=['x','y']).to_undirected()
    CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
    CG_big = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)[0]
    return from_networkx(CG_big)

def get_dataset(d_type, d_name, pth):    
    path = osp.join(pathlib.Path().absolute(), pth , d_name)
    if d_type == 'full':
        dataset = CitationFull(path, d_name, transform=T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, d_name, transform=T.NormalizeFeatures())
    
    # data = get_connected_comp(dataset[0])
    data = dataset[0]
    
    nparts = 3
    cluster_data = ClusterData(data, num_parts=nparts, recursive=False,
                           save_dir=dataset.processed_dir)
    n_nodes = data.num_nodes
    b_point = cluster_data.partptr[2]
    row, col, edge_attr = cluster_data.data.adj.t().coo()
    new_edge_index = torch.stack([row, col], dim=0)

    adjacency = torch.full((data.num_nodes, data.num_nodes), False, dtype= torch.bool)
    adjacency[new_edge_index[0], new_edge_index[1]] = True

    train_data = Data(x=cluster_data.data.x[:b_point], 
                     edge_index=subgraph(torch.arange(b_point), new_edge_index )[0], 
                        y=cluster_data.data.y[:b_point])
    test_data =  cluster_data[nparts-1]                 
    # test_data = Data(x=cluster_data.data.x[b_point:n_nodes], 
    #                  edge_index=subgraph(torch.arange(b_point, n_nodes), new_edge_index )[0], 
    #                     y=cluster_data.data.y[b_point:n_nodes])
    return train_data, test_data, adjacency, dataset[0]

def get_start(dat, b_size):
    inds, cnts = torch.unique(dat.edge_index, return_counts = True)
    sel = torch.argsort(cnts, descending=True)[:b_size*2]
    return inds[sel]

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
#     pth = '/home/eshikov/work/graph/graph-generation/graph_generation/data'
    pth = '/data/egor/graph_generation/graph_generation/data'
    path = osp.join(pathlib.Path().absolute(), pth , d_name)
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

        v_in_graph = vm[torch.arange(batch_size), binds[:,i],:]
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

def data_gen_edges(data, batch_size = 4, step = 8):
    istart = step
    n_edges = data.edge_index.shape[1]
    e_perm_edges = []
    for i in range(batch_size):
        e_perm_edges.append(data.edge_index[:,torch.randperm(n_edges)] + data.num_nodes *i)
    e_perm_edges = torch.stack(e_perm_edges)
    edge_index =  torch.transpose(e_perm_edges, 0, 1)

    active_mask = torch.full((data.num_nodes * batch_size,), False,  dtype=torch.bool)
    visited_mask = torch.full((data.num_nodes * batch_size,), False,  dtype=torch.bool)
    nodes = torch.arange(batch_size*data.num_nodes)
    visited_mask[torch.flatten(edge_index[:,:,:istart])] = True 

    for i in range(istart+step, n_edges, step):
        # *********  sampling edges
        graph = edge_index[:,:,:(i-step)]
        e_1 = edge_index[:,:,(i-step):i]

        # *********  node expansion
        active_mask.fill_(False)  
        active_mask[torch.flatten(e_1)] = True  
        v_exp_0_mask = visited_mask & ~active_mask # in graph but not expanding
        v_exp_1_mask = visited_mask & active_mask  # in graph AND expanding
        n_e_1 = v_exp_1_mask.sum()
        v_exp_1 = nodes[v_exp_1_mask]
        v_exp_0 = nodes[v_exp_0_mask][:n_e_1]

        # *********  node additon
        v_add_1_mask = active_mask & ~visited_mask
        
        v_add_0_mask = ~visited_mask 
        n_v_1 = v_add_1_mask.sum()
        v_left = nodes[v_add_0_mask]
        v_add_0 = v_left[torch.randperm(v_left.shape[0])][:n_v_1]
        v_add_1 = nodes[v_add_1_mask]
        # print(n_v_1, n_e_1)

        # *********  negative edges 
        r1,c1 = e_1
        c0 = torch.zeros(size=c1.shape)
        c0[:, 0] = c1[:, -1]
        c0[:, 1:] = c1[:, :-1]
        e_0 = torch.stack( (r1,c0))
            
        yield graph.reshape((2,-1)), \
                v_add_1_mask.view((batch_size, data.num_nodes)).sum(dim=1),\
                torch.cat((v_exp_0,v_exp_1)),\
                torch.cat((v_add_0,v_add_1)), \
                torch.cat((e_0,e_1), dim=2).reshape((2,-1)).to(dtype = torch.long),\
                visited_mask.view((batch_size, data.num_nodes))

        visited_mask = visited_mask | v_add_1_mask  # add new to visited
