import numpy as np
import torch
import networkx as nx


def create_nx_graph(edge_list, lbs):
    v,inv = torch.unique(edge_list, return_inverse=True)

    colours = {0:'red', 1:'blue', 2:'green',3:'yellow', 4:'grey', 5:'orange',6:'black'}
    labs = [colours[x] for x in lbs[v].numpy()]
    
    P=nx.Graph()
    P.add_edges_from(inv.numpy().T)
    P = P.to_undirected()

    largest_cc = max(nx.connected_components(P), key=len)
    G_conn = P.subgraph(largest_cc).copy()
#     G_conn = P
    
    lbs_fin =[]
    for i in range((len(labs))):
        if i in largest_cc:
#         if i in G_conn.nodes:
            lbs_fin.append(labs[i])
    
    return G_conn, lbs_fin

