import numpy as np
import torch
from collections import deque


def calc_BA(grp, n_nodes):
    v,  cnt = torch.unique(grp, return_counts=True)
    zrs = torch.zeros(n_nodes)
    zrs[v] = cnt.to(dtype=torch.float)/2
    
    return zrs/zrs.sum()

def data_gen_edges2(data, batch_size = 4, stepf = 8):
#     istart = step
    mask = data.edge_index[0] < data.edge_index[1]
    n_edges = int(mask.sum().item())
    e_perm_edges = []
    for i in range(batch_size):
        e_perm_edges.append(data.edge_index[:,mask][:,torch.randperm(n_edges)] + data.num_nodes *i)
#     e_perm_edges = torch.stack(e_perm_edges)
#     edge_index =  torch.transpose(e_perm_edges, 0, 1)
    
    n_steps = batch_size
    step_max = 128
    ratio = 0.1
    binds = []
    ind_q = deque(range(n_steps))
    for _ in range(batch_size):
        binds.append(list(ind_q))
        ind_q.rotate()
    binds = np.array(binds)
    
    n_min = 64
    
    ind_array = (np.linspace((n_min+step_max)/n_edges,1,n_steps)*n_edges).astype(int)
    step_array = np.clip(ind_array*ratio, a_min = 0, a_max = step_max).astype(int)
    
    for jj in range(n_steps):
        indxs = ind_array[binds[:,jj]]
        steps = step_array[binds[:,jj]]
#         print(indxs, steps)
    
        # *********  sampling edges
#         graph = torch.cat([e_perm_edges[j][:,:(indxs[j]-steps[j])] for j in range(batch_size)], dim=1)
#         BA_factors = [calc_BA(e_perm_edges[j][:,:(indxs[j]-steps[j])]) for j in range(batch_size)]
#         e_1 = torch.cat([e_perm_edges[j][:,(indxs[j]-steps[j]):indxs[j]] for j in range(batch_size)], dim=1)
        
        #calc_BA - peredelat!!!!!!!!!!!!!!
        
        graph = torch.cat([e_perm_edges[j][:,:(indxs[j]-step_max)] for j in range(batch_size)], dim=1)
        BA_factors = [calc_BA(e_perm_edges[j][:,:(indxs[j]-step_max)], data.num_nodes * batch_size) for j in range(batch_size)]
        e_1 = torch.cat([e_perm_edges[j][:,(indxs[j]-step_max):indxs[j]] for j in range(batch_size)], dim=1)        
#         graph = edge_index[:,:,:(i-step)]
#         e_1 = edge_index[:,:,(i-step):i]
        
        # *********  negative edges 
        r1,c1 = e_1
        c0 = torch.zeros(size=c1.shape)
        c0[0] = c1[ -1] # hz, rabotaet li????????????????
        c0[ 1:] = c1[:-1] # smeshaem po krugu
        e_0 = torch.stack( (r1,c0))
            
            
        vmask = torch.full((data.num_nodes*batch_size,), False,  dtype=torch.bool)
        vmask[torch.unique(graph)] = True
             
        yield graph, torch.cat((e_0,e_1), dim=1).to(dtype = torch.long), BA_factors, vmask

def train(sampler, mod, optimizer, data, bsz):
    mod.train()
        
    l_val = 0   
    mod.reset_state(data.num_nodes*bsz)
    for inp  in sampler(data, batch_size = bsz):
        optimizer.zero_grad() 
        _, _, loss = mod.iterate(data, inp, bsz) 
           
        l_val =  loss.item() 
        loss.backward(retain_graph=True)
        optimizer.step()
#         print(l_val) 
        
@torch.no_grad()
def test(sampler,mod, data, bsz):
    mod.eval()
        
    acc_e,num_e = [],[]
    mod.reset_state(data.num_nodes*bsz)
    for inp in sampler(data, batch_size = bsz):
        pred_e, targ_e, _ = mod.iterate(data, inp, bsz) 
        pred_e = torch.round(pred_e)
        tp_e = pred_e.eq(targ_e).sum().item()
        n_e = pred_e.shape[0]
         
        acc_e.append(tp_e)
        num_e.append(n_e)
    return  np.array(acc_e)/(np.array(num_e)+1e-9)

    