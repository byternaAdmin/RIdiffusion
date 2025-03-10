import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import time 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam,AdamW
import torch_geometric
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
# from pytorchtools import EarlyStopping

from tqdm.auto import tqdm
from ema_pytorch import EMA

from utils import PredefinedNoiseScheduleDiscrete
from model.hegnn.layers import EGNN_Sparse, GlobalLinearAttention_Sparse
from model.hegnn.utils import nodeEncoder, edgeEncoder
from dataset_src.large_dataset import RNAsolo

from torch.nn.modules.module import Module
import torch.nn.init as init
import manifolds
from hyp_utils import HNN
from manifolds.poincare import PoincareBall
import math
from sklearn.metrics import precision_recall_fscore_support
from math_utils import artanh, tanh
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout 
        self.use_bias = use_bias 
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        #self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight.to(device), x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )



def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor<0).any()

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr



class HEGNN(torch.nn.Module):
    def __init__(self,config ,input_feat_dim, hidden_channels, edge_attr_dim,  dropout=0.0, n_layers=1, output_dim = 4,
                 embedding=False, embedding_dim=64, mlp_num=2,update_edge = True,embed_ss = -1,norm_feat = False,hyperbolic_embedding=True, muti_ball=False):
        super(HEGNN, self).__init__()
        self.update_corr = False
        #torch.manual_seed(12345)
        self.muti_ball = muti_ball
        self.c = 1
        self.dropout = dropout
        self.config = config
        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.hyperbolic_embedding = hyperbolic_embedding
        self.manifold = getattr(manifolds, config['manifold'])() 
        self.hyp_linear = HypLinear(self.manifold, 15, 128, 1, dropout, False)
        self.embedding = embedding
        self.embed_ss = embed_ss
        self.n_layers = n_layers
        if embedding:
            self.time_mlp = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(),
                                       nn.Linear(hidden_channels, embedding_dim))   

        else:
            self.time_mlp = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(),
                                       nn.Linear(hidden_channels, input_feat_dim))
            

        for i in range(n_layers):
            if embedding:
                layer = EGNN_Sparse(embedding_dim, m_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    mlp_num=mlp_num,update_edge = self.update_edge,norm_feats=norm_feat)
            else:
                layer = EGNN_Sparse(input_feat_dim, m_dim=hidden_channels, edge_attr_dim=edge_attr_dim, dropout=dropout,
                                    mlp_num=mlp_num,update_edge = self.update_edge,norm_feats=norm_feat)
            self.mpnn_layes.append(layer)

            if embedding:
                time_mlp_layer = nn.Sequential(
                    nn.SiLU(), 
                    nn.Linear(embedding_dim, (embedding_dim) * 2)
                    )
                ff_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Dropout(p=dropout),nn.SiLU(), torch_geometric.nn.norm.LayerNorm(embedding_dim),nn.Linear(embedding_dim, embedding_dim)) 
            else:
                time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(input_feat_dim, (input_feat_dim) * 2))
                ff_layer = nn.Sequential(nn.Linear(input_feat_dim, input_feat_dim), nn.Dropout(p=dropout) ,nn.SiLU(), torch_geometric.nn.norm.LayerNorm(input_feat_dim), nn.Linear(input_feat_dim, input_feat_dim)) 

            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)


        if embedding:
            self.node_embedding = nodeEncoder(embedding_dim)
            self.edge_embedding = edgeEncoder(embedding_dim)
            self.lin = Linear(embedding_dim, output_dim)
        else:
            self.lin = Linear(input_feat_dim, output_dim)
            
            
        self.attn_layer = GlobalLinearAttention_Sparse(dim=3,
                                      heads=4,
                                      dim_head=64)

        self.MLP_out = nn.Sequential(
            nn.Linear(131, 131),
            nn.ReLU(),
            nn.Linear(131, 131)
        )


    def forward(self, data, time): 
        x, pos, extra_x, edge_index, edge_attr, batch = data.x, data.pos,  data.extra_x,data.edge_index, data.edge_attr,data.batch
        t = self.time_mlp(time)
        #ss_embed = self.ss_mlp(ss)
        # x = x.float()
        x = torch.cat([x,extra_x],dim=1)
        if self.hyperbolic_embedding: 
            a = self.manifold.proj_tan0(x, self.c)
            b = self.manifold.expmap0(a, c=self.c)
            x_hyp = self.manifold.proj(b, c=self.c)
            x = self.hyp_linear.forward(x_hyp)
            x = self.manifold.logmap0(x, c=self.c)
            
        else:
            x = self.nodeEncoder(x)

        
        
        edge_attr = self.edge_embedding(edge_attr)
        
        
        if self.muti_ball:
            self.config['n_nodes'], self.config['feat_dim'] = x.shape
            hyp_model = HNN(1.0,self.config)
            x = hyp_model.encode(x)
        
        x = torch.cat([pos, x], dim=1)
        
        for i, layer in enumerate(self.mpnn_layes):

            if self.embed_ss == -2 and i == self.n_layers-1:
                corr, feats = x[:,0:3], x[:,3:]
                #feats = feats + ss_embed #[N,hidden_dim]+[N,hidden_dim]
                x = torch.cat([corr, feats], dim=-1)
            if self.update_corr: 
                corr, feats = x[:,0:3],x[:,3:]
                corr_att = self.attn_layer(corr,corr,batch)
                x = torch.cat([corr_att[0], feats], dim=-1)

            if self.update_edge: 
                h,edge_attr = layer(x, edge_index, edge_attr, batch) #[N,hidden_dim] 
                # h,edge_attr = layer(x, edge_index, batch) 
            else:
                h = layer(x, edge_index, edge_attr, batch)
                # h = layer(x, edge_index, batch) #[N,hidden_dim]
                
            h = self.manifold.expmap0(h, c=self.c)
            
            corr, feats = h[:,0:3], h[:,3:]
            time_emb = self.time_mlp_list[i](t) #[B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2,dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats*(scale+1) +shift
            
            feats = self.ff_list[i](feats)
            
            x = torch.cat([corr, feats], dim=-1)

        corr, x = x[:,0:3],x[:,3:]
        
        if self.embed_ss == -1:
            x=x
            #x = x+ss_embed 

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes


    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x

class BlosumTransition:
    def __init__(self, blosum_path='dataset_src/blosum_substitute.pt',x_classes=4,timestep = 500):
        try:
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        except FileNotFoundError:
            blosum_path = '../'+blosum_path
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        self.X_classes = x_classes
        self.timestep = timestep
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        Qt_temperature = self.Qt_temperature.unsqueeze(dim=0)
        Qt_temperature = Qt_temperature.unsqueeze(dim=0)
        if temperature_list.shape[0] != self.timestep:
            output_tensor = F.interpolate(temperature_list, size=timestep+1, mode='linear', align_corners=True)
            self.temperature_list = output_tensor.squeeze()
            output_tensor = F.interpolate(Qt_temperature, size=timestep+1, mode='linear', align_corners=True)
            self.Qt_temperature = output_tensor.squeeze()
        else:    
            self.temperature_list = self.temperature_list
            self.Qt_temperature = self.Qt_temperature
    
    def get_Qt_bar(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        q_x[q_x < 1e-6] = 1e-6
        return q_x

    def get_Qt(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.Qt_temperature = self.Qt_temperature.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.Qt_temperature[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        return q_x

class RIdiffusion(nn.Module):
    def __init__(self,model,*,timesteps=500,sampling_timesteps = None,loss_type='CE',objective = 'pred_x0',config = {'noise_type':'uniform'},schedule_fn_kwargs = dict()):
        super().__init__()
        self.model = model
        # self.self_condition = self.model.self_condition
        self.objective = objective
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.transition_model = DiscreteUniformTransition(x_classes=4)
        self.config  = config
        if config['noise_type'] == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=4)
        elif config['noise_type'] == 'blosum':
            self.transition_model = BlosumTransition(timestep=self.timesteps+1)

        assert objective in {'pred_noise', 'pred_x0'}

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=self.timesteps,noise_type='uniform')

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy

    def apply_noise(self,data,t_int):
        t_float = t_int / self.timesteps
        if self.config['noise_type'] == 'uniform':
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t_float, device=data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:4].unsqueeze(2)).squeeze()
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 4)
        noise_data = data.clone()
        noise_data.x = noise_X
        return noise_data


    
    def sample_discrete_feature_noise(self,limit_dist ,num_node):
        x_limit = limit_dist[None,:].expand(num_node,-1) #[num_node,20]
        U_X = x_limit.flatten(end_dim=-2).multinomial(1).squeeze()
        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        return U_X


    def diffusion_loss(self,data,t_int):
        '''
        Compute the divergence between  q(x_t-1|x_t,x_0) and p_{\theta}(x_t-1|x_t)
        
        '''
        # q(x_t-1|x_t,x_0)
        s_int = t_int - 1 
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps    
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=data.x.device)
        Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:4].unsqueeze(2)).squeeze()       
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 4).type_as(data.x)
        prob_true = self.compute_posterior_distribution(noise_X,Qt,Qsb,Qtb,data)  #[N,d_t-1]


        #p_{\theta}(x_t-1|x_t) = \sum_{x0} q(x_t-1|x_t,x_0)p(x0|xt)
        noise_data = data.clone()
        noise_data.x = noise_X #x_t
        t = t_int*torch.ones(size=(data.batch[-1]+1, 1), device=data.x.device).float()
        pred = self.model(noise_data,t)
        pred_X = F.softmax(pred,dim = -1) #\hat{p(X)}_0
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=noise_X,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_pred = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        loss = self.loss_fn(prob_pred,prob_true,reduction='mean')
        return loss

    def compute_val_loss(self,data,evaluate_all=False):
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        diffusion_loss = self.diffusion_loss(data,t_int)
        return diffusion_loss
    
    def compute_batched_over0_posterior_distribution(self,X_t,Q_t,Qsb,Qtb,data):
        """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
        """
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Q_t.transpose(-1,-2)
        X_t_ = X_t.unsqueeze(dim = -2)
        left_term = X_t_ @ Qt_T[data.batch] #[N,1,d_t-1]
        # left_term = left_term.unsqueeze(dim = 1) #[N,1,dt-1]

        right_term = Qsb[data.batch] #[N,d0,d_t-1]

        numerator = left_term * right_term #[N,d0,d_t-1]

        prod = Qtb[data.batch] @ X_t.unsqueeze(dim=2) # N,d0,1
        denominator = prod
        denominator[denominator == 0] = 1e-6        

        out = numerator/denominator

        return out

    def compute_posterior_distribution(self,M_t, Qt_M, Qsb_M, Qtb_M,data):
        """ 
        M: is the distribution of X_0
        Compute  q(x_t-1|x_t,x_0) = xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        """
         
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Qt_M.transpose(-1,-2)
        X_t = M_t.unsqueeze(dim = -2)
        left_term = X_t @ Qt_T[data.batch] #[N,1,d_t-1]
        
        M_0 = data.x.unsqueeze(dim = -2) #[N,1,d_t-1]
        right_term = M_0@Qsb_M[data.batch] #[N,1,dt-1]
        numerator = (left_term * right_term).squeeze() #[N,d_t-1]


        X_t_T = M_t.unsqueeze(dim = -1)
        prod = M_0@Qtb_M[data.batch]@X_t_T # [N,1,1]
        denominator = prod.squeeze()
        denominator[denominator == 0] = 1e-6        

        out = (numerator/denominator.unsqueeze(dim=-1)).squeeze()

        return out        #[N,d_t-1]
    
    def sample_p_zs_given_zt(self,t,s,zt,data,cond,diverse,step,last_step):
        """
        sample zs~p(zs|zt)
        """
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        if self.config['noise_type'] == 'uniform':
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, data.x.device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, data.x.device)
        else:
            Qtb = self.transition_model.get_Qt_bar(t, data.x.device)
            Qsb = self.transition_model.get_Qt_bar(s, data.x.device)

        Qt = (Qsb/Qtb)/(Qsb/Qtb).sum(dim=-1).unsqueeze(dim=2) #approximate

        noise_data = data.clone()
        noise_data.x = zt 
        pred = self.model(noise_data,t*self.timesteps)
        pred_X = F.softmax(pred,dim = -1) 
        
        if isinstance(cond, torch.Tensor):
            pred_X[cond] = data.x[cond]

        if last_step:
            #pred_X = F.softmax(pred,dim = -1)
            sample_s = pred_X.argmax(dim = 1)
            final_predicted_X = F.one_hot(sample_s,num_classes = 4).float()

            return pred,final_predicted_X
            
        
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=zt,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20 approximate Q_t-s with Qt 
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        
        if diverse :
            sample_s = prob_X.multinomial(1).squeeze()
        else:
            sample_s = prob_X.argmax(dim=1).squeeze()

        X_s = F.one_hot(sample_s,num_classes = 4).float()

        return X_s,final_predicted_X if last_step else None
    
    def sample(self,data,cond = False,temperature=1.0,stop = 0):
        limit_dist = torch.ones(4)/4
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps)))): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + 1
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,cond,temperature,last_step=s_int==stop)
        return zt,final_predicted_X
    
    def ddim_sample(self,data,cond = False,diverse=False,stop = 0,step=50):
        limit_dist = torch.ones(4)/4
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps,step)))): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + step
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,cond,diverse,step,last_step=s_int==stop)
        return zt,final_predicted_X


    def forward(self,data,logit=False):
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        noise_data = self.apply_noise(data ,t_int)
        pred_X = self.model(noise_data,t_int) #have parameter

        if self.objective == 'pred_x0':
            target = data.x
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_fn(pred_X,target,reduction='mean')
        
        if logit:
            return loss, pred_X
        else:
            return loss

def seq_recovery(data,pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum()/ind.shape[0]
    return recovery,ind.cpu()
