###########################
# Continuous-Time Attention for Sequential Learning
# Author: Yi-Hsiang Chen
# Modify from: https://github.com/YuliaRubanova/latent_ode
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F

import lib.utils as utils

#####################################################################################################

class ODEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, layer_type = "concat", device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.layer_type = layer_type
        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        dx = torch.squeeze(y)
        for l, layer in enumerate(self.gradient_net):
            if isinstance(layer, utils.LAYER[self.layer_type]):
                dx = layer(t_local, dx)
            else:
                dx = layer(dx)
        if len(dx.size())!=len(y.size()):
            dx = torch.unsqueeze(dx,0)
        return dx

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

#####################################################################################################

class ODEFunc_att(nn.Module):

    def __init__(self, input_dim, latent_dim, ode_func_net, layer_type = "concat", device = torch.device("cpu")):
        super(ODEFunc_att, self).__init__()
        
        utils.init_network_weights(ode_func_net)
        self.ode_func_list = ode_func_net
        self.layer_type = layer_type
        
        self.weight = nn.Linear(latent_dim,latent_dim)
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.query = None

    def forward(self, t, states):
        assert len(states) == 3
        z = states[0]
        
        # increment num evals
        self._num_evals += 1
        # convert to tensor
        #t = torch.tensor(t).type_as(z)
        n_traj, n_tp, n_dims = self.query.size()[0], self.query.size()[1], self.query.size()[2]
        
        # compute grad
        dz = torch.squeeze(self.get_ode_gradient_nn(t, z))
        da = torch.exp(torch.tanh(torch.bmm(self.query, torch.unsqueeze(z, 0).permute(1,2,0))))
        # (n_traj, n_tp, n_dims)  X (n_traj, n_dims, 1) == (n_traj, n_tp, 1)
        dc = da*z.unsqueeze(1).repeat(1,n_tp,1) 
        # (n_traj, n_tp, 1) * (n_traj, n_tp, n_dims) == (n_traj, n_tp, n_dims)
        da = torch.cat((da,self.zero_mask[:,n_tp:,:1]),1)
        dc = torch.cat((dc,self.zero_mask[:,n_tp:,:]),1)
        return tuple([dz, da, dc])
        
    def get_ode_gradient_nn(self, t_local, y):
        dx = torch.squeeze(y)
        for l, layer in enumerate(self.ode_func_list):
            if isinstance(layer, utils.LAYER[self.layer_type]):
                dx = layer(t_local, dx)
            else:
                dx = layer(dx)
        if len(dx.size())!=len(y.size()):
            dx = torch.unsqueeze(dx,0)
        return dx
        
    def set_query(self, total_n_tp):
        self.query = None
        self.total_n_tp = total_n_tp
        self._num_evals.fill_(0)
        
    def feed_query(self, query):
        if self.query is None:
            self.query = torch.unsqueeze(self.weight(query), dim=1) # (n_traj, 1, n_dims)
            self.zero_mask = torch.zeros_like(self.query).repeat(1,self.total_n_tp,1).to(query)
        else:
            self.query = torch.cat((self.query,torch.unsqueeze(self.weight(query), dim=1)),1) # (n_traj, n_tp, n_dims)


class ODEFunc_selfatt(nn.Module):

    def __init__(self, input_dim, latent_dim, ode_func_net, layer_type = "concat", device = torch.device("cpu")):
        super(ODEFunc_selfatt, self).__init__()
        
        utils.init_network_weights(ode_func_net)
        self.ode_func_list = ode_func_net
        self.layer_type = layer_type
        
        self.weight_q = nn.Linear(input_dim+1,latent_dim)
        self.weight_k = nn.Linear(latent_dim,latent_dim)
        self.weight_v = nn.Linear(latent_dim,latent_dim)
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.query = None

    def forward(self, t, states):
        assert len(states) == 3
        z = states[0]
        
        # increment num evals
        self._num_evals += 1
        # convert to tensor
        #t = torch.tensor(t).type_as(z)
        
        # compute grad
        dz = self.get_ode_gradient_nn(t, z)
        da = torch.exp(torch.tanh(torch.bmm(self.query, torch.unsqueeze(self.weight_k(z), 0).permute(1,2,0))))
        # (n_traj, n_tp, n_dims)  X (n_traj, n_dims, 1) == (n_traj, n_tp, 1)
        
        dc = da*self.weight_v(z).unsqueeze(1).repeat(1,self.n_tp,1) 
        # (n_traj, n_tp, 1) * (n_traj, n_tp, n_dims) == (n_traj, n_tp, n_dims)
        return tuple([dz, da, dc])
        
    def get_ode_gradient_nn(self, t_local, dx):
        dx = dx.squeeze()
        for l, layer in enumerate(self.ode_func_list):
            if isinstance(layer, utils.LAYER[self.layer_type]):
                dx = layer(t_local, dx)
            else:
                dx = layer(dx)
        return dx
        
    def set_query(self, query, time_points):
        self.query = self.weight_q(query) # (n_traj, n_tp, n_dims)
        n_traj, n_tp, n_dims = self.query.size()
        self.n_tp = n_tp
        self._num_evals.fill_(0)

#####################################################################################################

class ODEFunc_causal_att(nn.Module):

    def __init__(self, input_dim, latent_dim, ode_func_net, layer_type = "concat", device = torch.device("cpu")):
        super(ODEFunc_causal_att, self).__init__()
        
        utils.init_network_weights(ode_func_net)
        self.ode_func_list = ode_func_net
        self.layer_type = layer_type
        
        self.weight_q = nn.Linear(latent_dim,latent_dim)
        self.weight_k = nn.Linear(latent_dim,latent_dim)
        self.weight_v = nn.Linear(latent_dim,latent_dim)
        self.register_buffer("_num_evals", torch.tensor(0.))
        self.query = None

    def forward(self, t, states):
        assert len(states) == 5 # z, A, C, K, V
        z = states[0]
        K = states[3]
        V = states[4]
        
        # increment num evals
        self._num_evals += 1
        # convert to tensor
        #t = torch.tensor(t).type_as(z)
        #n_traj, n_tp, n_dims = self.query.size()[0], self.query.size()[1], self.query.size()[2]
        
        # compute grad
        dz = torch.squeeze(self.get_ode_gradient_nn(t, z)) # (n_traj, n_dims)
        dK = torch.exp(self.weight_k(z)) # (n_traj, n_dims)
        v = self.weight_v(z) # (n_traj, n_dims)
        dV = torch.bmm(v.unsqueeze(0).permute(1,2,0), dK.unsqueeze(1)) # (n_traj, n_dims, n_dims)
        
        # compute dA
        q = torch.exp(self.weight_q(z))
        a = torch.bmm(q.unsqueeze(1), dK.unsqueeze(0).permute(1,2,0)).reshape(-1,1)
        # (n_traj, 1, n_dims)  X (n_traj, n_dims, 1) == (n_traj, 1)

        dq = q*self.weight_q(dz)
        da = a + torch.bmm(dq.unsqueeze(1), K.unsqueeze(0).permute(1,2,0)).reshape(-1,1)
        # (n_traj, 1, n_dims)  X (n_traj, n_dims, 1) == (n_traj, 1)
        
        # compute dC
        dc = (torch.bmm(dV, q.unsqueeze(0).permute(1,2,0)) + torch.bmm(V, dq.unsqueeze(0).permute(1,2,0))).squeeze()
        # (n_traj, n_dims, n_dims) X (n_traj, n_dims, 1) + (n_traj, n_dims, n_dims) X  (n_traj, n_dims, 1) == (n_traj, n_dims)
        
        return tuple([dz, da, dc, dK, dV])
        
    def get_ode_gradient_nn(self, t_local, y):
        dx = torch.squeeze(y)
        for l, layer in enumerate(self.ode_func_list):
            if isinstance(layer, utils.LAYER[self.layer_type]):
                dx = layer(t_local, dx)
            else:
                dx = layer(dx)
        dx = torch.unsqueeze(dx,0)
        return dx
