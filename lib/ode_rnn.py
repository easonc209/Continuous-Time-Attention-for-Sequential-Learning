###########################
# Continuous-Time Attention for Sequential Learning
# Author: Yi-Hsiang Chen
# Modify from: https://github.com/YuliaRubanova/latent_ode
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline


class ODE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100, use_att = False,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = (input_dim) *2, # input and the mask, +1 for meld, *2 for others
            z0_diffeq_solver = z0_diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.att = use_att
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, _, latent_ys, ode_info = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = False)
        
        latent_ys = latent_ys.permute(0,2,1,3)
        last_hidden = latent_ys[:,:,-1,:]

        
        outputs = latent_ys
        if self.att :
            latent_ys, attScore = utils.run_attention(latent_ys)
        
        if self.train_classif_w_reconstr:
            outputs = self.decoder(latent_ys)
            # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
            first_point = data[:,0,:]
            outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"ode_info": ode_info}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(latent_ys)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info


########################################
class ODE_RNN_att(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        diffeq_solver = None, n_gru_units = 100,  n_units = 100,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN_att( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = (input_dim) * 2, # input and the mask
            diffeq_solver = diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.diffeq_solver = diffeq_solver
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, _, latent_ys, context_vector, ode_info = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = True)
        
        latent_ys = latent_ys.permute(0,2,1,3)

        context_vector = latent_ys + utils.reverse_dim1(context_vector)
        last_hidden = context_vector[:,:,-1,:]
        
        outputs = context_vector
        if self.train_classif_w_reconstr:
            outputs = self.decoder(context_vector)
            # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
            first_point = data[:,0,:]
            outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"ode_info": ode_info}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(context_vector)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info 


class ODE_RNN_single_att(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        diffeq_solver = None, n_gru_units = 100,  n_units = 100,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN_single_att( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = (input_dim) + 1 , # input and the mask, +1 for meld
            diffeq_solver = diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.diffeq_solver = diffeq_solver

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None, save_info = False):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, latent_ys, tuple_sol, ode_info = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = False , save_info= save_info)
        
        # reverse
        #context_vector = utils.reverse_dim1(context_vector)
        context_vector = tuple_sol[1]/tuple_sol[0]
        
        latent_ys = latent_ys.permute(0,2,1,3) # permute to (n_traj, n_tp, n_dims)
        last_hidden = latent_ys[:, :,-1, :]
        
        #print("latent_ys", latent_ys.shape)
        #print("context_vector", context_vector.shape)

        context_vector = context_vector+latent_ys
            #assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        outputs = self.decoder(context_vector)
        # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
        first_point = data[:,0,:]
        outputs = utils.shift_outputs(outputs, first_point)

        #extra_info = {"first_point": (latent_ys[:,:,-1,:], 0.0, latent_ys[:,:,-1,:])}
        extra_info = {"ode_info": ode_info}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(context_vector)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj, n_tp, n_dims]
        return outputs, extra_info        

########################################

class ODE_RNN_causal_att(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN_causal_att( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = (input_dim) * 2, # input and the mask
            z0_diffeq_solver = z0_diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, _, latent_ys, context_vector, ode_info = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = False)
        
        
        latent_ys = latent_ys.permute(0,2,1,3)
        
        context_vector = nn.Tanh()(context_vector)+latent_ys 
        last_hidden = context_vector[:,:,-1,:]

        outputs = context_vector
        if self.train_classif_w_reconstr:
            outputs = self.decoder(context_vector)
            # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
            first_point = data[:,0,:]
            outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"ode_info": ode_info}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(context_vector)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info    


class ODE_RNN_causal_single_att(Baseline):
    def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
        z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
        concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
        classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False):

        Baseline.__init__(self, input_dim, latent_dim, device = device, 
            obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        ode_rnn_encoder_dim = latent_dim
    
        self.ode_gru = Encoder_z0_ODE_RNN_causal_single_att( 
            latent_dim = ode_rnn_encoder_dim, 
            input_dim = input_dim + 1, # input and the mask
            z0_diffeq_solver = z0_diffeq_solver, 
            n_gru_units = n_gru_units, 
            device = device).to(device)

        self.z0_diffeq_solver = z0_diffeq_solver

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),)

        utils.init_network_weights(self.decoder)


    def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
        mask = None, n_traj_samples = None, mode = None):

        if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
            raise Exception("Extrapolation mode not implemented for ODE-RNN")

        # time_steps_to_predict and truth_time_steps should be the same 
        assert(len(truth_time_steps) == len(time_steps_to_predict))
        assert(mask is not None)
        
        data_and_mask = data
        if mask is not None:
            data_and_mask = torch.cat([data, mask],-1)

        _, latent_ys, context_vector, ode_info = self.ode_gru.run_odernn(
            data_and_mask, truth_time_steps, run_backwards = False)
        
        
        latent_ys = latent_ys.permute(0,2,1,3)
        
        context_vector = nn.Tanh()(context_vector)+latent_ys 
        last_hidden = context_vector[:,:,-1,:]

        outputs = context_vector
        if self.train_classif_w_reconstr:
            outputs = self.decoder(context_vector)
            # Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
            first_point = data[:,0,:]
            outputs = utils.shift_outputs(outputs, first_point)

        extra_info = {"ode_info": ode_info}

        if self.use_binary_classif:
            if self.classif_per_tp:
                extra_info["label_predictions"] = self.classifier(context_vector)
            else:
                extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

        # outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
        return outputs, extra_info   