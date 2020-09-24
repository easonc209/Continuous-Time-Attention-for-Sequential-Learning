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
from torch.distributions import Categorical, Normal
import lib.utils as utils
from torch.nn.modules.rnn import LSTM, GRU
from lib.utils import get_device


# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100,
        device = torch.device("cpu")):
        super(GRU_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            utils.init_network_weights(self.update_gate)
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            utils.init_network_weights(self.reset_gate)
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim * 2 + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim * 2))
            utils.init_network_weights(self.new_state_net)
        else: 
            self.new_state_net  = new_state_net


    def forward(self, y_mean, y_std, x, masked_update = True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
        
        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        assert(not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1)//2
            mask = x[:, :, n_data_dims:]
            #utils.check_mask(x[:, :, :n_data_dims], mask)
            
            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            assert(not torch.isnan(mask).any())

            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std


class GRU_single_unit(nn.Module):
    def __init__(self, latent_dim, input_dim, 
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        n_units = 100,
        device = torch.device("cpu")):
        super(GRU_single_unit, self).__init__()

        if update_gate is None:
            self.update_gate = nn.Sequential(
               nn.Linear(latent_dim + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            utils.init_network_weights(self.update_gate)
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
               nn.Linear(latent_dim + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim),
               nn.Sigmoid())
            utils.init_network_weights(self.reset_gate)
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
               nn.Linear(latent_dim + input_dim, n_units),
               nn.Tanh(),
               nn.Linear(n_units, latent_dim))
            utils.init_network_weights(self.new_state_net)
        else: 
            self.new_state_net  = new_state_net


    def forward(self, y_mean, x, masked_update = True):
        y_concat = torch.cat([y_mean, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, x], -1)
        
        new_state = self.new_state_net(concat)

        new_y = (1-update_gate) * new_state + update_gate * y_mean

        assert(not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            #n_data_dims = x.size(-1)//2
            mask = x.squeeze()[:, -1].reshape(-1,1)
            #utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            assert(not torch.isnan(mask).any())
            
            new_y = mask * new_y + (1-mask) * y_mean

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        return new_y


#######################################

class Encoder_z0_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, lstm_output_size = 20, 
        use_delta_t = True, device = torch.device("cpu")):
        
        super(Encoder_z0_RNN, self).__init__()
    
        self.gru_rnn_output_size = lstm_output_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.use_delta_t = use_delta_t

        self.hiddens_to_z0 = nn.Sequential(
           nn.Linear(self.gru_rnn_output_size, 50),
           nn.Tanh(),
           nn.Linear(50, latent_dim * 2),)

        utils.init_network_weights(self.hiddens_to_z0)

        input_dim = self.input_dim

        if use_delta_t:
            self.input_dim += 1
        self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

    def forward(self, data, time_steps, run_backwards = True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)
        # t0: not used here
        n_traj = data.size(0)

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        data = data.permute(1,0,2) 

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = utils.reverse(data)

        if self.use_delta_t:
            delta_t = time_steps[1:] - time_steps[:-1]
            if run_backwards:
                # we are going backwards in time with
                delta_t = utils.reverse(delta_t)
            # append zero delta t in the end
            delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
            delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
            data = torch.cat((delta_t, data),-1)

        outputs, _ = self.gru_rnn(data)

        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

        mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
        z0_dim = None, GRU_update = None, 
        n_gru_units = 100, 
        device = torch.device("cpu")):
        
        super(Encoder_z0_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, 100),
           nn.Tanh(),
           nn.Linear(100, self.z0_dim * 2),)
        utils.init_network_weights(self.transform_z0)


    def forward(self, data, time_steps, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0


    def run_odernn(self, data, time_steps, 
        run_backwards = True, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        device = get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        #print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert(not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert(not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()
            #assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std            
            prev_t, t_i = time_steps[i],  time_steps[i-1]

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
                     "yi": yi.detach(), "yi_std": yi_std.detach(), 
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
           nn.Linear(latent_dim, input_dim),)

        utils.init_network_weights(decoder)    
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)

#########################################

class Encoder_z0_ODE_RNN_att(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, diffeq_solver = None, 
        z0_dim = None, GRU_update = None, n_gru_units = 100,
        device = torch.device("cpu")):
        
        super(Encoder_z0_ODE_RNN_att, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.diffeq_solver = diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None
        
        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim * 2, 100),
           nn.Tanh(),
           nn.Linear(100, self.z0_dim * 2),)
        utils.init_network_weights(self.transform_z0)


    def forward(self, data, time_steps, minimum_step = None, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, lt, context_vector, extra_info = self.run_odernn(
                data, time_steps, minimum_step = minimum_step, run_backwards = run_backwards,
                save_info = save_info)
            last_yi = last_yi + utils.reverse_dim1(context_vector)[:,-1,:]
        
        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)
        
        mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0


    def run_odernn(self, data, time_steps, minimum_step = None,
        run_backwards = True, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []
        device = self.device

        yi_ode = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50
        #print("minimum step: {}".format(minimum_step))
        
        self.diffeq_solver.ode_func.set_query(n_tp)
        
        init_condition = (torch.zeros([n_traj, n_tp, 1]).to(device), # A
                          torch.zeros([n_traj, n_tp, self.latent_dim]).to(device)) # C

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            xi = data[:,i,:].unsqueeze(0)
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
            
            prev_y, prev_std = yi, yi_std
            if i-1>=0:
                prev_t, t_i = time_steps[i], time_steps[i-1]
            else :
                prev_t, t_i = time_steps[i], time_steps[i]-0.01

            latent_ys.append(yi)
            
            if ( prev_t - t_i ) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                self.diffeq_solver.ode_func.feed_query(prev_y.squeeze())
                tuple_sol = self.diffeq_solver.ode_func(prev_t, (prev_y.squeeze(),) + init_condition) 

                ode_sol = prev_y + tuple_sol[0].unsqueeze(0) * (t_i - prev_t)
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)
                init_condition = tuple(i+ j*(t_i - prev_t) for i,j in zip(init_condition, tuple_sol[1:]))
                assert(not torch.isnan(ode_sol).any())
            else :
                n_intermediate_tp = 2
                if minimum_step is not None:
                    n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())
                
                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                self.diffeq_solver.ode_func.feed_query(prev_y.squeeze())
                tuple_sol = self.diffeq_solver(prev_y, time_points,  init_condition = init_condition)
                ode_sol = tuple_sol[0]
                init_condition = tuple(i[-1] for i in tuple_sol[1:])
                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            
            if save_info:
                d = {"att_score": sum_att_score[:,0,:].detach().cpu().numpy(),
                     "time_points": time_points.detach().cpu().numpy()}
                extra_info.append(d)
                

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())
        c = init_condition[1]/init_condition[0]
        return yi, yi_std, latent_ys, c, extra_info


class Encoder_z0_ODE_RNN_single_att(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_dim = None, n_gru_units = 100
        , diffeq_solver = None, GRU_update = None, 
        device = torch.device("cpu")):
        
        super(Encoder_z0_ODE_RNN_single_att, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_single_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.diffeq_solver = diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None


    def forward(self, data, time_steps, minimum_step = None, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, _, sum_att_score, context_vector, extra_info = self.run_odernn(
                data, time_steps, minimum_step = minimum_step, run_backwards = run_backwards,
                save_info = save_info)
        
        means_z0 = last_yi.reshape(n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(n_traj, self.latent_dim)
        
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0, sum_att_score, context_vector


    def run_odernn(self, data, time_steps, minimum_step = None,
        run_backwards = False, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []
        device = self.device

        yi_ode = torch.zeros((n_traj, self.latent_dim)).to(device)
        
        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50
        #print("minimum step: {}".format(minimum_step))
        
        # set query
        self.diffeq_solver.ode_func.set_query(data, time_steps)
        
        # initial condition
        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        init_condition = ( torch.zeros([n_traj, n_tp, 1]).to(device), # sum_att_score
                           torch.zeros([n_traj, n_tp, self.latent_dim]).to(device)) # context_vector
        prev_t, t_i = time_steps[0] - 1e-3,  time_steps[0]
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        #if run_backwards:
        #    time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (t_i - prev_t) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                tuple_sol = self.diffeq_solver.ode_func(prev_t, (prev_y.squeeze(),) + init_condition) 

                ode_sol = prev_y + tuple_sol[0].unsqueeze(0) * (t_i - prev_t)
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)
                init_condition = tuple(i+ j*(t_i - prev_t) for i,j in zip(init_condition, tuple_sol[1:]))
                assert(not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = 2
                if minimum_step is not None:
                    n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())
                
                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                tuple_sol = self.diffeq_solver(prev_y, time_points, init_condition=init_condition)
                
                ode_sol = tuple_sol[0]
                init_condition = tuple(i[-1] for i in tuple_sol[1:])

                assert(not torch.isnan(ode_sol).any())
            
            
            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            yi = self.GRU_update(yi_ode, xi)

            prev_y = yi
            if i+1<len(time_steps):
                prev_t, t_i = time_steps[i],  time_steps[i+1]
            else :
                prev_t = time_steps[i]

            latent_ys.append(yi)

            if save_info:
                d = {"att_score": tuple_sol[1].detach().cpu().numpy(),
                     "con_vect" : tuple_sol[2].detach().cpu().numpy(),
                     "time_points": time_points.detach().cpu().numpy()}
                extra_info.append(d)
                

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        
        return yi, latent_ys, init_condition, extra_info
        
#########################################

class Encoder_z0_ODE_RNN_causal_att(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_dim = None, n_gru_units = 100
        , z0_diffeq_solver = None, GRU_update = None, 
        device = torch.device("cpu")):
        
        super(Encoder_z0_ODE_RNN_causal_att, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None


    def forward(self, data, time_steps, minimum_step = None, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, _, sum_att_score, context_vector, extra_info = self.run_odernn(
                data, time_steps, minimum_step = minimum_step, run_backwards = run_backwards,
                save_info = save_info)
        
        means_z0 = last_yi.reshape(n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(n_traj, self.latent_dim)
        
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0, sum_att_score, context_vector


    def run_odernn(self, data, time_steps, minimum_step = None,
        run_backwards = False, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        #t0 = time_steps[-1]
        #if run_backwards:
        #    t0 = time_steps[0]

        device = self.device

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        init_condition = (torch.zeros((n_traj, 1)).to(device), #A
                          torch.zeros((n_traj, self.latent_dim)).to(device), #C
                          torch.zeros((n_traj, self.latent_dim)).to(device), #K
                          torch.zeros((n_traj, self.latent_dim, self.latent_dim)).to(device)) #V

        prev_t, t_i = time_steps[0] - 0.01,  time_steps[0]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        #print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        record_condition = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        #if run_backwards:
        #    time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if ( t_i - prev_t ) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                tuple_sol = self.diffeq_solver.ode_func(prev_t, (prev_y.squeeze(),) + init_condition) 
                
                ode_sol = prev_y + tuple_sol[0].unsqueeze(0) * (t_i - prev_t)
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)
                init_condition = tuple(i+ j*(t_i - prev_t) for i,j in zip(init_condition, tuple_sol[1:]))
                assert(not torch.isnan(ode_sol).any())
            else :
                n_intermediate_tp = 2
                if minimum_step is not None:
                    n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())
                    
                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                tuple_sol = self.diffeq_solver(prev_y, time_points,  init_condition = init_condition)
                ode_sol = tuple_sol[0]
                init_condition = tuple(i[-1] for i in tuple_sol[1:])
                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()
            #assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
            
            prev_y, prev_std = yi, yi_std
            if i+1<len(time_steps):
                prev_t, t_i = time_steps[i],  time_steps[i+1]
            else :
                prev_t = time_steps[i]

            latent_ys.append(yi)
            record_condition.append(init_condition[1:3]) # [n_tp, ((n_traj, 1), (n_traj, n_dims))]

            if save_info:
                d = {"att_score": sum_att_score[:,0,:].detach().cpu().numpy(),
                     "time_points": time_points.detach().cpu().numpy()}
                extra_info.append(d)
                

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())
        
        context_vector = torch.stack([ i[1]/i[0] for i in record_condition], 1) # (n_traj, n_tp, n_dims)
        return yi, yi_std, latent_ys, context_vector, extra_info


class Encoder_z0_ODE_RNN_causal_single_att(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_dim = None, n_gru_units = 100
        , z0_diffeq_solver = None, GRU_update = None, 
        device = torch.device("cpu")):
        
        super(Encoder_z0_ODE_RNN_causal_single_att, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_single_unit(latent_dim, input_dim, 
                n_units = n_gru_units, 
                device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None


    def forward(self, data, time_steps, minimum_step = None, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())
        
        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            
            last_yi, last_yi_std, _, sum_att_score, context_vector, extra_info = self.run_odernn(
                data, time_steps, minimum_step = minimum_step, run_backwards = run_backwards,
                save_info = save_info)
        
        means_z0 = last_yi.reshape(n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(n_traj, self.latent_dim)
        
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0, sum_att_score, context_vector


    def run_odernn(self, data, time_steps, minimum_step = None,
        run_backwards = False, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 

        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        device = self.device

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        #prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        init_condition = (torch.zeros((n_traj, 1)).to(device), #A
                          torch.zeros((n_traj, self.latent_dim)).to(device), #C
                          torch.zeros((n_traj, self.latent_dim)).to(device), #K
                          torch.zeros((n_traj, self.latent_dim, self.latent_dim)).to(device)) #V

        prev_t, t_i = time_steps[0] - 0.01,  time_steps[0]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        #print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        record_condition = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        #if run_backwards:
        #    time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if ( t_i - prev_t ) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                tuple_sol = self.diffeq_solver.ode_func(prev_t, (prev_y.squeeze(),) + init_condition) 
                
                ode_sol = prev_y + tuple_sol[0].unsqueeze(0) * (t_i - prev_t)
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)
                init_condition = tuple(i+ j*(t_i - prev_t) for i,j in zip(init_condition, tuple_sol[1:]))
                assert(not torch.isnan(ode_sol).any())
            else :
                n_intermediate_tp = 2
                if minimum_step is not None:
                    n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())
                    
                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
                tuple_sol = self.diffeq_solver(prev_y, time_points,  init_condition = init_condition)
                ode_sol = tuple_sol[0]
                init_condition = tuple(i[-1] for i in tuple_sol[1:])
                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()
            #assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            
            yi = self.GRU_update(yi_ode, xi)
            
            prev_y = yi
            if i+1<len(time_steps):
                prev_t, t_i = time_steps[i],  time_steps[i+1]
            else :
                prev_t = time_steps[i]

            latent_ys.append(yi)
            record_condition.append(init_condition[1:3]) # [n_tp, ((n_traj, 1), (n_traj, n_dims))]

            if save_info:
                d = {"att_score": sum_att_score[:,0,:].detach().cpu().numpy(),
                     "time_points": time_points.detach().cpu().numpy()}
                extra_info.append(d)
                

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        
        context_vector = torch.stack([ i[1]/i[0] for i in record_condition], 1) # (n_traj, n_tp, n_dims)
        return yi, latent_ys, context_vector, extra_info