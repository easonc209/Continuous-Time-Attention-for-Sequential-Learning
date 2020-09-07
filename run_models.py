import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection, metrics

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model
from lib.ode_func import ODEFunc, ODEFunc_att, ODEFunc_causal_att, ODEFunc_selfatt
from lib.diffeq_solver import DiffeqSolver, DiffeqSolver_att, DiffeqSolver_causal_att
from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
# paremater for dataset
parser.add_argument('-n',  type=int, default=2000, help="Size of the dataset")
parser.add_argument('--dataset', type=str, default='meld', help="Dataset to load. Available: physionet, activity, meld")

# paremater for training
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

# file path
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")

# model type
parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")

# model structure
parser.add_argument('--att', action='store_true', help="Use attention mechanism.")
parser.add_argument('--catt', action='store_true', help="Use continuous attention mechanism.")
parser.add_argument('--catt-v2', action='store_true', help="Use continuous attention mechanism v2.")
parser.add_argument('--with-t', action='store_true', help="Use create net with time.")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

# paremater for model
parser.add_argument('-l', '--latents', type=int, default=50, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--rec-layers', type=int, default=3, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=3, help="Number of layers in ODE func in generative ODE")
parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

# classification
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")

# visualization
parser.add_argument('--viz', action='store_true', help="Show plots while training")

# other

parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
    "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
    "Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
    "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
save_dir = os.path.join(args.save, args.dataset)
utils.makedirs(save_dir)


if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
            # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(save_dir, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))
        
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    # Load data

    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]

    classif_per_tp = False
    if ("classif_per_tp" in data_obj):
        # do classification per time point rather than on a time series as a whole
        classif_per_tp = data_obj["classif_per_tp"]

    if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
        raise Exception("Classification task is not available for MuJoCo and 1d datasets")

    n_labels = 1
    if args.classif:
        if ("n_labels" in data_obj):
            n_labels = data_obj["n_labels"]
        else:
            raise Exception("Please provide number of labels for classification task")

    ##################################################################
    # Create Model
    obsrv_std = 0.01
    if args.dataset == "hopper":
        obsrv_std = 1e-3 

    obsrv_std = torch.Tensor([obsrv_std]).to(device)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    
    experimentType = args.dataset + "/"
    
    if args.classic_rnn:
        if args.poisson:
            print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

        if args.extrap:
            raise Exception("Extrapolation for standard RNN not implemented")
        # Create RNN model
        use_att = False
        if args.att:
            experimentType += "att_"
            use_att = True
        experimentType += "rnn/"
        
        model = Classic_RNN(input_dim, args.latents, device, 
            concat_mask = True,
            n_units = args.units,
            use_binary_classif = args.classif,
            classif_per_tp = classif_per_tp,
            linear_classifier = args.linear_classif,
            input_space_decay = args.input_decay,
            cell = args.rnn_cell,
            n_labels = n_labels,
            train_classif_w_reconstr = (args.dataset == "physionet"),
            use_att = use_att
            ).to(device)
    elif args.ode_rnn:
        # Create ODE-GRU model
        n_ode_gru_dims = args.latents
                
        if args.poisson:
            print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

        if args.extrap:
            raise Exception("Extrapolation for ODE-RNN not implemented")
            
        layer_type = "linear"
        if args.with_t:
            layer_type = "concat"
            experimentType += "wt_"
        
        if args.catt:
            experimentType += "odernncatt/"
            ode_func_net = utils.create_net_with_time(n_ode_gru_dims, n_ode_gru_dims, 
                n_layers = args.rec_layers, n_units = args.units, layer_type = layer_type)

            rec_ode_func = ODEFunc_att( # ODEFunc_selfatt
                input_dim = input_dim, 
                latent_dim = n_ode_gru_dims,
                ode_func_net = ode_func_net,
                device = device).to(device)

            diffeq_solver = DiffeqSolver_att(input_dim, rec_ode_func, "dopri5", args.latents, 
                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
            model = ODE_RNN_att(input_dim, n_ode_gru_dims, device = device, # ODE_RNN_single_att
                diffeq_solver = diffeq_solver, n_gru_units = args.gru_units,
                concat_mask = True,
                use_binary_classif = args.classif,
                classif_per_tp = classif_per_tp,
                n_labels = n_labels,
                train_classif_w_reconstr = (args.dataset == "physionet")
                ).to(device)
        elif args.catt_v2 :
            experimentType += "odernncatt_v2/"
            ode_func_net = utils.create_net_with_time(n_ode_gru_dims, n_ode_gru_dims, 
                n_layers = args.rec_layers, n_units = args.units, layer_type = layer_type)

            rec_ode_func = ODEFunc_causal_att(
                input_dim = input_dim, 
                latent_dim = n_ode_gru_dims,
                ode_func_net = ode_func_net,
                device = device).to(device)

            z0_diffeq_solver = DiffeqSolver_causal_att(input_dim, rec_ode_func, "dopri5", args.latents, 
                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
            model = ODE_RNN_causal_att(input_dim, n_ode_gru_dims, device = device, # ODE_RNN_causal_single_att
                z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
                concat_mask = True, 
                use_binary_classif = args.classif,
                classif_per_tp = classif_per_tp,
                n_labels = n_labels,
                train_classif_w_reconstr = (args.dataset == "physionet")
                ).to(device)
        else:
            use_att = False
            if args.att:
                experimentType += "att_"
                use_att = True
            experimentType += "odernn/"
            
            ode_func_net = utils.create_net_with_time(n_ode_gru_dims, n_ode_gru_dims, 
                n_layers = args.rec_layers, n_units = args.units, layer_type = layer_type)

            rec_ode_func = ODEFunc(
                input_dim = input_dim, 
                latent_dim = n_ode_gru_dims,
                ode_func_net = ode_func_net,
                device = device).to(device)

            z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "dopri5", args.latents, 
                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
        
            model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
                z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units, use_att = use_att,
                concat_mask = True, 
                use_binary_classif = args.classif,
                classif_per_tp = classif_per_tp,
                n_labels = n_labels,
                train_classif_w_reconstr = (args.dataset == "physionet")
                ).to(device)
    elif args.latent_ode:
        experimentType += "latentode/"
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
            classif_per_tp = classif_per_tp,
            n_labels = n_labels)
    else:
        raise Exception("Model not specified")
    ##################################################################
    # Training

    log_path = "logs/" + experimentType + file_name + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/" + experimentType):
        utils.makedirs("logs/" + experimentType)
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info("number of parameters: {}".format(sum(param.numel() for param in model.parameters())))
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    sum_loss = torch.zeros(1)
    print('num_batches', num_batches)
    for itr in range(1, num_batches * (args.niters + 1)):
        model.train()
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
        
        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        startTime = time.time()
        train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()
        optimizer.step()
        endTime = time.time()
        logger.info("Train loss (one batch): {} | mse : {} | ce_loss {} | Cost time (one batch): {}".format(train_res["loss"].detach(), train_res["mse"].detach(), train_res["ce_loss"].detach(), endTime-startTime))
        sum_loss += train_res["loss"].detach()
        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * num_batches) == 0:
            with torch.no_grad():
                model.eval()
                test_res = compute_loss_all_batches(model, 
                    data_obj["test_dataloader"], args,
                    n_batches = data_obj["n_test_batches"],
                    experimentID = experimentID,
                    device = device,
                    n_traj_samples = 3, kl_coef = kl_coef)

                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f}'.format(
                    itr//num_batches, test_res["loss"].detach(), )

                logger.info("Experiment " + str(experimentID))
                logger.info(message)
                logger.info("KL coef: {}".format(kl_coef))
                logger.info("Train loss (epoch): {}".format(sum_loss.cpu().item()/num_batches))
                sum_loss = 0

                if "auc" in test_res:
                    logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

                if "mse" in test_res:
                    logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

                if "accuracy" in train_res:
                    logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

                if "accuracy" in test_res:
                    logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

                if "classification_report" in test_res:
                    logger.info("classification_report:")
                    logger.info(test_res["classification_report"])

                if "ce_loss" in test_res:
                    logger.info("CE loss: {}".format(test_res["ce_loss"]))

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)


    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
