import numpy as np
import click, wandb

import torch

from notears import utils, linear, nonlinear
from notears.nonlinear import NotearsMLP

from dagma.src.models.linear_dagma import DagmaLinear
from dagma.src.models.nonlinear_dagma import DagmaNonlinear, DagmaMLP
from dagma.src.utils import convert_pc_to_adjacency
from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_cv_general

def log_performance(B_true, B_est, B_est_normalised, exp_type: str="normalised"):
    perf, perf_normalised = eval(B_true, B_est), eval(B_true, B_est_normalised, exp_type=exp_type)

    wandb.log({**perf, **perf_normalised})
    wandb.log({'B_est': B_est})
    wandb.log({f'B_est_{counter_naming}': B_est_normalised})

    perf_diff = eval(B_est, B_est_normalised)

    wandb.log({'DAG-diff': perf_diff['shd']})


def eval(B, B_est, exp_type=None):
    perf = utils.count_accuracy(B, B_est)
    exp_type = '' if exp_type is None else f"_{exp_type}"
    return {f"{k}{exp_type}": v for k, v in perf.items()}


def _run_notearsnp(X, X_normalised, d):
    model = NotearsMLP(dims=[d, 10, 1], bias=True)#.double()
    model_normalised = NotearsMLP(dims=[d, 10, 1], bias=True)#.double()

    W_est = nonlinear.notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    W_est_normalised = nonlinear.notears_nonlinear(model_normalised, X_normalised, lambda1=0.01, lambda2=0.01)

    B_est, B_est_normalised = W_est, W_est_normalised
    B_est[B_est != 0] = 1 # assumes W_est is already thresholded for 0.
    B_est_normalised[B_est_normalised != 0] = 1 # assumes W_est is already thresholded for 0.

    return B_est, B_est_normalised

def _run_notears(X, X_normalised):
    W_est = linear.notears_linear(X, lambda1=0.1, loss_type='l2')
    W_est_normalised = linear.notears_linear(X_normalised, lambda1=0.1, loss_type='l2')
    
    B_est, B_est_normalised = W_est, W_est_normalised
    B_est[B_est != 0] = 1 # assumes W_est is already thresholded for 0.
    B_est_normalised[B_est_normalised != 0] = 1 # assumes W_est is already thresholded for 0.

    return B_est, B_est_normalised

def _run_dagmanp(X, X_normalised, d):
    eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)#.double()
    eq_model_normalised = DagmaMLP(dims=[d, 10, 1], bias=True)#.double()
    
    model = DagmaNonlinear(eq_model, dtype=torch.double, verbose = True)
    model_normalised = DagmaNonlinear(eq_model_normalised, dtype=torch.double, verbose = True)
    
    W_est = model.fit(X, lambda1=0.02, lambda2=0.005)
    W_est_normalised = model_normalised.fit(X_normalised, lambda1=0.02, lambda2=0.005)

    B_est, B_est_normalised = W_est, W_est_normalised
    B_est[B_est != 0] = 1 # assumes W_est is already thresholded for 0.
    B_est_normalised[B_est_normalised != 0] = 1 # assumes W_est is already thresholded for 0.

    return B_est, B_est_normalised

def _run_dagma(X, X_normalised):
    
    model = DagmaLinear(loss_type='l2') # create a linear model with least squares loss
    model_normalised = DagmaLinear(loss_type='l2') # create a linear model with least squares loss
    
    W_est = model.fit(X, lambda1=0.02) # fit the model with L1 reg. (coeff. 0.02)
    W_est_normalised = model_normalised.fit(X_normalised, lambda1=0.1)
    
    B_est, B_est_normalised = W_est, W_est_normalised
    B_est[B_est != 0] = 1 # assumes W_est is already thresholded for 0.
    B_est_normalised[B_est_normalised != 0] = 1 # assumes W_est is already thresholded for 0.

    return B_est, B_est_normalised

def _run_ges(X, X_normalised):
    Record = ges(X)
    graph_ges = (Record["G"].graph)
    B_est = convert_pc_to_adjacency(graph_ges)
    
    Record_normalized = ges(X_normalised)
    graph_ges_normalized = (Record_normalized["G"].graph)
    B_est_normalised = convert_pc_to_adjacency(graph_ges_normalized)
    
    return B_est, B_est_normalised