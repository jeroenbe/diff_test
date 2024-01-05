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

from run_utils import *
    

LINEAR_SEMS = [
    "gauss",
    "exp",
    "gumbel",
    "uniform",
    "logistic",
    "poisson",
]
NONLINEAR_SEMS = [
    "mlp",
    "mim",
    "gp",
    "gp-add",
]


@click.command()
@click.option('-s0', '--s0', required=True, type=int, default=20)
@click.option('-n', '--n', required=True, type=int, default=100)
@click.option('-r', '--run-count', required=True, type=int, default=100)
@click.option('-d', '--d', required=True, type=int, default=20)
@click.option('-g', '--graph-type', required=True, type=click.Choice(['ER', 'SF', 'BP'], case_sensitive=False), default='ER')
@click.option('-sem', '--sem-type', required=True, type=click.Choice([*LINEAR_SEMS, *NONLINEAR_SEMS], case_sensitive=False), default='gauss')
@click.option('-m', '--methods', required=True, type=click.Choice([
    'notears', 'notears-np', 'dagma-np', 'dagma', 'ges'
]), multiple=True)
@click.option('--group', required=True, type=str)
def cli(
    s0, n, run_count, d, graph_type, sem_type, methods, group
):
    

    config = {
        "s0": s0, 
        "n": n, 
        "run_count": run_count, 
        "d": d, 
        "graph_type": graph_type, 
        "sem_type": sem_type, 
        "methods": methods,
    }

    

    Xs, Xs_normalised, Xs_inverted = np.ndarray((0, n, d)), np.ndarray((0, n, d)), np.ndarray((0, n, d))

    methods = list(set(methods))

    # Let's simulate (and keep) the DAG
    B_true = utils.simulate_dag(d, s0, graph_type)
    

    for _ in range(run_count):
        if sem_type in LINEAR_SEMS:
            W_true = utils.simulate_parameter(B_true)
            X = utils.simulate_linear_sem(W_true, n, sem_type)
        else:
            X = utils.simulate_nonlinear_sem(B_true, n, sem_type)
        
        Xs = np.append(Xs, [X], axis=0)
        Xs_normalised = np.append(Xs_normalised, [X / X.std(axis=0) + X.mean(axis=0)], axis=0)
        # TODO: set inverted


    for m in methods:
        for i in range(run_count):
            X = Xs[i]
            X_normalised = Xs_normalised[i]
            # TODO: inverted

            if m == 'notears':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})

                B_est, B_est_normalised = _run_notears(X, X_normalised)
                log_performance(B_true, B_est, B_est_normalised)

                wandb.finish()

            if m == 'notears-np':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})

                B_est, B_est_normalised = _run_notearsnp(X, X_normalised, d)

                log_performance(B_true, B_est, B_est_normalised)

                wandb.finish()
                
            if m == 'dagma-np':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_normalised = _run_dagmanp(X, X_normalised, d)
                
                log_performance(B_true, B_est, B_est_normalised)
                
                wandb.finish()

            if m == "dagma":
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_normalised = _run_dagma(X, X_normalised)
                
                log_performance(B_true, B_est, B_est_normalised)
                
                wandb.finish()
                
            if m == "ges":
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_normalised = _run_ges(X, X_normalised)
                
                log_performance(B_true, B_est, B_est_normalised)
                
                wandb.finish()






if __name__ == '__main__':
    wandb.login()
    torch.set_default_dtype(torch.double)
    cli()