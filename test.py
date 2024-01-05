import numpy as np
import click, wandb

import torch

import igraph as ig

from notears import utils

from run_utils import _run_dagma, _run_dagmanp, _run_ges, _run_notears, _run_notearsnp, log_performance




@click.command()
@click.option('-s0', '--s0', required=True, type=int, default=20)
@click.option('-n', '--n', required=True, type=int, default=100)
@click.option('-r', '--run-count', required=True, type=int, default=100)
@click.option('-d', '--d', required=True, type=int, default=20)
@click.option('-g', '--graph-type', required=True, type=click.Choice(['ER', 'SF', 'BP'], case_sensitive=False), default='ER')
@click.option('-m', '--methods', required=True, type=click.Choice([
    'notears', 'notears-np', 'dagma-np', 'dagma', 'ges'
]), multiple=True)
@click.option('-vs', '--var-sort', required=True, type=click.Choice([
    'exp', 'lin', 'log', 'exp-inv', 'lin-inv', 'log-inv'
]), multiple=False)
@click.option('--group', required=True, type=str)
def cli(
    s0, n, run_count, d, graph_type, methods, var_sort, group
):

    config = {
        "s0": s0, 
        "n": n, 
        "run_count": run_count, 
        "d": d, 
        "graph_type": graph_type, 
        "sem_type": "gauss",
        "var_sort": var_sort,
        "methods": methods,
    }



    methods = list(set(methods))

    # Let's simulate (and keep) the DAG
    B_true = utils.simulate_dag(d, s0, graph_type)

    # Simulate a default Gaussian linear dataset
    W = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(B_true, n, "gauss")


    # Figure out the topological ordering
    g = ig.Graph.Adjacency(B_true, loops=False)
    g.vs["label"] = list(range(d))

    sorting = g.topological_sorting()

    # Normalise the data
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    var_sort_functions = {
        "exp": var_sort_exp,
        "lin": var_sort_lin,
        "log": var_sort_log,

        "exp-inv": var_sort_exp_inv,
        "lin-inv": var_sort_lin_inv,
        "log-inv": var_sort_log_inv,
    }

    X_varsorted = var_sort_functions[var_sort](X_norm, d, sorting)



    print("original", X.std(axis=0))
    print("lin-var-sort", X_varsorted.std(axis=0))


    for m in methods:
        for i in range(run_count):
            if m == 'notears':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})

                B_est, B_est_varsorted = _run_notears(X, X_varsorted)
                log_performance(B_true, B_est, B_est_varsorted)

                wandb.finish()

            if m == 'notears-np':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})

                B_est, B_est_varsorted = _run_notearsnp(X, X_varsorted, d)

                log_performance(B_true, B_est, B_est_varsorted)

                wandb.finish()
                
            if m == 'dagma-np':
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_varsorted = _run_dagmanp(X, X_varsorted, d)
                
                log_performance(B_true, B_est, B_est_varsorted)
                
                wandb.finish()

            if m == "dagma":
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_varsorted = _run_dagma(X, X_varsorted)
                
                log_performance(B_true, B_est, B_est_varsorted)
                
                wandb.finish()
                
            if m == "ges":
                wandb.init(project='structure-learning', config=config, group=group)
                wandb.log({'model': m})
                
                B_est, B_est_varsorted = _run_ges(X, X_varsorted)
                
                log_performance(B_true, B_est, B_est_varsorted)
                
                wandb.finish()


def var_sort_lin(X_norm, d, sorting):
    X_varsorted = X_norm.copy()
    
    vars = np.linspace(1, d, d)

    X_varsorted[:, sorting] *= vars
    return X_varsorted

def var_sort_lin_inv(X_norm, d, sorting):
    X_varsorted = X_norm.copy()
    
    vars = np.linspace(d, 1, d)

    X_varsorted[:, sorting] *= vars
    return X_varsorted

def var_sort_exp(X_norm, d, sorting):
    X_varsorted = X_norm.copy()
    
    vars = np.logspace(1, d, d, base=2)
    vars /= (vars[-1] / (d+1))
    
    X_varsorted[:, sorting] *= vars
    return X_varsorted

def var_sort_exp_inv(X_norm, d, sorting):
    X_varsorted = X_norm.copy()

    vars = np.logspace(1, d, d, base=2)
    vars /= (vars[-1] / (d+1))
    vars = vars[::-1]

    X_varsorted[:, sorting] *= vars
    return X_varsorted

def var_sort_log(X_norm, d, sorting):
    X_varsorted = X_norm.copy()

    vars = np.logspace(1, d, d, base=0.5)
    vars = np.full(vars.shape, vars.max()) - vars
    vars /= (vars[-1] / (d+1))

    X_varsorted[:, sorting] *= vars
    return X_varsorted

def var_sort_log_inv(X_norm, d, sorting):
    X_varsorted = X_norm.copy()

    vars = np.logspace(1, d, d, base=0.5)
    vars = np.full(vars.shape, vars.max()) - vars
    vars /= (vars[-1] / (d+1))
    vars = vars[::-1]

    X_varsorted[:, sorting] *= vars
    return X_varsorted

if __name__ == '__main__':
    wandb.login()
    torch.set_default_dtype(torch.double)
    cli()