import numpy as np
import click, joblib

from notears import utils, linear, nonlinear


LINEAR_SEMS = ["gauss", "exp", "gumbel", "uniform", "logistic", "poisson"]
NONLINEAR_SEMS = ["mlp", "mim", "gp", "gp-add"]




@click.command()
@click.option('-s0', '--s0', required=True, type=int, default=20)
@click.option('-n', '--n', required=True, type=int, default=100)
@click.option('-r', '--run-count', required=True, type=int, default=100)
@click.option('-d', '--d', required=True, type=int, default=20)
@click.option('-g', '--graph-type', required=True, type=click.Choice(['ER', 'SF', 'BP'], case_sensitive=False), default='ER')
@click.option('-sem', '--sem-type', required=True, type=click.Choice([*LINEAR_SEMS, *NONLINEAR_SEMS], case_sensitive=False), default='gauss')
@click.option('-m', '--methods', required=True, type=click.Choice([
    'notears', 'notears-np'
]), multiple=True)
def cli(
    s0, n, run_count, d, graph_type, sem_type, methods
):
    Xs, Xs_normalised, Xs_inverted = np.ndarray((0, n, d)), np.ndarray((0, n, d)), np.ndarray((0, n, d))

    methods = list(set(methods))

    # Let's simulate (and keep) the DAG
    B_true = utils.simulate_dag(d, s0, graph_type)
    

    for _ in range(run_count):
        if sem_type in LINEAR_SEMS:
            W_true = utils.simulate_parameter(B_true)
            X = utils.simulate_linear_sem(W_true, n, sem_type)
        else:
            X = utils.simulate_nonlinear_sem(W_true, n, sem_type)
        
        Xs = np.append(Xs, [X], axis=0)
        Xs_normalised = np.append(Xs_normalised, [X / X.std(axis=0) + X.mean(axis=0)])
        # TODO: set inverted


        for m in methods:
            # TODO: learn DAGs
            # TODO: log perf to w&b
            if m == 'notears':
                pass
            if m == 'notears-np':
                pass


        





if __name__ == '__main__':
    cli()