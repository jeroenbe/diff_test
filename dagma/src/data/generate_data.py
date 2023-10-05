import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
import torch



def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == 'Fully':
        B = np.triu(np.ones((d,d)), 1)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def simulate_categorical(B, n, list_n_classes, sem_type, deterministic = False):
    """Simulate categorical data only, based on the adjacency matrix and the number of samples

    Args:
        B (np.array): numpy array
        n (int): number of samples
    """
    def _simulate_single_equation(X, n_classes):
        n = len(X)
        #for the leaves we sample from a uniform distribution
        p_uniform = np.ones(n_classes)/n_classes
        samples = np.random.choice(n_classes, size = n, p=p_uniform)
        pa_size = X.shape[1]
        list_entropy = []
        temperature = 1e6 if deterministic else 1

        if pa_size == 0: #if leaves, then we sample from a uniform distribution
            probabilities = np.ones(n_classes)/n_classes
            entropy = -np.sum(probabilities * np.log(probabilities))
           
            return samples, entropy
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=[hidden, n_classes]) 
            W2[np.random.rand(hidden) < 0.5] *= -1
            #!replace this so that we get the logits and sample from this categorical distribution
            logits = sigmoid(X @ W1) @ W2*temperature
            #x is the logits that we use to sample from, of dimensions [n, n_classes]
            probabilities = softmax(logits, axis=1)
            entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
            mean_entropy = np.mean(entropy)
            #print(probabilities)
            samples = []
            
            # Sample from each row's categorical distribution
            for row in probabilities:
                sample = np.random.choice(len(row), p=row)
                samples.append(sample)
                
            samples = np.array(samples)

        if sem_type == "linear":
            W = np.random.uniform(low=0.5, high=2.0, size=[pa_size, n_classes])
            W[np.random.rand(*W.shape) < 0.5] *= -1
            logits = X @ W *temperature
            #x is the logits that we use to sample from, of dimensions [n, n_classes]
            probabilities = softmax(logits, axis=1)
            entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
            mean_entropy = np.mean(entropy)

            samples = [] #store the samples, of dimensions [n]
            
            # Sample from each row's categorical distribution
            for row in probabilities:
                sample = np.random.choice(len(row), p=row)
                samples.append(sample)
            samples = np.array(samples)

        return samples, mean_entropy

    d = B.shape[0]
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    list_entropy = []
    
    for j in ordered_vertices:
        
        parents = G.neighbors(j, mode=ig.IN)
        #! one hot encode the parents 
        X_parents = X[:, parents]
        if len(parents)>0:
            encoder = OneHotEncoder()
            X_parents_onehot = encoder.fit_transform(X_parents).toarray()
        else:
            X_parents_onehot = np.zeros((n,0))
        n_classes = list_n_classes[j]
        
        X[:, j], mean_entropy = _simulate_single_equation(X_parents_onehot, n_classes)
        list_entropy.append((j,mean_entropy))
    return X, list_entropy


def simulate_mixed(B, n, list_categorical_features, list_n_classes, sem_type, deterministic = False):
    """Simulate categorical data only, based on the adjacency matrix and the number of samples

    Args:
        B (np.array): numpy array
        n (int): number of samples
        list_cateogorical_features (list): list containing the categorical features, which will have to be one-hot encoded
    """
    def _simulate_single_equation(X, n_classes, isCategorical):
        n = len(X)
        #for the leaves we sample from a uniform distribution
        
        pa_size = X.shape[1]
        list_entropy = []
        temperature = 1e6 if deterministic else 1

        if pa_size == 0: 
            
            if isCategorical:
                #if leaves, then we sample from a uniform distribution
                p_uniform = np.ones(n_classes)/n_classes
                samples = np.random.choice(n_classes, size = n, p=p_uniform)
                probabilities = np.ones(n_classes)/n_classes
                entropy = -np.sum(probabilities * np.log(probabilities))
            else:
                z = np.random.normal(scale=scale, size=n)
                samples = z
                entropy = None
                
            return samples, entropy
        
        #Not a leaf, hence has to use the parents
        if sem_type == 'mlp':
            hidden = 100
            
            #!replace this so that we get the logits and sample from this categorical distribution
            
            if isCategorical:
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=[hidden, n_classes]) 
                W2[np.random.rand(hidden) < 0.5] *= -1
                
                
                logits = sigmoid(X @ W1) @ W2*temperature
                #x is the logits that we use to sample from, of dimensions [n, n_classes]
                probabilities = softmax(logits, axis=1)
                entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
                mean_entropy = np.mean(entropy)
                #print(probabilities)
                samples = []
                
                # Sample from each row's categorical distribution
                for row in probabilities:
                    sample = np.random.choice(len(row), p=row)
                    samples.append(sample)
                    
                samples = np.array(samples)
            else:
                
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                
                #add some gaussian noise
                temperature_continuous = 1e-6 if deterministic else 1
                
                z = np.random.normal(scale=scale * temperature_continuous, size=n)
                
                samples = sigmoid(X @ W1) @ W2 + z
                mean_entropy = None
                

        if sem_type == "linear":
            
            
            if isCategorical:
                W = np.random.uniform(low=0.5, high=2.0, size=[pa_size, n_classes])
                W[np.random.rand(*W.shape) < 0.5] *= -1
                
                logits = X @ W *temperature
                #x is the logits that we use to sample from, of dimensions [n, n_classes]
                probabilities = softmax(logits, axis=1)
                entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
                mean_entropy = np.mean(entropy)

                samples = [] #store the samples, of dimensions [n]
                
                # Sample from each row's categorical distribution
                for row in probabilities:
                    sample = np.random.choice(len(row), p=row)
                    samples.append(sample)
                samples = np.array(samples)
                
                
            else:
                W = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                W[np.random.rand(*W.shape) < 0.5] *= -1
                
                #add some gaussian noise
                z = np.random.normal(scale=scale, size=n)
                
                samples = X @ W + z
                mean_entropy = None

        return samples, mean_entropy

    d = B.shape[0]
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    scale = 1
    assert len(ordered_vertices) == d
    list_entropy = []
    for j in ordered_vertices:
        
        parents = G.neighbors(j, mode=ig.IN)
        X_parents = np.zeros((n,0))
        #! one hot encode the parents 
        if len(parents)>0:
            
            for feature in parents:
                if feature in list_categorical_features:
                    encoder = OneHotEncoder()
                    feature_transformed = encoder.fit_transform(X[:, feature].reshape(-1, 1)).toarray()
                else:
                    feature_transformed = X[:, feature].reshape(-1, 1)
                    
            
                X_parents = np.hstack((X_parents, feature_transformed))
                    
        
            
        n_classes = list_n_classes[j]
        jIsCat = j in list_categorical_features  
        sampled_feature, mean_entropy = _simulate_single_equation(X_parents, n_classes, jIsCat)
        X[:, j] = sampled_feature
        list_entropy.append((j,mean_entropy))
    return X, list_entropy
