from .locally_connected import LocallyConnected
from .lbfgsb_scipy import LBFGSBScipy
from .trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import wandb

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsMLPhet(nn.Module):
    def __init__(self, list_dims, indicator_continuous, list_columns, bias=True):
        super(NotearsMLPhet, self).__init__()
        
        self.list_dims = list_dims
        self.d_in =  list_dims[0][0]
        self.n_features = len(self.list_dims)
        
        
        self.indicator_continuous = indicator_continuous #keep track of the categorical variables [n_features]
        self.list_columns = list_columns #keep track of the correspond columns given the features
        self.length_columns = [len(list_columns[i]) for i in range(self.n_features)] #keep track of the length of each column
        
        
        self.m1 = list_dims[0][1]
        
         
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d_in, self.n_features * self.m1, bias=bias)
        self.fc1_neg = nn.Linear(self.d_in, self.n_features * self.m1, bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        
        
        self.list_fc2 = []
        for i in range(self.n_features):
            layers = []
            for l in range(len(list_dims[i])-2):
                layers.append(nn.Linear(list_dims[i][l+1], list_dims[i][l+2], bias = bias))
            self.list_fc2.append(nn.ModuleList(layers))
        self.list_fc2 = nn.ModuleList(self.list_fc2)

    def _bounds(self):
        
        bounds = []
        
       
        for j in range(self.d_in):
            for m in range(self.m1):
                for i in range(self.n_features):
                    
                    if j in self.list_columns[i]:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        outputs = [] 
        latent = self.fc1_pos(x) -self.fc1_neg(x) # [n, d*m1] 
        latent = latent.view(-1, self.n_features, self.m1) # [n, d, m1]
        #swap the first two dimensions
        latent = latent.transpose(0,1) # [d, n, m1]

        for i in range(self.n_features): #go through each output feature
            out = latent[i] #! call the module list directly?
            for fc in self.list_fc2[i]:
                out = torch.sigmoid(out)
                out = fc(out)
            outputs.append(out)
        
        return outputs

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        
        stacked_weights = fc1_weight.view(self.n_features, self.m1, self.d_in)**2 #[d,m1,d]
        split_by_columns = torch.split(stacked_weights, self.length_columns, dim=2) #[[d,m1,G1], [d,m1, G2],...]
        reduced = [torch.sum(split_by_columns[j], dim=(1,2)) for j in range(len(self.list_columns))] #[[d], [d], [d], [d]]
        A = torch.stack(reduced, dim=1) #[d,d]
        
        
        
        h = trace_expm(A) - self.n_features  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc2 in self.list_fc2:
            for fc in fc2:
                reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        
        stacked_weights = fc1_weight.view(self.n_features, self.m1, self.d_in)**2 #[d,m1,d]
        split_by_columns = torch.split(stacked_weights, self.length_columns, dim=2) #[[d,m1,G1], [d,m1, G2],...]
        reduced = [torch.sum(split_by_columns[j], dim=(1,2)) for j in range(len(self.list_columns))] #[[d], [d], [d], [d]]
        A = torch.stack(reduced, dim=1) #[d,d]
        
        
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W





class NotearsSobolev(nn.Module):
    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)  # ik -> j
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * torch.sin(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = torch.stack(seq, dim=2)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def forward(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(self.d) + A / self.d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, self.d - 1)
        # h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

class NotearsNonlinearHet:
    def __init__(self, model: nn.Module, verbose=False, dtype=torch.double, coef_continuous = 0.1, coef_categorical = 1., use_log_mse= False):
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype
        self.coef_continuous = coef_continuous
        self.coef_categorical = coef_categorical
        self.use_log_mse = use_log_mse
    
    def log_mse_loss(self, output, target):
        #!Use the normal MSE instead of wrapping it into a log
        n = len(target.shape)
        output = torch.squeeze(output, dim = -1)
        d = 1 #!because we pass the target for a single variable
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def mse_loss(self, output, target):
        n = len(target.shape)
        
        output = torch.squeeze(output, dim = -1)      
        
        loss = 1 / n * torch.sum((output - target) ** 2)
        return loss
    
    def compute_loss_categorical(self, output, target):
        #first convert the target to Long
        target = target.type(torch.LongTensor)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss
    
    def heterogeneous_score(self, list_output, X_preprocessed):
        total_score = 0
        indicator_continuous = self.model.indicator_continuous
        
        scores_continuous = []
        scores_categorical = []
        
        if self.use_log_mse:
            scoring_continuous = self.log_mse_loss
        else:
            scoring_continuous = self.mse_loss
        
        for i in range(len(list_output)):
            output, target = list_output[i], X_preprocessed[:,i]
            if indicator_continuous[i]:
                loss = scoring_continuous(output, target)
                total_score += self.coef_continuous*loss
                scores_continuous.append(loss)
                
                
            else:
                loss = self.compute_loss_categorical(output, target)
                total_score += self.coef_categorical*loss
                scores_categorical.append(loss)
       
            
        wandb.log({"mean_loss_categorical_notears": torch.mean(torch.tensor(scores_categorical))})
        wandb.log({"mean_loss_continuous_notears": torch.mean(torch.tensor(scores_continuous))})
    
        
        return total_score



    def dual_ascent_step(self, lambda1, lambda2, rho, alpha, h, rho_max):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        optimizer = LBFGSBScipy(self.model.parameters())
        while rho < rho_max:
            def closure():
                optimizer.zero_grad()
                list_output = self.model(self.X) #list_output is a list of predictions, one tensor for each input feature (there are d input features)
                score = self.heterogeneous_score(list_output, self.X_preprocessed)
                h_val = self.model.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * self.model.l2_reg()
                l1_reg = lambda1 * self.model.fc1_l1_reg()
                primal_obj = score + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj
            optimizer.step(closure)  # NOTE: updates model in-place
            with torch.no_grad():
                h_new = self.model.h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new


    def fit(self,
                        X: np.ndarray,
                        X_preprocessed : np.ndarray,
                        lambda1: float = 0.,
                        lambda2: float = 0.,
                        max_iter: int = 100,
                        h_tol: float = 1e-8,
                        rho_max: float = 1e+16,
                        w_threshold: float = 0.3):
        rho, alpha, h = 1.0, 0.0, np.inf
        
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")
        
        self.X_preprocessed = X_preprocessed
        
        for _ in range(max_iter):
            rho, alpha, h = self.dual_ascent_step( lambda1, lambda2,
                                            rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
        W_est = self.model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est


