from .locally_connected import LocallyConnected
import torch
import torch.nn as nn
import numpy as np
from  torch import optim
import copy
from tqdm.auto import tqdm
import wandb
import math 

#
class DagmaMLP(nn.Module): 
    def __init__(self, dims, bias=True, dtype=torch.double):
        torch.set_default_dtype(dtype)
        super(DagmaMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            #locally connected prend en compte le fait qu'on ait un output pour chaque d
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
            print("dims of locally connected", self.d, dims[l + 1], dims[l + 2])
        self.fc2 = nn.ModuleList(layers)


    #replace this part to get the right dimension at the end
    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s=1.0):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1.weight #[d, d*m1] m1 latent dimension . FOr each feature, [d*m1]
        fc1_weight = fc1_weight.view(self.d, -1, self.d) # [d, m1, d]  
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j] [d,d]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  
        A = torch.sum(fc1_weight ** 2, dim=1).t() 
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W



class DagmaMLPhet(nn.Module): 
    #! inherit from DagmaMLP
    def __init__(self, list_dims, indicator_continuous, list_columns, bias=True, dtype=torch.double):
        #list_columns is an array of size n_features, such that the ith element tells us which column corresponds to this feature on the first layer (necesary because of one-hot encoding)
        #indicator_continuous is an array of size n_features, such that its ith element indicates whether or not the feature is continuous (versus categorical).
        torch.set_default_dtype(dtype)
        super(DagmaMLPhet, self).__init__()
        assert len(list_dims[0]) >= 2

        #d is the one-hot encoded dimension
        self.list_dims = list_dims
        self.d_in =  list_dims[0][0]
        
        self.n_features = len(self.list_dims) 
        self.I = torch.eye(self.n_features)

        self.indicator_continuous = indicator_continuous #keep track of the categorical variables [n_features]
        self.list_columns = list_columns #keep track of the correspond columns given the features
        self.length_columns = [len(list_columns[i]) for i in range(self.n_features)] #keep track of the length of each column
        
        
        self.m1 = list_dims[0][1]
        self.fc1 = nn.Linear(self.d_in, self.n_features * self.m1, bias=bias)
        #initialize at zero to make sure we start with a DAG at the beginning of the training
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
           

        self.list_fc2 = []
        for i in range(self.n_features):
            layers = []
            for l in range(len(list_dims[i])-2):
                linear_layer = nn.Linear(list_dims[i][l+1], list_dims[i][l+2], bias = bias)
                
                k = 1.0 / self.m1
                bound = math.sqrt(k)
                nn.init.uniform_(linear_layer.weight, -bound, bound)
                if bias:
                    nn.init.uniform_(linear_layer.bias, -bound, bound)
                    
                layers.append(linear_layer)

                
                
                
            self.list_fc2.append(nn.ModuleList(layers))
            
        self.list_fc2 = nn.ModuleList(self.list_fc2)
        

    def forward(self, x):  
        #outputs is a list of the forward pass results for each dimension, hence dimensions are [(n,dim_1), (n,dim_2),..., (n,dim_d)]
        #where dim_1 is 1 for continuous variables, else it is the number of categories.
        outputs = [] 
        latent = self.fc1(x) # [n, d*m1] 
        latent = latent.view(-1, self.n_features, self.m1) # [n, d, m1]
        #swap the first two dimensions
        latent = latent.transpose(0,1) # [d, n, m1]
    
        for i in range(self.n_features): #go through each output feature
            out = latent[i]  # [n,m1]
            for fc in self.list_fc2[i]:
                out = torch.sigmoid(out)
                out = fc(out)
            
            outputs.append(out) #out: [n, 1]
        
        return outputs

    def h_func(self, s=1.0):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        
        stacked_weights = self.fc1.weight.view(self.n_features, self.m1, self.d_in)**2 #[d,m1,d]
        
        split_by_columns = torch.split(stacked_weights, self.length_columns, dim=2) #[[d,m1,G1], [d,m1, G2],...]
        
        reduced = [torch.sum(split_by_columns[j], dim=(1,2)) for j in range(len(self.list_columns))] #[[d], [d], [d], [d]]

        A = torch.stack(reduced, dim=0) #[d,d]
        h = -torch.slogdet(s * self.I - A)[1] + self.n_features * np.log(s)
        return h
        
        

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
       
        l1_reg = torch.norm(self.fc1.weight, p=1)
        return l1_reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        
        stacked_weights = self.fc1.weight.view(self.n_features, self.m1, self.d_in)**2 #[n_features,m1,d]
        split_by_columns = torch.split(stacked_weights, self.length_columns, dim=2) #[[n_features,m1,G1], [n_features,m1, G2],...]
        reduced = [torch.sum(split_by_columns[j], dim=(1,2)) for j in range(len(self.list_columns))] #[[n_features], [n_features], [n_features], [n_features]]
        A = torch.stack(reduced, dim=0) #[n_features,n_features]
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W




class DagmaNonlinear:
    def __init__(self, model: nn.Module, verbose=False, dtype=torch.double):
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype
    
    def log_mse_loss(self, output, target):
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
   
    def minimize(self, max_iter, lr, lambda1, lambda2, mu, s,
                 lr_decay=False, checkpoint=1000, tol=1e-6, pbar=None
        ):
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            
            #! Replace this part to get a list of outputs instead of a tensor
            X_hat = self.model(self.X)

            #Replace this part 
            score = self.log_mse_loss(X_hat, self.X)
            
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {score}')
                self.vprint(f'\tl1_reg(model): {l1_reg.item()}')
                self.vprint(f'\obj: {obj.item()}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
                
                
            pbar.update(1)
        return True

    def fit(self, X, lambda1=.02, lambda2=.005,
            T=4, mu_init=.1, mu_factor=.1, s=1.0,
            warm_iter=5e4, max_iter=8e4, lr=.0002, 
            w_threshold=0.3, checkpoint=1000
        ):
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")
        
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                        lr_decay, checkpoint=checkpoint, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est



class DagmaNonlinearHet:
    def __init__(self, model: nn.Module, verbose=False, dtype=torch.double, coef_continuous = 0.1, coef_categorical = 1., use_log_mse= False, use_gradnorm = False):
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype
        self.coef_continuous = coef_continuous
        self.coef_categorical = coef_categorical
        self.use_log_mse = use_log_mse
        weights = torch.tensor([coef_continuous, coef_categorical])
        self.weights = torch.nn.Parameter(weights)
        self.use_gradnorm = use_gradnorm
    
    
    def mse_loss(self, output, target):
        n = target.shape[0]
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
        
        individual_losses = torch.zeros(2)
        scoring_continuous = self.mse_loss
        
        sum_continuous = 0
        sum_categorical = 0
        
        
        for i in range(len(list_output)):
            output, target = list_output[i], X_preprocessed[:,i]
            if indicator_continuous[i]:
                loss = scoring_continuous(output, target)
                sum_continuous += loss
        
            else:
                #print("N values", self.model.length_columns[i])
                #print("output", output[:3])
                #print("target", target[:3])
                loss = self.compute_loss_categorical(output, target)
                sum_categorical += loss
        
        d= np.sum(indicator_continuous)
        if self.use_log_mse:
            sum_continuous = 0.5*d*torch.log(sum_continuous)
            
        individual_losses[0] = sum_continuous
        individual_losses[1] = sum_categorical

                
        total_score = self.weights[0]*sum_continuous + self.weights[1]*sum_categorical
       
            
        wandb.log({"mean_loss_categorical_dagma": sum_categorical/(len(indicator_continuous)-np.sum(indicator_continuous))})
        wandb.log({"mean_loss_continuous_dagma": sum_continuous})
        wandb.log({"weights continuous": self.weights[0]})
        wandb.log({"weights categorical": self.weights[1]})
        
        return total_score, individual_losses
    

    def minimize(self, max_iter, lr, lambda1, lambda2, mu, s,
                 lr_decay=False, checkpoint=1000, tol=1e-6, pbar=None, lr_weights = 0.001
        ):
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)
        
        if self.use_gradnorm:
            optimizer_weights = optim.Adam([self.weights], lr = 0.01)
            
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            list_output = self.model(self.X) #list_output is a list of predictions, one tensor for each input feature (there are d input features)

            score, individual_losses = self.heterogeneous_score(list_output, self.X_preprocessed)
            
            if self.use_gradnorm and i % checkpoint == 0 :
                l0 = individual_losses.detach() 

            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward(retain_graph = True)
            
            
            if self.use_gradnorm:
                alpha = 0.12
                T = 1
                
                gw = []
                for i in range(len(individual_losses)):
                    dl = torch.autograd.grad(self.weights[i]*individual_losses[i], self.model.fc1.parameters(), retain_graph=True, create_graph=True)[0]
                    gw.append(torch.norm(dl))
                gw = torch.stack(gw)
                # compute loss ratio per task
                loss_ratio = individual_losses.detach() / l0
                # compute the relative inverse training rate per task
                rt = loss_ratio / loss_ratio.mean()
                # compute the average gradient norm
                gw_avg = gw.mean().detach()
                # compute the GradNorm loss
                constant = (gw_avg * rt ** alpha).detach()
                gradnorm_loss = torch.abs(gw - constant).sum()
                # clear gradients of weights
                optimizer_weights.zero_grad()
                # backward pass for GradNorm
                gradnorm_loss.backward()
            
                # update loss weights
                optimizer_weights.step()
                # renormalize weights
                weights = (self.weights / self.weights.sum() * T).detach()
                self.weights = torch.nn.Parameter(weights)
                optimizer_weights = torch.optim.Adam([self.weights], lr=lr_weights)
                
            
            
            
            
            
            optimizer.step()
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {score.item()}')
                self.vprint(f'\tl1_reg(model): {l1_reg.item()}')
                self.vprint(f'\tobj(model): {obj_new}')
               
                wandb.log({"max coef first layer": torch.max(torch.abs(self.model.fc1.weight))})
                wandb.log({"l1 reg": l1_reg.item()})
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
        
            pbar.update(1)
        return True

    def fit(self, X, X_preprocessed, lambda1=.02, lambda2=.005,
            T=4, mu_init=.1, mu_factor=.1, s=1.0,
            warm_iter=5e4, max_iter=8e4, lr=.0002, 
            w_threshold=0.3, checkpoint=1000, skip_l1 = False
        ):
        torch.set_default_dtype(self.dtype)
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")
        
        self.X_preprocessed = X_preprocessed #array of size [n,n_features], for continuous variables this is the value, for categorical variables, this is the class index.
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                factor_l1 = 1
                while success is False:
                    
                    if i == 0 and skip_l1: 
                        factor_l1 = 0  
                    
                    success = self.minimize(inner_iter, lr, factor_l1*lambda1, lambda2, mu, s_cur, 
                                        lr_decay, checkpoint=checkpoint, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
                W_est = self.model.fc1_to_adj()
                
                wandb_image = wandb.Image(
                    (W_est*256).astype(int), 
                    caption=f"Estimated DAG with Dagma, iteration {i}"
                    )
                wandb.log({"W dagma": wandb_image})
                
        W_est = self.model.fc1_to_adj()
        print("Non thresholded", W_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
