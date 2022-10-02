"""
The main file to train/test the uGLAD algorithm.
Contains code to generate data, run training and the 
loss function.
"""
import copy
from tabnanny import check
import numpy as np
import pandas as pd
from sklearn import covariance
from sklearn.model_selection import KFold
import sys
from time import time
import torch

# Helper functions for uGLAD
from ngm.utils.uGLAD.glad.glad_params import glad_params
from ngm.utils.uGLAD.glad import glad
from ngm.utils.uGLAD.utils.metrics import reportMetrics

import ngm.utils.uGLAD.utils.prepare_data as prepare_data

############### Wrapper class to match sklearn package #############
class uGLAD_GL(object):
    def __init__(self):
        """Wrapper class to match the sklearn GraphicalLassoCV
        output signature. Initializing the uGLAD model.
        """
        super(uGLAD_GL, self).__init__()
        self.covariance_ = None
        self.precision_ = None
        self.model_glad = None
        
    def fit(
        self, 
        X, 
        true_theta=None, 
        eval_offset=0.1,
        centered=False, 
        epochs=250, 
        lr=0.002,
        INIT_DIAG=0,
        L=15,
        verbose=True, 
        k_fold=3, 
        mode='direct', 
        cov=False
        ):
        """Takes in the samples X and returns
        a uGLAD model which stores the corresponding
        covariance and precision matrices.
        
        Args:
            X (2D np array): num_samples x dimension
            true_theta (2D np array): dim x dim of the 
                true precision matrix
            eval_offset (float): eigenval adjustment in
                case the cov is ill-conditioned
            centered (bool): Whether samples are mean 
                adjusted or not. True/False
            epochs (int): Training epochs
            lr (float): Learning rate of glad for the adam optimizer
            INIT_DIAG (int): 0/1 for initilization strategy of GLAD
            L (int): Num of unrolled iterations of GLAD
            verbose (bool): Print training output
            k_fold (int): num batches in missing mode
                num splits for k-fold in CV mode, 
                k=0 will run the direct mode
            mode (str): direct/cv/missing
            cov (bool): If True, X = covariance matrix (DxD)

        Returns:
            compare_theta (dict): Dictionary of comparison metrics
                between the predicted and true precision matrix
        """
        print(f'Running uGLAD')
        start = time()
        if not cov:
            print(f'Processing the input table for basic compatibility check')
            X = prepare_data.process_table(
                pd.DataFrame(X), 
                NORM='min_max', 
                VERBOSE=verbose
            )
        X = np.array(X)
        # Running the uGLAD model
        M, D = X.shape
        # Reshaping due to GLAD algorithm requirements
        Xb = X.reshape(1, M, D)
        true_theta_b = None
        if true_theta is not None:
            true_theta_b = true_theta.reshape(1, D, D)
        if mode=='missing':
            print(f'Handling missing data')
            pred_theta, compare_theta, model_glad = run_uGLAD_missing(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs, 
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose, 
                K_batch=k_fold 
                )
        elif mode=='cv' and k_fold>=0:
            print(f'CV mode: {k_fold}-fold')
            pred_theta, compare_theta, model_glad = run_uGLAD_CV(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs, 
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose, 
                k_fold=k_fold
                )
        elif mode=='direct':
            print(f'Direct Mode')
            pred_theta, compare_theta, model_glad = run_uGLAD_direct(
                Xb,
                trueTheta=true_theta_b,
                eval_offset=eval_offset,
                EPOCHS=epochs, 
                lr=lr,
                INIT_DIAG=INIT_DIAG,
                L=L,
                VERBOSE=verbose, 
                cov=cov
                )
        else:
            print(f'ERROR Please enter K-fold value in valid range [0, ), currently entered {k_fold}; Check mode {mode}')
            print(f'Note that cov={cov} only valid for mode=direct')
            sys.exit(0)
        # np.dot((X-mu)T, (X-mu)) / X.shape[0]
        self.covariance_ = covariance.empirical_covariance(
            X,
            assume_centered=centered
            )
        self.precision_ = pred_theta[0].detach().numpy()
        self.model_glad = model_glad
        print(f'Total runtime: {time()-start} secs\n')
        return compare_theta


class uGLAD_multitask(object):
    def __init__(self):
        """Initializing the uGLAD model in multi-task
        mode. It saves the covariance and predicted
        precision matrices for the input batch of data
        """
        super(uGLAD_multitask, self).__init__()
        self.covariance_ = []
        self.precision_ = []
        self.model_glad = None
        
    def fit(
        self, 
        Xb, 
        true_theta_b=None, 
        eval_offset=0.1,
        centered=False, 
        epochs=250, 
        lr=0.002,
        INIT_DIAG=0,
        L=15,
        verbose=True, 
        ):
        """Takes in the samples X and returns
        a uGLAD model which stores the corresponding
        covariance and precision matrices.
        
        Args:
            Xb (list of 2D np.array): batch * [num_samples' x dimension]
                NOTE: num_samples can be different for different data
            true_theta (3D np.array): batch x dim x dim of the 
                true precision matrix
            eval_offset (float): eigenval adjustment in
                case the cov is ill-conditioned
            centered (bool): Whether samples are mean 
                adjusted or not. True/False
            epochs (int): Training epochs
            lr (float): Learning rate of glad for the adam optimizer
            INIT_DIAG (int): 0/1 for initilization strategy of GLAD
            L (int): Num of unrolled iterations of GLAD
            verbose (bool): Print training output

        Returns:
            compare_theta (list[dict]): Dictionary of comparison metrics
                between the predicted and true precision matrices
        """
        print(f'Running uGLAD in multi-task mode')
        start = time()
        print(f'Processing the input table for basic compatibility check')
        processed_Xb = []
        for X in Xb:
            X = prepare_data.process_table(
                pd.DataFrame(X), 
                NORM='min_max', 
                VERBOSE=verbose
            )
            processed_Xb.append(np.array(X))
        Xb = processed_Xb
        # Running the uGLAD model
        pred_theta, compare_theta, model_glad = run_uGLAD_multitask(
            Xb,
            trueTheta=true_theta_b,
            eval_offset=eval_offset,
            EPOCHS=epochs, 
            lr=lr,
            INIT_DIAG=INIT_DIAG,
            L=L,
            VERBOSE=verbose, 
            )

        # np.dot((X-mu)T, (X-mu)) / X.shape[0]
        self.covariance_ = []
        for b in range(len(Xb)):
            self.covariance_.append(
                covariance.empirical_covariance(
                Xb[b],
                assume_centered=centered
                )
            )
        self.covariance_ = np.array(self.covariance_)
        self.precision_ = pred_theta.detach().numpy()
        self.model_glad = model_glad
        print(f'Total runtime: {time()-start} secs\n')
        return compare_theta
#####################################################################


#################### Functions to prepare model ######################
def init_uGLAD(lr, theta_init_offset=1.0, nF=3, H=3):
    """Initialize the GLAD model parameters and the optimizer
    to be used.

    Args:
        lr (float): Learning rate of glad for the adam optimizer
        theta_init_offset (float): Initialization diagonal offset 
            for the pred theta (adjust eigenvalue)
        nF (int): #input features for the entrywise thresholding
        H (int): The hidden layer size to be used for the NNs
    
    Returns:
        model: class object
        optimizer: class object
    """
    model = glad_params(
        theta_init_offset=theta_init_offset,
        nF=nF, 
        H=H
        )
    optimizer = glad.get_optimizers(model, lr_glad=lr)
    return model, optimizer


def forward_uGLAD(Sb, model_glad, L=15, INIT_DIAG=0, loss_Sb=None):
    """Run the input through the unsupervised GLAD algorithm.
    It executes the following steps in batch mode
    1. Run the GLAD model to get predicted precision matrix
    2. Calculate the glasso-loss
    
    Args:
        Sb (torch.Tensor BxDxD): The input covariance matrix 
        model_glad (dict): Contains the learnable params
        L (int): Num of unrolled iterations of GLAD
        INIT_DIAG (int): 0/1 for initilization strategy of GLAD
        loss_Sb (torch.Tensor BxDxD): The input covariance matrix 
            against which loss is calculated. If None, then use
            the input covariance matrix Sb
    
    Returns:
        predTheta (torch.Tensor BxDxD): The predicted theta
        loss (torch.scalar): The glasso loss 
    """
    # 1. Running the GLAD model 
    predTheta = glad.glad(Sb, model_glad, L=L, INIT_DIAG=INIT_DIAG)
    # 2. Calculate the glasso-loss
    if loss_Sb is None:
        loss = loss_uGLAD(predTheta, Sb)
    else:
        loss = loss_uGLAD(predTheta, loss_Sb)
    return predTheta, loss


def loss_uGLAD(theta, S):
    """The objective function of the graphical lasso which is 
    the loss function for the unsupervised learning of glad
    loss-glasso = 1/M(-log|theta| + <S, theta>)

    NOTE: We fix the batch size B=1 for `uGLAD`

    Args:
        theta (tensor 3D): precision matrix BxDxD
        S (tensor 3D): covariance matrix BxDxD (dim=D)
    
    Returns:
        loss (tensor 1D): the loss value of the obj function
    """
    B, D, _ = S.shape
    t1 = -1*torch.logdet(theta)
    # Batch Matrix multiplication: torch.bmm
    t21 = torch.einsum("bij, bjk -> bik", S, theta)
    # getting the trace (batch mode)
    t2 = torch.einsum('jii->j', t21)
    # print(t1, torch.det(theta), t2) 
    # regularization term 
    # tr = 1e-02 * torch.sum(torch.abs(theta))
    glasso_loss = torch.sum(t1+t2)/B # sum over the batch
    return glasso_loss 


def run_uGLAD_direct(
    Xb, 
    trueTheta=None, 
    eval_offset=0.1, 
    EPOCHS=250,
    lr=0.002,
    INIT_DIAG=0,
    L=15,
    VERBOSE=True,
    cov=False
    ):
    """Running the uGLAD algorithm in direct mode
    
    Args:
        Xb (np.array 1xMxD): The input sample matrix
        trueTheta (np.array 1xDxD): The corresponding 
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for 
            covariance matrix adjustment
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initilization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        EPOCHS (int): The number of training epochs
        VERBOSE (bool): if True, prints to sys.out
        cov (bool): if True, Xb= cov matrix (1xDxD)

    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if 
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Calculating the batch covariance
    if cov:
        Sb = []
        for X in Xb:
            Sb.append(prepare_data.adjustCov(
                X,
                offset=eval_offset,
                max_con=np.inf # Usually no need to adjust the 
                # covariance matrix if calculated entrywise
            ))
        Sb = np.array(Sb) 
    else:
        Sb = prepare_data.getCovariance(Xb, offset=eval_offset) # BxDxD
    # Converting the data to torch 
    Xb = prepare_data.convertToTorch(Xb, req_grad=False)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    if trueTheta is not None:
        trueTheta = prepare_data.convertToTorch(
            trueTheta,
            req_grad=False
            )
    B, _, _ = Xb.shape
    # NOTE: We fix the batch size B=1 for `uGLAD`
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(
        lr=lr,
        theta_init_offset=1.0,
        nF=3,
        H=3
        )
    PRINT_EVERY = int(EPOCHS/10)
    # print max 10 times per training
    # Optimizing for the glasso loss
    for e in range(EPOCHS):      
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss
        predTheta, loss = forward_uGLAD(
            Sb,
            model_glad,
            L=L,
            INIT_DIAG=INIT_DIAG
            )
        # calculate the backward gradients
        loss.backward()
        if not e%PRINT_EVERY and VERBOSE: print(f'epoch:{e}/{EPOCHS} loss:{loss.detach().numpy()}')
        # updating the optimizer params with the grads
        optimizer_glad.step()
    # reporting the metrics if true thetas provided
    compare_theta = None
    if trueTheta is not None:
        for b in range(B):
            compare_theta = reportMetrics(
                trueTheta[b].detach().numpy(), 
                predTheta[b].detach().numpy()
            )
            print(f'Compare - {compare_theta}')
    return predTheta, compare_theta, model_glad


def run_uGLAD_CV(
    Xb, 
    trueTheta=None, 
    eval_offset=0.1, 
    EPOCHS=250,
    lr=0.002,
    INIT_DIAG=0,
    L=15,
    VERBOSE=True, 
    k_fold=5
    ):
    """Running the uGLAD algorithm and select the best 
    model using 5-fold CV. 
    
    Args:
        Xb (np.array 1xMxD): The input sample matrix
        trueTheta (np.array 1xDxD): The corresponding 
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for 
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initilization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out
        k_fold (int): #splits for k-fold CV

    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if 
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Batch size is fixed to 1
    Sb = prepare_data.getCovariance(Xb, offset=eval_offset)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    # Splitting into k-fold for cross-validation 
    kf = KFold(n_splits=k_fold)
    # For each fold, collect the best model and the glasso-loss value
    results_Kfold = {}
    for _k, (train, test) in enumerate(kf.split(Xb[0])):
        if VERBOSE: print(f'Fold num {_k}')
        Xb_train = np.expand_dims(Xb[0][train], axis=0) # 1 x Mtrain x D
        Xb_test = np.expand_dims(Xb[0][test], axis=0) # 1 x Mtest x D
        # Calculating the batch covariance
        Sb_train = prepare_data.getCovariance(Xb_train, offset=eval_offset) # BxDxD
        Sb_test = prepare_data.getCovariance(Xb_test, offset=eval_offset) # BxDxD
        # Converting the data to torch 
        Sb_train = prepare_data.convertToTorch(Sb_train, req_grad=False)
        Sb_test = prepare_data.convertToTorch(Sb_test, req_grad=False)
        if trueTheta is not None:
            trueTheta = prepare_data.convertToTorch(
                trueTheta,
                req_grad=False
                )
        B, M, D = Xb_train.shape
        # NOTE: We fix the batch size B=1 for `uGLAD'
        # model and optimizer for uGLAD
        model_glad, optimizer_glad = init_uGLAD(
            lr=lr,
            theta_init_offset=1.0,
            nF=3,
            H=3
            )
        # Optimizing for the glasso loss
        best_test_loss = np.inf
        PRINT_EVERY = int(EPOCHS/10)
        # print max 10 times per training
        for e in range(EPOCHS):      
            # reset the grads to zero
            optimizer_glad.zero_grad()
            # calculate the loss for test and precision matrix for train
            predTheta, loss_train = forward_uGLAD(
                Sb_train, 
                model_glad,
                L=L,
                INIT_DIAG=INIT_DIAG
                )
            with torch.no_grad():
                _, loss_test = forward_uGLAD(
                    Sb_test,
                    model_glad,
                    L=L,
                    INIT_DIAG=INIT_DIAG
                    )
            # calculate the backward gradients
            loss_train.backward()
            # updating the optimizer params with the grads
            optimizer_glad.step()
            # Printing output
            _loss = loss_test.detach().numpy()
            if not e%PRINT_EVERY and VERBOSE: print(f'Fold {_k}: epoch:{e}/{EPOCHS} test-loss:{_loss}')
            # Updating the best model for this fold
            if _loss < best_test_loss: # and e%10==9:
                if VERBOSE and not e%PRINT_EVERY:
                    print(f'Fold {_k}: epoch:{e}/{EPOCHS}: Updating the best model with test-loss {_loss}')
                best_model_kfold = copy.deepcopy(model_glad)
                best_test_loss = _loss
        # updating with the best model and loss for the current fold
        results_Kfold[_k] = {}
        results_Kfold[_k]['test_loss'] = best_test_loss
        results_Kfold[_k]['model'] = best_model_kfold
        if VERBOSE: print('\n')

    # Strategy I: Select the best model from the results Kfold dictionary 
    # with the best score on the test fold.
    # print(f'Using Strategy I to select the best model')
    best_loss = np.inf
    for _k in results_Kfold.keys():
        curr_loss = results_Kfold[_k]['test_loss']
        if curr_loss < best_loss:
            model_glad = results_Kfold[_k]['model']
            best_loss = curr_loss

    # Run the best model on the complete data to retrieve the 
    # final predTheta (precision matrix)
    with torch.no_grad():
        predTheta, total_loss = forward_uGLAD(
            Sb,
            model_glad,
            L=L,
            INIT_DIAG=INIT_DIAG)
        
    # reporting the metrics if true theta is provided
    compare_theta = None
    if trueTheta is not None: 
        for b in range(B):
            compare_theta = reportMetrics(
                trueTheta[b].detach().numpy(), 
                predTheta[b].detach().numpy()
            )
        print(f'Comparison - {compare_theta}')
    return predTheta, compare_theta, model_glad


def run_uGLAD_missing(
    Xb, 
    trueTheta=None, 
    eval_offset=0.1, 
    EPOCHS=250,
    lr=0.002,
    INIT_DIAG=0,
    L=15,
    VERBOSE=True,
    K_batch=3
    ):
    """Running the uGLAD algorithm in missing data mode. We do a
    row-subsample of the input data and then train using multi-task
    learning approach to obtain the final precision matrix.
    
    Args:
        Xb (np.array 1xMxD): The input sample matrix with 
            missing entries as np.NaNs
        trueTheta (np.array 1xDxD): The corresponding 
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for 
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initilization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out
        K_batch (int): number of row-sumsampled batch for 
            multi-task learning (For less conflict in deciding the
            sign of final precision matrix, choose K as a odd value)

    Returns:
        predTheta (torch.Tensor 1xDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if 
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    # Batch size is fixed to 1
    if K_batch == 0: K_batch = 3  # setting the default
    # Step I:  Do statistical mean imputation
    Xb = mean_imputation(Xb)

    # Step II: Getting the batches and preparing data for uGLAD
    Sb = prepare_data.getCovariance(Xb, offset=eval_offset)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    # Splitting into k-fold for getting row-subsampled batches
    kf = KFold(n_splits=K_batch)
    print(f'Creating K={K_batch} row-subsampled batches')
    # Collect all the subsample in batch form: K x M' x D
    X_K = np.array([Xb[0][Idx] for Idx, _ in kf.split(Xb[0])])
    # Calculating the batch covariance
    S_K = prepare_data.getCovariance(X_K, offset=eval_offset) # BxDxD
    # Converting the data to torch 
    S_K = prepare_data.convertToTorch(S_K, req_grad=False)
    # Initialize the model and prepare theta if provided
    if trueTheta is not None:
        trueTheta = prepare_data.convertToTorch(
            trueTheta,
            req_grad=False
            )
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(
        lr=lr,
        theta_init_offset=1.0,
        nF=3,
        H=3
        )
    # STEP III: Optimizing for the glasso loss
    PRINT_EVERY = int(EPOCHS/10)
    # print max 10 times per training
    for e in range(EPOCHS):      
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss and precision matrix
        predTheta, loss = forward_uGLAD(
            S_K, 
            model_glad,
            L=L,
            INIT_DIAG=INIT_DIAG, 
            loss_Sb=Sb
            )
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer_glad.step()
        # Printing output
        _loss = loss.detach().numpy()
        if not e%PRINT_EVERY and VERBOSE: print(f'epoch:{e}/{EPOCHS} loss:{_loss}')

    # STEP IV: Getting the final precision matrix
    print(f'Getting the final precision matrix using the consensus strategy')
    predTheta = get_final_precision_from_batch(predTheta, type='min')
        
    # reporting the metrics if true theta is provided
    compare_theta = None
    if trueTheta is not None: 
        compare_theta = reportMetrics(
            trueTheta[0].detach().numpy(), 
            predTheta[0].detach().numpy()
        )
        print(f'Comparison - {compare_theta}')

    return predTheta, compare_theta, model_glad

def mean_imputation(Xb):
    """Replace nans of the input data by column
    means

    Args:
        Xb (torch.Tensor 1xMxD): The input sample matrix with 
            missing entries as np.NaNs
    
    Returns:
        Xb (torch.Tensor 1xMxD): Mean imputed matrix
    """
    Xb = Xb[0]
    # Mean of columns (ignoring NaNs) 
    col_mean = np.nanmean(Xb, axis=0)
    #Find indices that you need to replace
    inds = np.where(np.isnan(Xb))
    # Place column means in the indices. Align the arrays using take
    Xb[inds] = np.take(col_mean, inds[1])
    # Check if any column is full of NaNs, raise sys.exit()
    if np.isnan(np.sum(Xb)):
        print(f'ERROR: One or more columns have all NaNs')
        sys.exit(0)
    # Reshaping Xb with an extra dimension for compatability with glad
    Xb = np.expand_dims(Xb, axis=0)
    return Xb

def get_final_precision_from_batch(predTheta, type='min'):
    """The predTheta contains a batch of K precision 
    matrices. This function calculates the final 
    precision matrix by following the consensus 
    strategy

    \Theta^{f}_{i,j} = max-count(sign(\Theta^K_{i,j})) 
                        * min/mean{|\Theta^K_{i,j}|}
    (`min` is the recommended setting) 

    Args:
        predTheta (torch.Tensor KxDxD): Predicted graphs
            with batch_size = K
        type (str): min/mean to get the entry values

    Returns:
        predTheta (torch.Tensor 1xDxD): Final precision matrix
    """
    K, _, D = predTheta.shape
    # get the value term
    if type=='min':
        value_term = torch.min(torch.abs(predTheta), 0)[0]
    elif type=='mean':
        value_term = torch.mean(torch.abs(predTheta), 0)[0]
    else:
        print(f'Enter valid type min/mean, currently {type}')
        sys.exit(0)
    # get the sign term
    max_count_sign = torch.sum(torch.sign(predTheta), 0)
    # If sign is 0, then assign +1 
    max_count_sign[max_count_sign>=0] = 1
    max_count_sign[max_count_sign<0] = -1
    # Get the final precision matrix
    predTheta = max_count_sign * value_term
    return predTheta.reshape(1, D, D)


def run_uGLAD_multitask(
    Xb, 
    trueTheta=None, 
    eval_offset=0.1, 
    EPOCHS=250,
    lr=0.002,
    INIT_DIAG=0,
    L=15,
    VERBOSE=True,
    ):
    """Running the uGLAD algorithm in multitask mode. We
    train using multi-task learning approach to obtain 
    the final precision matrices for the batch of input data
    
    Args:
        Xb (list of 2D np.array):  The input sample matrix K * [M' x D]
            NOTE: num_samples can be different for different data
        trueTheta (np.array KxDxD): The corresponding 
            true graphs for reporting metrics or None
        eval_offset (float): eigenvalue offset for 
            covariance matrix adjustment
        EPOCHS (int): The number of training epochs
        lr (float): Learning rate of glad for the adam optimizer
        INIT_DIAG (int): 0/1 for initilization strategy of GLAD
        L (int): Num of unrolled iterations of GLAD
        VERBOSE (bool): if True, prints to sys.out

    Returns:
        predTheta (torch.Tensor BxDxD): Predicted graphs
        compare_theta (dict): returns comparison metrics if 
            true precision matrix is provided
        model_glad (class object): Returns the learned glad model
    """
    K = len(Xb)
    # Getting the batches and preparing data for uGLAD
    Sb = prepare_data.getCovariance(Xb, offset=eval_offset)
    Sb = prepare_data.convertToTorch(Sb, req_grad=False)
    # Initialize the model and prepare theta if provided
    if trueTheta is not None:
        trueTheta = prepare_data.convertToTorch(
            trueTheta,
            req_grad=False
            )
    # model and optimizer for uGLAD
    model_glad, optimizer_glad = init_uGLAD(
        lr=lr,
        theta_init_offset=1.0,
        nF=3,
        H=3
        )
    # Optimizing for the glasso loss
    PRINT_EVERY = int(EPOCHS/10)
    # print max 10 times per training
    for e in range(EPOCHS):      
        # reset the grads to zero
        optimizer_glad.zero_grad()
        # calculate the loss and precision matrix
        predTheta, loss = forward_uGLAD(
            Sb, 
            model_glad,
            L=L,
            INIT_DIAG=INIT_DIAG
            )
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer_glad.step()
        # Printing output
        _loss = loss.detach().numpy()
        if not e%PRINT_EVERY and VERBOSE: print(f'epoch:{e}/{EPOCHS} loss:{_loss}')
        
    # reporting the metrics if true theta is provided
    compare_theta = []
    if trueTheta is not None: 
        for b in range(K):
            rM = reportMetrics(
                    trueTheta[b].detach().numpy(), 
                    predTheta[b].detach().numpy()
                )
            print(f'Metrics for graph {b}: {rM}\n')
            compare_theta.append(rM)
    return predTheta, compare_theta, model_glad
######################################################################

# DO NOT USE 
def post_threshold(theta, s=80.0):
    """Apply post-hoc thresholding to zero out the 
    entries based on the input sparsity percentile.
    Usually we take conservative value of sparsity 
    percentage, so that we do not miss important 
    edges.

    Args:
        theta (2d np array): The DxD precision matrix
        s (float): Percentile sparsity desired
    
    Returns:
        theta (2d np array): The DxD precision matrix
    """
    # getting the threshold for s percentile
    cutoff = np.percentile(np.abs(theta), s)
    theta[np.abs(theta)<cutoff]=0
    return theta