import torch
from ngm.utils.uGLAD.glad.torch_sqrtm import MatrixSquareRoot

torch_sqrtm = MatrixSquareRoot.apply

def get_optimizers(model_glad, lr_glad=0.002, use_optimizer='adam'):
    if use_optimizer == 'adam':
        optimizer_glad = torch.optim.Adam(
            model_glad.parameters(),
            lr=lr_glad, 
            betas=(0.9, 0.999),
            eps=1e-08,
            # weight_decay=0
        )
    else:
        print('Optimizer not found!')
    return optimizer_glad


def batch_matrix_sqrt(A):
    # A should be PSD
    # if shape of A is 2D, i.e. a single matrix
    if len(A.shape)==2:
        return torch_sqrtm(A)
    else:
        n = A.shape[0]
        sqrtm_torch = torch.zeros(A.shape).type_as(A)
        for i in range(n):
            sqrtm_torch[i] = torch_sqrtm(A[i])
        return sqrtm_torch


def get_frobenius_norm(A, single=False):
    if single:
        return torch.sum(A**2)
    return torch.mean(torch.sum(A**2, (1,2)))


def glad(Sb, model, lambda_init=1, L=15, INIT_DIAG=0, USE_CUDA = False):
    """Unrolling the Alternating Minimization algorithm which takes in the 
    sample covariance (batch mode), runs the iterations of the AM updates and 
    returns the precision matrix. The hyperparameters are modeled as small 
    neural networks which are to be learned from the backprop signal of the 
    loss function. 

    Args:
        Sb (3D torch tensor (float)): Covariance (batch x dim x dim)
        model (class object): The GLAD neural network parameters 
                              (theta_init, rho, lambda)
        lambda_init (float): The initial lambda value
        L (int): The number of unrolled iterations
        INIT_DIAG (int): if 0 - Initial theta as (S + theta_init_offset * I)^-1
                         if 1 - Initial theta as (diag(S)+theta_init_offset*I)^-1
        USE_CUDA (bool): `True` if GPUs present else `False`
    
    Returns:
        theta_pred (3D torch tensor (float)): The output precision matrix
                                              (batch x dim x dim)
        loss (torch scalar (float)): The graphical lasso objective function
    """
    D = Sb.shape[-1]  # dimension of matrix
    # if batch is 1, then reshaping Sb
    if len(Sb.shape)==2:
        Sb = Sb.reshape(1, Sb.shape[0], Sb.shape[1])
    # Initializing the theta
    if INIT_DIAG == 1:
        #print('extract batchwise diagonals, add offset and take inverse')
        batch_diags = 1/(torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) 
                        + model.theta_init_offset)
        theta_init = torch.diag_embed(batch_diags)
    else:
        #print('(S+theta_offset*I)^-1 is used')
        theta_init = torch.inverse(Sb+model.theta_init_offset * 
                                    torch.eye(D).expand_as(Sb).type_as(Sb))

    theta_pred = theta_init#[ridx]
    identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
    # diagonal mask
#    mask = torch.eye(Sb.shape[-1], Sb.shape[-1]).byte()
#    dim = Sb.shape[-1]
#    mask1 = torch.ones(dim, dim) - torch.eye(dim, dim)
    if USE_CUDA == True:
        identity_mat = identity_mat.cuda()
#        mask = mask.cuda()
#        mask1 = mask1.cuda()

    zero = torch.Tensor([0])
    dtype = torch.FloatTensor
    if USE_CUDA == True:
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lambda_k = model.lambda_forward(zero + lambda_init, zero,  k=0)
    for k in range(L):
        # GLAD CELL
        b = 1.0/lambda_k * Sb - theta_pred
        b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0/lambda_k * identity_mat
        sqrt_term = batch_matrix_sqrt(b2_4ac)
        theta_k1 = 1.0/2*(-1*b+sqrt_term)

        theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred)
        # update the lambda
        lambda_k = model.lambda_forward(torch.Tensor(
                                [get_frobenius_norm(theta_pred-theta_k1)]
                                ).type(dtype), lambda_k, k)
    return theta_pred