import torch
import torch.nn as nn

class glad_params(torch.nn.Module): 
    """The AM hyperparameters are parameterized in the glad_params.
    rho, lambda and theta_init_offset are learnable. 
    """
    def __init__(self, theta_init_offset, nF, H, USE_CUDA=False):
        """Initializing the GLAD model

        Args:
            theta_init_offset (float): The initial eigenvalue offset, set to a high value > 0.1
            nF (int): The number of input features for the entrywise thresholding
            H (int): The hidden layer size to be used for the NNs
            USE_CUDA (bool): Use GPU if True else CPU
        """
        super(glad_params, self).__init__() 
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        self.theta_init_offset = nn.Parameter(
            torch.Tensor(
                [theta_init_offset]
                ).type(self.dtype)
            )
        self.nF = nF # number of input features 
        self.H = H # hidden layer size
        self.rho_l1 = self.rhoNN()
        self.lambda_f = self.lambdaNN()
        self.zero = torch.Tensor([0]).type(self.dtype)

    def rhoNN(self):# per iteration NN
        l1 = nn.Linear(self.nF, self.H).type(self.dtype)
        lH1 = nn.Linear(self.H, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(),
                             lH1, nn.Tanh(),
                             l2, nn.Sigmoid()).type(self.dtype)

    def lambdaNN(self):
        l1 = nn.Linear(2, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(),
                             l2, nn.Sigmoid()).type(self.dtype)

    def eta_forward(self, X, S, k, F3=[]):
        batch_size, shape1, shape2 = X.shape
        Xr = X.reshape(batch_size, -1, 1)
        Sr = S.reshape(batch_size, -1, 1)
        feature_vector = torch.cat((Xr, Sr), -1)
        if len(F3)>0:
            F3r = F3.reshape(batch_size, -1, 1)
            feature_vector = torch.cat((feature_vector, F3r), -1)
        # elementwise thresholding
        rho_val = self.rho_l1(feature_vector).reshape(X.shape) 
        return torch.sign(X)*torch.max(self.zero, torch.abs(X)-rho_val)

    def lambda_forward(self, normF, prev_lambda, k=0):
        feature_vector = torch.Tensor([normF, prev_lambda]).type(self.dtype)
        return self.lambda_f(feature_vector)
        

