import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from sklearn import covariance
from time import time
import torch

#################### Functions to generate data #####################
def get_data(
    num_nodes,
    sparsity,
    num_samples,
    batch_size=1,
    # typeG='RANDOM', 
    w_min=0.5, 
    w_max=1.0,
    eig_offset=0.1, 
    ):
    """Prepare true adj matrices as theta and then sample from 
    Gaussian to get the corresponding samples.
    
    Args:
        num_nodes (int): The number of nodes in graph
        sparsity ([float, float]): The [min, max] probability of edges
        num_samples (int): The number of samples to simulate
        batch_size (int, optional): The number of batches
        typeG (str): RANDOM/GRID/CHAIN
        w_min (float): Precision matrix entries ~Unif[w_min, w_max]
        w_max (float):  Precision matrix entries ~Unif[w_min, w_max]
    
    Returns:
        Xb (torch.Tensor BxMxD): The sample data
        trueTheta (torch.Tensor BxDxD): The true precision matrices
    """
    Xb, trueTheta = [], []
    for b in range(batch_size):
        # I - Getting the true edge connections
        edge_connections = generateRandomGraph(
            num_nodes, 
            sparsity,
            #typeG=typeG
            )
        # II - Gettings samples from fitting a Gaussian distribution
        # sample the entry of the matrix 
        
        X, true_theta = simulateGaussianSamples(
            num_nodes,
            edge_connections,
            num_samples, 
            u=eig_offset,
            w_min=w_min,
            w_max=w_max
            )
        # collect the batch data
        Xb.append(X)
        trueTheta.append(true_theta)
    return np.array(Xb), np.array(trueTheta)

def add_noise_dropout(Xb, dropout=0.25):
    """ Add the dropout noise to the input data.

    Args:
        Xb (torch.Tensor BxMxD): The sample data
        dropout (float): [0, 1) The percentage of 
            values to be replaced by NaNs 

    Returns:
        Xb_miss (torch.Tensor BxMxD): The sample with dropout
    """
    B, M, D = Xb.shape
    Xb_miss = []  # collect the noisy data
    for b in range(B):
        X = Xb[b].copy()  # M x D
        # Unroll X to 1D array: M*D
        X = X.reshape(-1)
        # Get the indices to mask/add noise
        mask_indices = np.random.choice(
            np.arange(X.size), 
            replace=False,
            size=int(X.size * dropout)
        )
        # Introduce missing values as NaNs
        X[mask_indices] = np.NaN
        # Reshape into the original dimensions
        X = X.reshape(M, D)
        Xb_miss.append(X)
    return np.array(Xb_miss)
######################################################################
#################### Functions to process data #####################

# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    """
    Calculate Cramers V statistic for categorial-categorial association.
    Similarly to correlation, the output is in the range of [0,1], 
    where 0 means no association and 1 is full association.
    
    Source: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Cramer's V is used with accounting for Bias correction.
    
    Note: chi-square = 0 implies that Cramér’s V = 0
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    num = phi2corr
    denom = min((kcorr-1),(rcorr-1))
    if denom==0:
        return 0 # No association
    return np.sqrt(num/denom)

def correlation_ratio(categories, measurements):
    """Finding correlation between categorical and numerical 
    features. 
    
    Source: https://en.wikipedia.org/wiki/Correlation_ratio
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def pairwise_cov_matrix(df, dtype):
    """Calculate the covariance matrix using pairwise calculations.
    Accounts for categorical, numerical & Real features. 

    `Cat-Cat' association is calculated using cramers V statistic.
    `Cat-Num' value is obtained using the correlation ratio.
    `Num-Num' correlation is calculated using the Pearson coefficient.
    
    Args:
        df (pd.DataFrame): The input data M(samples) x D(features)
        dtype (dict): {'column': 'r'/'c'}, where r=real, c=cat 

    Returns:
        cov (pd.DataFrame): Covariance matrix DxD 
    """
    features = df.columns
    D = len(features)
    cov = np.zeros((D, D))
    for i, fi in enumerate(features):
        print(f'row feature {i, fi}')
        for j, fj in enumerate(features):
            # print(f'col feature {j, fj}')
            if j>=i:
                if dtype[fi]=='c' and dtype[fj]=='c':
                    cov[i, j] = cramers_v(df[fi], df[fj])
                elif dtype[fi]=='c' and dtype[fj]=='r':
                    cov[i, j] = correlation_ratio(df[fi], df[fj])
                elif dtype[fi]=='r' and dtype[fj]=='c':
                    cov[i, j] = correlation_ratio(df[fj], df[fi])
                elif dtype[fi]=='r' and dtype[fj]=='r':
                    cov[i, j] = pearsonr(df[fi], df[fj])[0]
                cov[j, i] = cov[i, j]  # cov is symmetric
    # Convert to pd.Dataframe
    cov = pd.DataFrame(cov, index=features, columns=features)
    return cov


def convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(dtype)
    data.requires_grad = req_grad
    return data


def eigVal_conditionNum(A):
    """Calculates the eigenvalues and the condition 
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

def adjustCov(S, offset=0.1, min_eig=1e-6, max_con=1e5):
    # calculate the eigenvalue of the covariance S
    eig, con = eigVal_conditionNum(S)
    if min(eig)<=min_eig and con>max_con:
        # adjust the eigenvalue
        print(f'Adjust the eval: min {min(eig)}, con {con}')
        S += np.eye(S.shape[-1]) * (offset-min(eig))
        eig, con = eigVal_conditionNum(S)
        print(f'new eval: min {min(eig)}, con {con}')
    return S

def getCovariance(Xb, offset=0.1):
    """Calculate the batch covariance matrix 

    Args:
        Xb (3D np array): The input sample matrices (B x M x D)
        offset (float): The eigenvalue offset in case of bad 
                        condition number
    Returns:
        Sb (3D np array): Covariance matrices (B x D x D)
    """
    Sb = []
    for X in Xb:
        S = covariance.empirical_covariance(X, assume_centered=False)
        Sb.append(adjustCov(S, offset))
    return np.array(Sb)


def generateRandomGraph(num_nodes, sparsity, seed=None):
    """Generate a random erdos-renyi graph with a given
    sparsity. 

    Args:
        num_nodes (int): The number of nodes in the graph
        sparsity ([float, float]): The [min, max] probability of edges
        seed (int, optional): set the numpy random seed

    Returns:
        edge_connections (2D np array (float)): Adj matrix
    """
    # if seed: np.random.seed(seed)
    min_s, max_s = sparsity
    s =  np.random.uniform(min_s, max_s, 1)[0]
    G = nx.generators.random_graphs.gnp_random_graph(
        num_nodes, 
        s, 
        seed=seed, 
        directed=False
    )
    edge_connections = nx.adjacency_matrix(G).todense()
    return edge_connections


def simulateGaussianSamples(
    num_nodes,
    edge_connections, 
    num_samples, 
    seed=None, 
    u=0.1,
    w_min=0.5,
    w_max=1.0, 
    ): 
    """Simulating num_samples from a Gaussian distribution. The 
    precision matrix of the Gaussian is determined using the 
    edge_connections

    Args:
        num_nodes (int): The number of nodes in the DAG
        edge_connections (2D np array (float)): Adj matrix
        num_sample (int): The number of samples
        seed (int, optional): set the numpy random seed
        u (float): Min eigenvalue offset for the precision matrix
        w_min (float): Precision matrix entries ~Unif[w_min, w_max]
        w_max (float):  Precision matrix entries ~Unif[w_min, w_max]

    Returns:
        X (2D np array (float)): num_samples x num_nodes
        precision_mat (2D np array (float)): num_nodes x num_nodes
    """
    # zero mean of Gaussian distribution
    mean_value = 0 
    mean_normal = np.ones(num_nodes) * mean_value
    # Setting the random seed
    if seed: np.random.seed(seed)
    # uniform entry matrix [w_min, w_max]
    U = np.matrix(np.random.random((num_nodes, num_nodes))
                  * (w_max - w_min) + w_min)
    theta = np.multiply(edge_connections, U)
    # making it symmetric
    theta = (theta + theta.T)/2 + np.eye(num_nodes)
    smallest_eigval = np.min(np.linalg.eigvals(theta))
    # Just in case : to avoid numerical error in case an 
    # epsilon complex component present
    smallest_eigval = smallest_eigval.real
    # making the min eigenvalue as u
    precision_mat = theta + np.eye(num_nodes)*(u - smallest_eigval)
    # print(f'Smallest eval: {np.min(np.linalg.eigvals(precision_mat))}')
    # getting the covariance matrix (avoid the use of pinv) 
    cov = np.linalg.inv(precision_mat) 
    # get the samples 
    if seed: np.random.seed(seed)
    # Sampling data from multivariate normal distribution
    data = np.random.multivariate_normal(
        mean=mean_normal,
        cov=cov, 
        size=num_samples
        )
    return data, precision_mat  # MxD, DxD

############## Functions to check the input ########

# Processing the input data to be compatiable for the sparse graph recovery models
def process_table(
    table, 
    NORM='no', 
    MIN_VARIANCE=0.0, 
    msg='', 
    COND_NUM=np.inf, 
    eigval_th=1e-3,
    VERBOSE=True
    ):
    """Processing the input data to be compatiable for the 
    sparse graph recovery models. Checks for the following
    issues in the input tabular data (real values only).
    Note: The order is important. Repeat the function 
    twice: process_table(process_table(table)) to ensure
    the below conditions are satisfied.
    1. Remove all the rows with zero entries
    2. Fill Nans with column mean
    3. Remove columns containing only a single entry
    4. Remove columns with duplicate values
    5. Remove columns with low variance after centering
    The above steps are taken in order to ensure that the
    input matrix is well-conditioned. 
    Args:
        table (pd.DataFrame): The input table with headers
        NORM (str): min_max/mean/no
        MIN_VARIANCE (float): Drop the columns below this 
            variance threshold
        COND_NUM (int): The max condition number allowed
        eigval_th (float): Min eigval threshold. Making sure 
            that the min eigval is above this threshold by 
            droppping highly correlated columns
    Returns:
        table (pd.DataFrame): The processed table with headers
    """
    start = time()
    if VERBOSE:
        print(f'{msg}: Processing the input table for basic compatibility check')
        print(f'{msg}: The input table has sample {table.shape[0]} and features {table.shape[1]}')
    
    total_samples = table.shape[0]

    # typecast the table to floats
    table = table._convert(numeric=True)

    # 1. Removing all the rows with zero entries as the samples are missing
    table = table.loc[~(table==0).all(axis=1)]
    if VERBOSE: print(f'{msg}: Total zero samples dropped {total_samples - table.shape[0]}')

    # 2. Fill nan's with mean of columns
    table = table.fillna(table.mean())

    # 3. Remove columns containing only a single value
    single_value_columns = []
    for col in table.columns:
        if len(table[col].unique()) == 1:
            single_value_columns.append(col)
    table.drop(single_value_columns, inplace=True, axis=1)
    if VERBOSE: print(f'{msg}: Single value columns dropped: total {len(single_value_columns)}, columns {single_value_columns}')

    # Normalization of the input table
    table = normalize_table(table, NORM)

    # Analysing the input table's covariance matrix condition number
    analyse_condition_number(table, 'Input', VERBOSE)
 
    # 4. Remove columns with duplicate values
    all_columns = table.columns
    table = table.T.drop_duplicates().T  
    duplicate_columns = list(set(all_columns) - set(table.columns))
    if VERBOSE: print(f'{msg}: Duplicates dropped: total {len(duplicate_columns)}, columns {duplicate_columns}')

    # 5. Columns having similar variance have a slight chance that they might be almost duplicates 
    # which can affect the condition number of the covariance matrix. 
    # Also columns with low variance are less informative
    table_var = table.var().sort_values(ascending=True)
    # print(f'{msg}: Variance of the columns {table_var.to_string()}')
    # Dropping the columns with variance < MIN_VARIANCE
    low_variance_columns = list(table_var[table_var<MIN_VARIANCE].index)
    table.drop(low_variance_columns, inplace=True, axis=1)
    if VERBOSE: 
        print(f'{msg}: Low Variance columns dropped: min variance {MIN_VARIANCE},\
        total {len(low_variance_columns)}, columns {low_variance_columns}')

    # Analysing the processed table's covariance matrix condition number
    cov_table, eig, con = analyse_condition_number(table, 'Processed', VERBOSE)

    itr = 1
    while con > COND_NUM: # ill-conditioned matrix
        if VERBOSE: 
            print(f'{msg}: {itr} Condition number is high {con}. \
            Dropping the highly correlated features in the cov-table')
        # Find the number of eig vals < eigval_th for the cov_table matrix.
        # Rough indicator of the lower bound num of features that are highly correlated.
        eig = np.array(sorted(eig))
        lb_ill_cond_features = len(eig[eig<eigval_th])
        if VERBOSE: print(f'Current lower bound on ill-conditioned features {lb_ill_cond_features}')
        if lb_ill_cond_features == 0:
            if VERBOSE: print(f'All the eig vals are > {eigval_th} and current cond num {con}')
            if con > COND_NUM:
                lb_ill_cond_features = 1
            else:
                break
        highly_correlated_features = get_highly_correlated_features(cov_table)
        # Extracting the minimum num of features making the cov_table ill-conditioned
        highly_correlated_features = highly_correlated_features[
            :min(lb_ill_cond_features, len(highly_correlated_features))
        ]
        # The corresponding column names
        highly_correlated_columns = table.columns[highly_correlated_features]
        if VERBOSE: print(f'{msg} {itr}: Highly Correlated features dropped {highly_correlated_columns}, \
        {len(highly_correlated_columns)}')
        # Dropping the columns
        table.drop(highly_correlated_columns, inplace=True, axis=1)
        # Analysing the processed table's covariance matrix condition number
        cov_table, eig, con = analyse_condition_number(
            table, 
            f'{msg} {itr}: Corr features dropped',
            VERBOSE,
        )
        # Increasing the iteration number
        itr += 1
    if VERBOSE:
        print(f'{msg}: The processed table has sample {table.shape[0]} and features {table.shape[1]}')
        print(f'{msg}: Total time to process the table {np.round(time()-start, 3)} secs')
    return table


def get_highly_correlated_features(input_cov):
    """Taking the covariance of the input covariance matrix
    to find the highly correlated features that makes the 
    input cov matrix ill-conditioned.
    Args:
        input_cov (2D np.array): DxD matrix
    Returns:
        features_to_drop (np.array): List of indices to drop
    """
    cov2 = covariance.empirical_covariance(input_cov)
    # mask the diagonal 
    np.fill_diagonal(cov2, 0)
    # Get the threshold for top 10% 
    cov_upper = upper_tri_indexing(np.abs(cov2))
    sorted_cov_upper = [i for i in sorted(enumerate(cov_upper), key=lambda x:x[1], reverse=True)]
    th = sorted_cov_upper[int(0.1*len(sorted_cov_upper))][1]
    # Getting the feature correlation dictionary
    high_indices = np.transpose(np.nonzero(np.abs(cov2) >= th))
    high_indices_dict = {}
    for i in high_indices: # the upper triangular part
        if i[0] in high_indices_dict:
            high_indices_dict[i[0]].append(i[1])
        else:
            high_indices_dict[i[0]] = [i[1]]
    # sort the features based on the number of other correlated features.
    top_correlated_features = [[f, len(v)] for (f, v) in high_indices_dict.items()]
    top_correlated_features.sort(key=lambda x: x[1], reverse=True)
    top_correlated_features = np.array(top_correlated_features)
    features_to_drop = top_correlated_features[:, 0] 
    return features_to_drop


def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]


def analyse_condition_number(table, MESSAGE='', VERBOSE=True):
    S = covariance.empirical_covariance(table, assume_centered=False)
    eig, con = eig_val_condition_num(S)
    if VERBOSE: print(f'{MESSAGE} covariance matrix: The condition number {con} and min eig {min(eig)} max eig {max(eig)}')
    return S, eig, con
     

def eig_val_condition_num(A):
    """Calculates the eigenvalues and the condition
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


def normalize_table(df, typeN):
    if typeN == 'min_max':
        return (df-df.min())/(df.max()-df.min())
    elif typeN == 'mean':
        return (df-df.mean())/df.std()
    else:
        print(f'No Norm applied : Type entered {typeN}')
        return df