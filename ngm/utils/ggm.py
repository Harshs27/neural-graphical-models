"""
Contains functions for using NGMs to model
Gaussian Grapical models. 
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import io, sys
from scipy.stats import multivariate_normal

# Local imports
import ngm.utils.data_processing as dp


def get_data(
    num_nodes,
    sparsity,
    num_samples,
    batch_size=1,
    typeG='CHAIN', 
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
        Xb (BxMxD): The sample data
        trueTheta (BxDxD): The true precision matrices
    """
    Xb, trueTheta = [], []
    for b in range(batch_size):
        # I - Getting the true edge connections
        edge_connections = generateGraph(
            num_nodes, 
            sparsity,
            typeG=typeG
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


def generateGraph(num_nodes, sparsity, typeG='RANDOM', seed=None):
    """Generate a random erdos-renyi graph with a given
    sparsity. 

    Args:
        num_nodes (int): The number of nodes in the graph
        sparsity ([float, float]): The [min, max] probability of edges
        seed (int, optional): set the numpy random seed
        typeG (str): RANDOM/GRID/CHAIN
    
    Returns:
        edge_connections (2D np array (float)): Adj matrix
    """
    if typeG == 'RANDOM':
        min_s, max_s = sparsity
        s =  np.random.uniform(min_s, max_s, 1)[0]
        G = nx.generators.random_graphs.gnp_random_graph(
            num_nodes, 
            s, 
            seed=seed, 
            directed=False
        )
    elif typeG == 'CHAIN':
        G = nx.generators.path_graph(num_nodes)
    else:
        print(f'Type of graph {typeG} not found.')
        sys.exit(0)
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
    edge_connections. Randomly assign +/-ve signs to entries.

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
    # Randomly assign +/-ve signs
    gs = nx.Graph()
    gs.add_weighted_edges_from(
        (u,v,np.random.choice([+1, -1], 1)[0]) 
        for u,v in nx.complete_graph(num_nodes).edges()
    )
    signs = nx.adjacency_matrix(gs).todense()
    theta = np.multiply(theta, signs) # update theta with the signs
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


def get_partial_correlations(precision):
    """Get the partial correlation matrix from the 
    precision matrix. It applies the following 
    
    Formula: rho_ij = -p_ij/sqrt(p_ii * p_jj)
    
    Args:
        precision (2D np.array): The precision matrix
    
    Returns:
        rho (2D np.array): The partial correlations
    """
    precision = np.array(precision)
    D = precision.shape[0]
    rho = np.zeros((D, D))
    for i in range(D): # rows
        for j in range(D): # columns
            if i==j: # diagonal elements
                rho[i][j] = 1
            elif j < i: # symmetric
                rho[i][j] = rho[j][i]
            else: # i > j
                num = -1*precision[i][j]
                den = np.sqrt(precision[i][i]*precision[j][j])
                rho[i][j] = num/den
    return rho


# Plot the graph
def graph_from_partial_correlations( 
    rho, 
    names, # node names
    sparsity=1,
    title='', 
    fig_size=12, 
    PLOT=True,
    save_file=None,
    roundOFF=5
    ):
    G = nx.Graph()
    G.add_nodes_from(names)
    D = rho.shape[-1]

    # determining the threshold to maintain the sparsity level of the graph
    def upper_tri_indexing(A):
        m = A.shape[0]
        r,c = np.triu_indices(m,1)
        return A[r,c]

    rho_upper = upper_tri_indexing(np.abs(rho))
    num_non_zeros = int(sparsity*len(rho_upper))
    rho_upper.sort()
    th = rho_upper[-num_non_zeros]
    print(f'Sparsity {sparsity} using threshold {th}')
    th_pos, th_neg = th, -1*th

    graph_edge_list = []
    for i in range(D):
        for j in range(i+1, D):
            if rho[i,j] > th_pos:
                G.add_edge(names[i], names[j], color='green', weight=round(rho[i,j], roundOFF), label='+')
                _edge = '('+names[i]+', '+names[j]+', '+str(round(rho[i,j], roundOFF))+', green)'
                graph_edge_list.append(_edge)
            elif rho[i,j] < th_neg:
                G.add_edge(names[i], names[j], color='red', weight=round(rho[i,j], roundOFF), label='-')
                _edge = '('+names[i]+', '+names[j]+', '+str(round(rho[i,j], roundOFF))+', red)'
                graph_edge_list.append(_edge)

    # if PLOT: print(f'graph edges {graph_edge_list, len(graph_edge_list)}')

    edge_colors = [G.edges[e]['color'] for e in G.edges]
    edge_width = np.array([abs(G.edges[e]['weight']) for e in G.edges])
    # Scaling the intensity of the edge_weights for viewing purposes
    if len(edge_width) > 0:
        edge_width = edge_width/np.max(np.abs(edge_width))
    image_bytes = None
    if PLOT:
        fig = plt.figure(1, figsize=(fig_size,fig_size))
        plt.title(title)
        n_edges = len(G.edges())
        pos = nx.spring_layout(G, scale=0.2, k=1/np.sqrt(n_edges+10))
        # pos = nx.nx_agraph.graphviz_layout(G, prog='fdp') #'fdp', 'sfdp', 'neato'
        nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=100)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width)
        y_off = 0.008
        nx.draw_networkx_labels(G, pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()})
        plt.title(f'{title}', fontsize=20)
        plt.margins(0.15)
        plt.tight_layout()
        # saving the file
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        # Saving the figure in-memory
        buf = io.BytesIO()
        plt.savefig(buf)
        # getting the image in bytes
        buf.seek(0)
        image_bytes = buf.getvalue() # Image.open(buf, mode='r')
        buf.close()
        # closing the plt
        plt.close(fig)
    return G, image_bytes, graph_edge_list


def viz_graph_from_precision(theta, column_names, sparsity=0.1, title=''):
    rho = get_partial_correlations(theta)
    Gr, _, _ = graph_from_partial_correlations(
        rho, 
        column_names,
        sparsity=sparsity
    )
    print(f'Num nodes: {len(Gr.nodes)}')
    Gv = dp.get_interactive_graph(Gr, title, node_PREFIX=None)
    return Gr, Gv


######################################################################
# Functions to analyse the marginal and conditional distributions
######################################################################

def get_distribution_function(target, source, model_GGM, Xi, count=100):
    """Plot the function target=GGM(source) or Xp=f(Xi).
    Vary the range of the source and collect the values of the 
    target variable. We keep the rest of the targets & sources
    constant given in Xi (input to the GGM). 

    Args:
        target (str/int/float): The feature of interest 
        source (str/int/float): The feature having a direct connection
            with the target in the neural view of NGM.
        model_GGM (list): [
            mean (pd.Series) = {feature: mean value}
            cov (2D np.array) = Covariance matrix between features
            scaler (list of pd.Series): [data_min_, data_max_]
        ]
        Xi (pd.DataFrame): Initial values of the input to the model.
            All the values except the source nodes remain constant
            while varying the input over the range of source feature.
        count (int): The number of points to evaluate f(x) in the range.

    Returns:
        x_vals (np.array): range of source values
        fx_vals (np.array): predicted f(source) values for the target
    """
    mean, cov, scaler = model_GGM
    data_min_, data_max_ = scaler
    column_names = Xi.columns
    print(f'target={target}, source={source}')
    # Get the min and max range of the source 
    source_idx = Xi.columns.get_loc(source)
    source_min = data_min_[source_idx]
    source_max = data_max_[source_idx]
    # Get the min and max range of the target 
    target_idx = Xi.columns.get_loc(target)
    target_min = data_min_[target_idx]
    target_max = data_max_[target_idx]
    # print(f'Source {source} at index {source_idx}: range ({source_min}, {source_max})')
    # Get the range of the source and target values
    x_vals = np.linspace(source_min, source_max, count)
    y_vals = np.linspace(target_min, target_max, count)
    # Collect the fx_vals
    fx_vals = []
    # For each x_val, find the expected value of y from the pdf
    for _x in x_vals:  # expected_value calculation
        # Set the source value
        Xi[source] = _x
        # Replicate the Xi entries to have count rows
        Xi_batch = pd.DataFrame(np.repeat(Xi.values, count, axis=0), columns=column_names)
        # Get the range of possible target values
        Xi_batch[target] = y_vals
        # Get the probabilitites using the probability density function
        py = multivariate_normal.pdf(Xi_batch, mean=mean, cov=cov)
        # Normalize the probabilities to make it proportional to conditional 
        # distribution p(target, source| X{remaining}) = p(S, T, {Xr})/p({Xr})
        py = py/np.sum(py)
        _y = np.dot(py, y_vals)  # Direct expectation calculation 
        # Choose the y based on sample count
        # _y = np.random.choice(y_vals, count, p=py) 
        fx_vals.append(_y)
    return x_vals, fx_vals
    

def analyse_feature(target_feature, model_GGM, G, Xi=[]):
    """Analyse the feature of interest with regards to the
    underlying multivariate Gaussian distribution defining 
    the conditional independence graph G.

    Args:
        target_feature (str/int/float): The feature of interest, should 
            be present as one of the nodes in graph G
        model_GGM (list): [
            mean (pd.Series) = {feature: mean value}
            cov (2D np.array) = Covariance matrix between features
            scaler (list of pd.Series): [data_min_, data_max_]
        ]
        G (nx.Graph): Conditional independence graph.
        Xi (pd.DataFrame): Initial input sample.
    
    Returns:
        None (Plots the dependency functions)
    """
    mean, cov, scaler = model_GGM
    model_features = mean.index
    # Preliminary check for the presence of target feature
    if target_feature not in model_features:
        print(f'Error: Input feature {target_feature} not in model features')
        sys.exit(0)
    # Drop the nodes not in the model from the graph
    common_features = set(G.nodes()).intersection(model_features)
    features_dropped = G.nodes() - common_features
    print(f'Features dropped from graph: {features_dropped}')
    G = G.subgraph(list(common_features))
    # 1. Get the neighbors (the dependent vars in CI graph) of the target  
    # feature from Graph G.
    target_nbrs = G[target_feature]
    # 2. Set the initial values of the nodes. 
    if len(Xi)==0: 
        Xi = mean
    Xi = dp.series2df(Xi)
    # Arrange the columns based on the model_feature names for compatibility
    Xi = Xi[model_features]
    # 3. Getting the plots by varying each nbr node and getting the regression 
    # values for the target node.
    plot_dict = {target_feature:{}}
    for nbr in target_nbrs.keys():
        x, fx = get_distribution_function(
            target_feature, 
            nbr, 
            model_GGM, 
            Xi
        )
        title = f'GGM: {target_feature} (y-axis) vs {nbr} (x-axis)'
        plot_dict[target_feature][nbr] = [x, fx, title]
    dp.function_plots_for_target(plot_dict)
    return None