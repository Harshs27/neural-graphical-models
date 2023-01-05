"""
Additional data processing and post-processing
functions for neural graphical model analytics.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pyvis import network as net
from PIL import Image
import io
import pandas as pd
from sklearn import covariance
from sklearn import preprocessing
from time import time
import torch



def function_plots_for_target(plot_dict):
    """
    plot_dict ={
        target: {
            source1: [x, fx, title], 
            source2: [x, fx, title], 
            ...,
            }
    }
    """
    # Get the target
    target = list(plot_dict.keys())
    if len(target)==1:
        target = target[0]
    num_sources = len(plot_dict[target])
    # fig = plt.figure(figsize=(int(3*num_sources), 25))
    # fig = plt.figure(figsize=(5, int(5*num_sources)))
    fig = plt.figure(figsize=(15, 15))
    p=min(num_sources, 3)
    for i, source in enumerate(plot_dict[target].keys()):
        ax = plt.subplot(p+1, int(num_sources/p), i+1) # (grid_x, grid_y, plot_num)
        # plt.subplot(num_sources, 1, i+1)
        x, fx, title = plot_dict[target][source]
        # plot the function
        plt.plot(x, fx, 'b')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        plt.title(title, fontsize=20)
    # show the plot
    plt.savefig(f'plot_{target}.jpg', dpi=300)
    plt.show()


def plot_function(x, fx, title=f'plot of (x, fx)'):
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot the function
    plt.plot(x, fx, 'b')
    plt.title(title)
    # show the plot
    # plt.savefig('plot.jpg', dpi=300)
    # plt.show()
    return 


def retrieve_graph(graph_edges):
    """ Read the graph edgelist and 
    convert it to a networkx graph.
    """
    graph_edges = graph_edges.replace('(', '').replace(')', '')
    graph_edges = graph_edges[2:-1].split("', '")
    edge_list = []
    for e in graph_edges:
        e = e.split(',')
        edge_list.append(
            (e[0], ''.join(e[1:-2]).lstrip(), 
            {"weight":float(e[-2]), 'color':e[-1][1:]})
        )
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for n in G.nodes():
        G.nodes[n].update({'category':'unknown'})
    return G


def get_interactive_graph(G, title='', node_PREFIX='ObsVal'):
    Gv = net.Network(
        notebook=True, 
         height='750px', width='100%', 
    #     bgcolor='#222222', font_color='white',
        heading=title
    )
    Gv.from_nx(G.copy(), show_edge_weights=True, edge_weight_transf=(lambda x:x) )
    for e in Gv.edges:
        e['title'] = str(e['weight'])
        e['value'] = abs(e['weight'])
    if node_PREFIX is not None:
        for n in Gv.nodes:
            n['title'] = node_PREFIX+':'+n['category']
    Gv.show_buttons()
    return Gv


def set_feature_values(features_dict, features_known):
    """Updates the feature values with the known categories

    Args:
        features_dict (dict): {'name':'category'}
        node_attribute_konwn (dict): {'name':'category'}

    Returns:
        features_dict (dict): {'name':'category'}
    """
    for n, c in features_known.items():
        if n in features_dict.keys():
            features_dict[n] = c
        else:
            print(f'node {n} not found in features_dict')
    return features_dict


def series2df(series):
    "Convert a pd.Series to pd.Dataframe and set the index as header."
    # Convert the series to dictionary.
    series_dict = {n:v for n, v in zip(series.index, series.values)}
    # Create the dataframe from series and transpose.
    df = pd.DataFrame(series_dict.items()).transpose()
    # Set the index row as header and drop it from values.
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df


def t2np(x):
    "Convert torch to numpy"
    return x.detach().cpu().numpy()


def convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data.astype(np.float, copy=False)).type(dtype)
    data.requires_grad = req_grad
    return data


def normalize_table(X, method='min_max'):
    """Normalize the input data X.

    Args:
        X (pd.Dataframe): Samples(M) x Features(D).
        methods (str): min_max/mean 

    Returns:
        Xnorm (pd.Dataframe): Samples(M) x Features(D).
        scaler (object): The scaler to scale X
    """
    if method=='min_max':
        scaler = preprocessing.MinMaxScaler()
    elif method=='mean':
        scaler = preprocessing.StandardScaler()
    else:
        print(f'Scaler "{method}" not found')
    # Apply the scaler on the data X
    Xnorm = scaler.fit_transform(X)
    # Convert back to pandas dataframe
    Xnorm = pd.DataFrame(Xnorm, columns=X.columns)
    return Xnorm, scaler


def inverse_norm_table(Xnorm, Xscaler):
    """
    Apply the inverse transform on input normalized
    data to get back the original data.
    """
    return Xscaler.inverse_transform(Xnorm)

def analyse_condition_number(table, MESSAGE=''):
    S = covariance.empirical_covariance(table, assume_centered=False)
    eig, con = eig_val_condition_num(S)
    print(f'{MESSAGE} covariance matrix: The condition number {con} and min eig {min(eig)} max eig {max(eig)}')
    return S, eig, con
     

def eig_val_condition_num(A):
    """Calculates the eigenvalues and the condition 
    number of the input matrix A

    condition number = max(|eig|)/min(|eig|)
    """
    eig = [v.real for v in np.linalg.eigvals(A)]
    condition_number = max(np.abs(eig)) / min(np.abs(eig))
    return eig, condition_number


# Processing the input data to be compatiable for the CI graph recovery models
def process_data_for_CI_graph(table, NORM='min_max', msg='', drop_duplicate=True):
    """Processing the input data to be compatiable for the 
    regression network model. Checks for the following
    issues in the input tabular data (real values only).
    
    1. Remove all the rows with zero entries
    2. Fill Nans with column mean
    3. Remove columns containing only a single entry
    4. Remove columns with duplicate values
    
    Args:
        X (pd.DataFrame): The input table with headers
        NORM (str): min_max/mean

    Returns:
        table (pd.DataFrame): The processed table with headers
    """
    start = time()
    print(f'{msg}: Processing the input table for basic compatibility check')
    print(f'{msg}: The input table has sample {table.shape[0]} and features {table.shape[1]}')
    
    total_samples = table.shape[0]

    # typecast the table to floats
    table = table._convert(numeric=True)

    # 1. Removing all the rows with zero entries as the samples are missing
    table = table.loc[~(table==0).all(axis=1)]
    print(f'{msg}: Total zero samples dropped {total_samples - table.shape[0]}')

    # 2. Fill nan's with mean of columns
    table = table.fillna(table.mean())

    # 3. Remove columns containing only a single value
    single_value_columns = []
    for col in table.columns:
        if len(table[col].unique()) == 1:
            single_value_columns.append(col)
    table.drop(single_value_columns, inplace=True, axis=1)
    print(f'{msg}: Single value columns dropped: total {len(single_value_columns)}, columns {single_value_columns}')

    # Normalization of the input table
    table, scaler = normalize_table(table, NORM)

    if drop_duplicate:
        # 4. Remove columns with duplicate values
        all_columns = table.columns
        table = table.T.drop_duplicates().T  
        duplicate_columns = list(set(all_columns) - set(table.columns))
        print(f'{msg}: Duplicates dropped: total {len(duplicate_columns)}, columns {duplicate_columns}')

    # # Analysing the processed table's covariance matrix condition number
    # cov_table, eig, con = analyse_condition_number(table, 'Processed')

    print(f'{msg}: The processed table has sample {table.shape[0]} and features {table.shape[1]}')
    print(f'{msg}: Total time to process the table {np.round(time()-start, 3)} secs')
    return table, scaler


def get_cat_names(ohe, dtype):
    # Collecting the number of categories in cat features
    # categorical features in the original df. 
    categorical_features = [k for k, v in dtype.items() if v=='c']
    cat_names = {}
    for name, cat in zip(categorical_features, ohe.categories_):
        cat_names[name] = [str(name)+'_'+str(c) for c in cat]
    return cat_names

def convert_to_onehot(df, prefix=None):
    ohe = preprocessing.OneHotEncoder()#(handle_unknown='ignore')
    ohe.fit(df)
    # transforming the entire array
    df_ohe = ohe.transform(df).toarray()
    # transforming a single input
    # single_ohe = ohe.transform([df.loc[0].values]).toarrayray()
    # setting the column names
    col_names = ohe.get_feature_names_out()
    df_ohe = pd.DataFrame(df_ohe, columns=col_names)

    return df_ohe, ohe


# Graph processing tools

def plot_graph_compare(G, pos=None, title='', scale_wt=1, intensity=1):
    edge_colors = [G.edges[e]['color'] for e in G.edges]
    edge_width = [intensity*abs(float(G.edges[e]['weight']))/scale_wt for e in G.edges]    
    plt.title(title, fontsize=20)
    n_edges = len(G.edges)
    if pos is None:
        pos = nx.spring_layout(G, scale=0.2, k=1/np.sqrt(n_edges+10))
        # pos = nx.nx_agraph.graphviz_layout(G, prog='fdp') #'fdp', 'sfdp', 'neato'
    nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=100)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_width)
    y_off = 0.008
    nx.draw_networkx_labels(G, pos = {k:([v[0], v[1]+y_off]) for k,v in pos.items()})
    return 

def compare_graphs(G1, G2, t1='Title1', t2='Title2', graph_type={'G1':'undirected', 'G2':'undirected'}):
    """
    1. Finding the common edges in the graphs (edges present in both G1 and G2)
    2. Finding the unique edges in graphs (edges in G1 and not in G2 and vice-versa)
    """
    # Find the common nodes between the graphs
    common_nodes = set(G1.nodes()).intersection(G2.nodes())
    print(f'DA: Common nodes: {len(common_nodes)}, Nodes in G1: {len(G1.nodes())}, Nodes in G2: {len(G2.nodes())}')
    # Reduce the graphs to just the common nodes
    G1_int = G1.subgraph(common_nodes)
    G2_int = G2.subgraph(common_nodes)

    # NOTE: There is some logic problem with the networkx 2.8.6 implementation
    # G_int = nx.intersection(G1_int, G2_int)

    # print(f'Nx function: G_int {G_int.edges(data=False)}')

    def get_graph_intersection(G1, G2):
        G1_v = G1.to_undirected(G1)
        G2_v = G2.to_undirected(G2)
        common_edges = []
        for edge in G1_v.edges(data=False):
            if G2_v.has_edge(*edge):
                common_edges.append(edge)
        G_int = nx.Graph()
        G_int.add_edges_from(common_edges)
        return G_int

    G_int = get_graph_intersection(G1_int, G2_int)


    # print(f'G1_int {G1_int.edges(data=False)}')
    # print(f'G2_int {G2_int.edges(data=False)}')
    # print(f'My function: G_int {G_int.edges(data=False)}')
    # print(f'CHEKCE: {G_int.has_edge("cause_of_death","manner"), G1_int.has_edge("cause_of_death", "manner"), G2_int.has_edge("manner","cause_of_death")}')
    
    # remove isolated nodes with no edge connections
    isolated_nodes = list(nx.isolates(G_int))
    print(f'isolated nodes {isolated_nodes}')
    G_int.remove_nodes_from(isolated_nodes)
    # unfreeze the graphs
    G1_int = nx.Graph(G1_int) if graph_type['G1']=='undirected' else nx.DiGraph(G1_int)
    G2_int = nx.Graph(G2_int) if graph_type['G2']=='undirected' else nx.DiGraph(G2_int)
    # Remove isolated nodes from the G1 and G2 subgraphs
    G1_int.remove_nodes_from(isolated_nodes)
    G2_int.remove_nodes_from(isolated_nodes)
    # freeze the graphs
    G1_int = nx.freeze(G1_int)
    G2_int = nx.freeze(G2_int)

    # Find the common set of edges 
    common_edges = G_int.edges()
    print(f'common_edges{common_edges}')
    # 1. Updating the G1 and G2 graphs with only the common edges
    # 2. Unique edges present in G1
    if graph_type['G1']=='undirected':
        G1_int = nx.Graph(((u, v, e) for u,v,e in G1_int.edges(data=True) if G_int.has_edge(u, v)))
        G1_unique = nx.Graph(((u, v, e) for u,v,e in G1.edges(data=True) if not G_int.has_edge(u, v)))
        #  G1_int = nx.Graph(((u, v, e) for u,v,e in G1_int.edges(data=True) if (u, v) in G_int.edges))
        # G1_unique = nx.Graph(((u, v, e) for u,v,e in G1.edges(data=True) if (u, v) not in G_int.edges))
    else:
        G1_int = nx.DiGraph(((u, v, e) for u,v,e in G1_int.edges(data=True) if G_int.has_edge(u, v)))
        G1_unique = nx.DiGraph(((u, v, e) for u,v,e in G1.edges(data=True) if not G_int.has_edge(u, v)))

    if graph_type['G2']=='undirected':
        G2_int = nx.Graph(((u, v, e) for u,v,e in G2_int.edges(data=True) if G_int.has_edge(u, v)))
        G2_unique = nx.Graph(((u, v, e) for u,v,e in G2.edges(data=True) if not G_int.has_edge(u, v)))
    else:
        G2_int = nx.DiGraph(((u, v, e) for u,v,e in G2_int.edges(data=True) if G_int.has_edge(u, v)))
        G2_unique = nx.DiGraph(((u, v, e) for u,v,e in G2.edges(data=True) if not G_int.has_edge(u, v)))

    # if pos is None:
    pos = nx.spring_layout(G_int, scale=40, k=3/np.sqrt(G_int.order()))
    # pos = nx.nx_agraph.graphviz_layout(G_int, prog='neato') #'fdp', 'sfdp', 'neato'
    # nx.draw(G1_int, pos=pos, with_labels=True)
    # fig = plt.figure(figsize=(fig_size, fig_size))


    def get_scaling_wt(G):
        edge_width_G = np.array([abs(G.edges[e]['weight']) for e in G.edges])
        # Scaling the intensity of the edge_weights for viewing purposes
        scale_wt_G = np.max(np.abs(edge_width_G)) if len(edge_width_G) > 0 else 1
        return scale_wt_G

    scale_wt_G1 = get_scaling_wt(G1)
    scale_wt_G2 = get_scaling_wt(G2)

    plt.figure(figsize=(24, 24)) 
    plt.subplot(221)
    # plt.figure(1, figsize=(fig_size, fig_size))
    plot_graph_compare(G1_int, pos, title=t1+': Edges present in both graphs', scale_wt=scale_wt_G1, intensity=3)
    plt.subplot(222)#, figsize=(fig_size, fig_size))
    plot_graph_compare(G2_int, pos, title=t2+': Edges present in both graphs', scale_wt=scale_wt_G2)
    plt.subplot(223)#, figsize=(fig_size, fig_size))
    plot_graph_compare(G1_unique, title=t1+': Unique edges', scale_wt=scale_wt_G1, intensity=3)
    plt.subplot(224)#, figsize=(fig_size, fig_size))
    # G2_unique.remove_nodes_from(['no_mmorb', 'attend'])
    plot_graph_compare(G2_unique, title=t2+': Unique edges', scale_wt=scale_wt_G2)#, get_image_bytes=True)

    plt.savefig('compare_graphs', bbox_inches='tight')
    # Saving the figure in-memory
    buf = io.BytesIO()
    plt.savefig(buf)
    # getting the image in bytes
    buf.seek(0)
    image_bytes = buf.getvalue() # Image.open(buf, mode='r')
    buf.close()
    # closing the plt
    plt.close()
    return image_bytes
