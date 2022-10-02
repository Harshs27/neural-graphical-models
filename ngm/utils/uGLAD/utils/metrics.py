import numpy as np
from sklearn import metrics
from pprint import pprint

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr

def reportMetrics(trueG, G, beta=1):
    """Compute various metrics
    Args:
        trueG (2D numpy arr[floats]): ground truth precision matrix
        G (2D numpy arr[floats]): predicted precsion mat
        beta (int, optional): beta for the Fbeta score

    Returns:
        Dict: {fdr (float): (false positive) / prediction positive = FP/P
                tpr (float): (true positive) / condition positive = TP/T
                fpr (float): (false positive) / condition negative = FP/F
                shd (int): undirected extra + undirected missing = E+M
                nnz (int): number of non-zeros for trueG and predG
                ps (float): probability of success, sign match
                Fbeta (float): F-score with beta
                aupr (float): area under the precision-recall curve
                auc (float): area under the ROC curve}
    """
    trueG = trueG.real
    G =G.real
    # trueG and G are numpy arrays
    # convert all non-zeros in G to 1
    d = G.shape[-1]

    # changing to 1/0 for TP and FP calculations
    G_binary = np.where(G!=0, 1, 0)
    trueG_binary = np.where(trueG!=0, 1, 0)
    # extract the upper diagonal matrix
    indices_triu = np.triu_indices(d, 1)
    trueEdges = trueG_binary[indices_triu] #np.triu(G_true_binary, 1)
    predEdges = G_binary[indices_triu] #np.triu(G_binary, 1)
    # Getting AUROC value
    predEdges_auc = G[indices_triu] #np.triu(G_true_binary, 1)
    auc, aupr = get_auc(trueEdges, np.absolute(predEdges_auc))
    # Now, we have the edge array for comparison
    # true pos = pred is 1 and true is 1
    TP = np.sum(trueEdges * predEdges) # true_pos
    # False pos = pred is 1 and true is 0
    mismatches = np.logical_xor(trueEdges, predEdges)
    FP = np.sum(mismatches * predEdges)
    # Find all mismatches with Xor and then just select the ones with pred as 1 
    # P = Number of pred edges : nnzPred 
    P = np.sum(predEdges)
    nnzPred = P
    # T = Number of True edges :  nnzTrue
    T = np.sum(trueEdges)
    nnzTrue = T
    # F = Number of non-edges in true graph
    F = len(trueEdges) - T
    # SHD = total number of mismatches
    SHD = np.sum(mismatches)
    # FDR = False discovery rate
    FDR = FP/P
    # TPR = True positive rate
    TPR = TP/T
    # FPR = False positive rate
    FPR = FP/F
    # False negative = pred is 0 and true is 1
    FN = np.sum(mismatches * trueEdges)
    # F beta score
    num = (1+beta**2)*TP
    den = ((1+beta**2)*TP + beta**2 * FN + FP)
    Fbeta = num/den
    # precision 
    precision = TP/(TP+FP)
    # recall 
    recall = TP/(TP+FN)
    return {'FDR': FDR, 'TPR': TPR, 'FPR': FPR, 'SHD': SHD, 'nnzTrue': nnzTrue, 
            'nnzPred': nnzPred, 'precision': precision, 'recall': recall, 
            'Fbeta': Fbeta, 'aupr': aupr, 'auc': auc}

def summarize_compare_theta(compare_dict_list, method_name='Method Name'):
    avg_results = {}
    for key in compare_dict_list[0].keys():
        avg_results[key] = []
    
    total_runs = len(compare_dict_list)
    for cd in compare_dict_list:
        for key in cd.keys():
            avg_results[key].append(cd[key])
    # getting the mean and std dev
    for key in avg_results.keys():
        avk = avg_results[key]
        avg_results[key] = (np.mean(avk), np.std(avk))
    print(f'Avg results for {method_name}\n')
    pprint(avg_results)
    print(f'\nTotal runs {total_runs}\n\n')
    return avg_results