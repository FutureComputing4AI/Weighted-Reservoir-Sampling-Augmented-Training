# only needs numpy + sparse matrices as import!
import numpy as np
from scipy.sparse import csr_matrix, load_npz


# encode our instantaneous loss function directly too
def l(w, x, y):
    
    # make sure we have properly encoded binary classification.
    w = w.reshape(-1, 1)
    
    # we are ALWAYS assuming that x is SPARSE and a ROW vector!
    return np.maximum( 0.0, 1.0 - (y*((x @ w[:-1,:]) + w[-1,0]) ))

# also get our hinge loss on A SET OF DATAPOINTS
def group_hinge(w, X, y):
    
    # make sure we have column wt_dense to stay safe
    w = w.reshape(-1, 1)
    
    # direct hinge loss, X is matrix, y is column, return mean hinge loss.
    return np.maximum(0.0, 1.0 - (y * ((X @ w[:-1,:]) + w[-1,0]))).mean()
    

# helper function for computing accuracy quickly ON ONE DATAPOINT.
def indiv_acc(w, x_t, y_t):
    w = w.reshape(-1, 1)
    return (np.sign(((x_t @ w[:-1,:]) + w[-1,0])) == y_t)[0,0]


# helper function for computing accuracy quickly assuming y (a GROUP OF OBSERVATONS) is already a column vector!
def group_acc(w, X, y):
    w = w.reshape(-1, 1)
    return ( np.sign( (X @ w[:-1,:]) + float(w[-1,0]) ) == y ).mean()


# helper function for computing sparsity concisely. 
def sparsity(w):
    w = w.reshape(-1, 1)
    return 1.0 - (w != 0.0).mean()


# helper function for getting combo of metrics at once
def metric_pack(w, x_t, y_t, X_train, y_train, X_test, y_test):
    
    w = w.reshape(-1, 1)
    
    '''
    1. Need instantaneous training accuracy + hinge loss on current observation, BUT ALSO
    2. Full accuracy + hinge loss ON ALL TRAINING DATA POINTS.
    3. Full accuracy + hinge loss ON ALL TESTING DATA POINTS.
    4. sparsity of the weight vector. No need for the L1 norm anymore.
    '''
    # compute our individual metrics
    return [indiv_acc(w, x_t, y_t), l(w, x_t, y_t)[0,0], 
            group_acc(w, X_train, y_train), group_hinge(w, X_train, y_train),
            group_acc(w, X_test, y_test), group_hinge(w, X_test, y_test), 
            sparsity(w)]


# function that extracts the NORMALIZED weights of each vector in reservoir, based on abbreviated w_i_vals
def extract_normalized_reservoir_weights(w_i_vals):
    
    # need to account for case where all weights are 0 ... just set equal weights.
    if w_i_vals.sum() == 0:
        n = len(w_i_vals)
        return np.full(shape=n, fill_value=1/n)
    
    # proceed with normalization, as usual.
    else:
        return w_i_vals / w_i_vals.sum()


# function that returns the unnormalized weight of a given candidate_wt_{dense, sparse} pair
def compute_weights(weight_scheme, candidate_wt_dense):
    
    # to support exp-dense and dense methods.
    if ("dense" in weight_scheme) and ("hybrid" not in weight_scheme):

        # directly take no. of passive steps and potentially exponentiate: DENSE + SPARSE use SAME WEIGHTS!
        w_i = np.exp(candidate_wt_dense["PS"]) if "exp" in weight_scheme else candidate_wt_dense["PS"]
        return w_i
    
    # error case
    else:
        raise Exception(f"{weight_scheme} is not supported.")