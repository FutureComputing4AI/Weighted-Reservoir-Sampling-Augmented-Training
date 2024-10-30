import numpy as np
import pandas as pd
import sys, copy, os, pickle
import time
import scipy
from scipy import sparse
from numba import jit

# dataloader methods, metric extractors
from utils_v2 import load_data 
from tqdm.notebook import tqdm

# what are our possible settings for this script? start with the model + dataset
models = ["tgd"]
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# list of settings to build up
settings = []

# create a separate job for each combination of (model, dataset, seed)
for model in models: # 1x
    for dataset in ordered_datasets: # 16x
        for seed in range(5): # 5x
            settings.append((model, dataset, seed))
                
# command-line argument to get which setting we're running (80 settings total!)
model, dataset, seed = settings[int(sys.argv[1])]

# see pg. 790 of Langford 2009
K_tgd, g, theta, eta = 10, 0.15, np.inf, 0.1

# encode our instantaneous loss function directly too
def l(w, x, y):
    
    # make sure we have properly encoded binary classification.
    w = w.reshape(-1, 1)
    
    # we are ALWAYS assuming that x is SPARSE and a ROW vector!
    return np.maximum( 0.0, 1.0 - (y*((x @ w)) ))

# also get our hinge loss on A SET OF DATAPOINTS
def group_hinge(w, X, y):
    
    # make sure we have column wt_dense to stay safe
    w = w.reshape(-1, 1)
    
    # direct hinge loss, X is matrix, y is column, return mean hinge loss.
    return np.maximum(0.0, 1.0 - (y * ((X @ w)))).mean()
    

# helper function for computing accuracy quickly ON ONE DATAPOINT.
def indiv_acc(w, x_t, y_t):
    w = w.reshape(-1, 1)
    return (np.sign(((x_t @ w))) == y_t)[0,0]


# helper function for computing accuracy quickly assuming y (a GROUP OF OBSERVATONS) is already a column vector!
def group_acc(w, X, y):
    w = w.reshape(-1, 1)
    return ( np.sign( (X @ w) ) == y ).mean()


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

@jit
def T1(v, nonzeros, alpha, theta, last_update, current_time):

  for i in nonzeros:
    v_i = v[0,i]
    if v_i != 0: #acumulated update
      alpha_scaled = (current_time - last_update[i]) * alpha
    else: #simple update b/c you are getting knocked out of zero
      alpha_scaled = alpha 
    if 0 <= v_i and v_i <= theta:
      v[0,i] = np.maximum(0.0, v_i - alpha_scaled)
    elif -theta <= v_i and v_i <= 0:
      v[0,i] = np.minimum(0.0, v_i + alpha_scaled)
    # else:
    #   v[i] = v_i
    last_update[i] = current_time


# specify default maximum sizes of our train and test sets
N_train_max, N_test_max = np.inf, np.inf

# load in our dataset, with automatic shuffling
X, y = load_data(dataset=dataset, remove_zeroes=True, add_bias=False, seed=seed)
N, D = X.shape # no bias yet
# D += 1 # account for bias. FOR TRUNCATED GRADIENT, WE WILL NOT BE USING A BIAS TERM!

# get train + test set sizes + sanity-check for leakage: 70/30 split!
N_train = int(N * 0.7)
N_test = min(N - N_train, N_test_max)
assert N_train + N_test <= N, "Train-test leakage!"

# split data into train + test partitions: make sure y_train, y_test are COLUMNS!
X_train, y_train = X[:N_train], y[:N_train].reshape(-1, 1)
X_test, y_test = X[-N_test:], y[-N_test:].reshape(-1, 1)

# what's our WRS computation resolution? How often are we logging metrics?
resolution_step = int(N_train / 200)

# set a seed for reproducibility
np.random.seed(858)

# we'll use a fixed K + standard dense weights for our modified WRS-augmentation
K, weight_scheme = 64, "dense"

# what's the implied file name?
fheader = f"model={model}_K={K}_seed={seed}"

# we need to collect BOTH INSTANTANEOUS METRICS + WRS-AUGMENTED (K=64, dense, simple average) METRICS!
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"]

# which variables that we want to keep a record of?
inst_metrics = [f"inst_{metric}" for metric in metrics]
WRS_metrics = [f"WRS_{metric}" for metric in metrics]
metric_vars = ["timestep"] + inst_metrics + WRS_metrics + ["time_elapsed", "base_time_elapsed"]

# create a .csv that we want to load + repeatedly save to, and save our columns
with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "w") as file:
    file.write(','.join([str(entry) for entry in metric_vars]))
    file.write('\n')


# In[6]:


###########

'''
WRS-specific data structure: a reservoir, implemented as a dictionary
1. Index members of the reservoir using weight_id.
2. We only add to reservoir once a wt has run out of passive step.
*PS = # of subsequent passive steps
'''
reservoir = {} # members of form (w_t, no. of subseq. passive steps)

# to speed up, track k_i, w_i, PS, weight_id value, also weight_ids inside the reservoir
k_i_vals, w_i_vals, PS_i_vals, wid_i_vals = np.array([]), np.array([]), np.array([]), np.array([])

# store the weight vectors in the reservoir as a matrix.
W_matrix = np.array([])

# create a counter for the number of unique weights observed (0 vector corresponds to idx 0)
weight_id = 0

###########


# In[8]:


# initialize our wt
wt_dense = np.zeros(shape=(1, D))
lastUpdated = np.zeros(shape=(D))

# track our current candidate weights
candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1), "PS" : 0}

# iterate through our training datapoints
for i in range(N_train):
    
    # what is our g_i for this round? pg. 787 of Langford 2009
    g_i = K_tgd*g if i % K_tgd == 0 else 0.0
    
    # instantiate our row for insertion into metric_logs
    row = [i]

    ######## OBSERVE OUR DATA

    x_i = X_train[i,:]
    y_i = y_train[i]
    
    ######## COMPUTING INSTANTANEOUS + WRS METRICS USING PRE-UPDATE CANDIDATES!

    # instantaneous metrics only if recording timestep or final timestep
    if (len(w_i_vals) != 0) and (( (i % resolution_step == 0) or (i == N_train - 1))):
        
        # just take our instantaneous solution + update
        row += metric_pack(wt_dense.reshape(-1, 1), x_i, y_i, X_train, y_train, X_test, y_test)
        
        # simple average, no voting-based zeroing
        wt_WRS_SA = W_matrix.mean(axis=1).reshape(-1, 1)
        row += metric_pack(wt_WRS_SA, x_i, y_i, X_train, y_train, X_test, y_test)

    ######## ITERATIVE UPDATES: RESERVOIR FIRST (IF ERROR), AND THEN WEIGHT CANDIDATE!
    
    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss on current candidate -- we will consider adding to reservoir if make error.
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_i, y_i)[0,0]
    
    # see if we need to add to the reservoir (based on strict train accuracy)
    if np.sign(x_i.dot(wt_dense.T)) == y_i:
        
        #### RESERVOIR MANAGEMENT
        
        # compute our w_i and k_i. 1e-8 is buffer in case of w_i=0.
        u_i = np.random.uniform()
        w_i = compute_weights(weight_scheme, candidate_wt_dense)
        k_i = u_i ** (1 / (w_i + 1e-8) )
        PS_i = candidate_wt_dense["PS"]
        
        # encode the w_i and k_i for our candidate vector
        candidate_wt_dense["w_i"] = w_i
        candidate_wt_dense["k_i"] = k_i

        # with weights w_i and keys k_i computed, consider whether add to reservoir
        # ALWAYS add, of course, if reservoir is not full.
        if len(w_i_vals) < K:

            # deterministically add the weight-vectors to our reservoir!
            reservoir[weight_id] = copy.deepcopy(candidate_wt_dense)

            # update our {k_i, w_i, PS}_vals and W_matrix. SMALLEST k_i is always LEFT-MOST!
            if len(w_i_vals) == 0:
                k_i_vals = np.array([k_i])
                w_i_vals = np.array([w_i])
                PS_i_vals = np.array([PS_i])
                wid_i_vals = np.array([weight_id])
                W_matrix = np.array(candidate_wt_dense["wt"].reshape(-1,1))

            # SMALLEST k_i is always LEFT-MOST!
            elif k_i < k_i_vals[0]:
                k_i_vals = np.concatenate([[k_i], k_i_vals])
                w_i_vals = np.concatenate([[w_i], w_i_vals])
                PS_i_vals = np.concatenate([[PS_i], PS_i_vals])
                wid_i_vals = np.concatenate([[weight_id], wid_i_vals])
                W_matrix = np.concatenate([candidate_wt_dense["wt"], W_matrix], axis=1)

            # put our current k_i to the very end, should be fine. Just keep min. k_i in front!
            else:
                k_i_vals = np.concatenate([k_i_vals, [k_i]])
                w_i_vals = np.concatenate([w_i_vals, [w_i]])
                PS_i_vals = np.concatenate([PS_i_vals, [PS_i]])
                wid_i_vals = np.concatenate([wid_i_vals, [weight_id]])
                W_matrix = np.concatenate([W_matrix, candidate_wt_dense["wt"]], axis=1)

        # reservoir already full! need to do sampling.
        else:

            # find the smallest k_i, which we will set as our threshold
            min_dense_idx = np.argmin(k_i_vals)
            T = k_i_vals[min_dense_idx]

            # if current k_i > T, replace the members in reservoirs + speed-arrays
            if k_i > T:

                # delete the old reservoir member from the dictionary + add new one
                del reservoir[wid_i_vals[min_dense_idx]]
                reservoir[weight_id] = copy.deepcopy(candidate_wt_dense)

                # update our speed-arrays
                k_i_vals[min_dense_idx] = k_i
                w_i_vals[min_dense_idx] = w_i
                PS_i_vals[min_dense_idx] = PS_i
                wid_i_vals[min_dense_idx] = weight_id
                W_matrix[:,min_dense_idx] = candidate_wt_dense["wt"].flatten()
          
        # finally, reset counter
        candidate_wt_dense["PS"] = 0.0
        
        # also need to increment our "unique" encountered weight_id
        weight_id += 1
        
    # if we stayed passive, just increment the counters. candidate_wts lives another cycle.
    else:
        
        # increment counters on candidate_wt_dense's subseq. PS steps.
        candidate_wt_dense["PS"] += 1.0
    
    #### MAKE OUR WEIGHT-UPDATE FOR TRUNCATED GRADIENT DESCENT
    
    # tracking the base algorithm time
    base_start_time = time.time()
    
    # Langford 2009 update step, pg. 782
    subgrad_scale = y_i * x_i.dot(wt_dense.T)[0,0]
    
    if subgrad_scale > 1:
      subgrad_scale = (y_i * 0.0)
    else:
      subgrad_scale = -y_i
    #else, update w
    wt_dense -= (eta * subgrad_scale) * x_i #sparse update of w_i , parenthesis are important
    if g_i != 0:
      T1(wt_dense, x_i.nonzero()[-1], eta*g_i, theta, lastUpdated, i)
    
    # tracking the base algorithm time
    base_end_time = time.time()
    
    # update our candidate_wt_dense
    candidate_wt_dense["wt"] = wt_dense.copy().reshape(-1, 1)
    
    #### FINISH LOGGING + WRITE TO FILE.
    
    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    
    # only record the time elapsed if we're adding it to our dataframe
    if i != 0:
        row.append(end_time - start_time)
        row.append(base_end_time - base_start_time)

    # finally, add this row to our .csv
    if (len(w_i_vals) != 0) and (( (i % resolution_step == 0) or (i == N_train - 1))):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')

