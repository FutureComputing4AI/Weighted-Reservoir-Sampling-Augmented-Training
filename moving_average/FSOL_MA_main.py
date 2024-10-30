import numpy as np
import pandas as pd
import sys, copy, os, pickle
import time
import scipy
from scipy import sparse

# dataloaders + update methods, metric extractors
from utils_v2 import *
from metric_extractors import * 

# what are our possible settings for this script? start with the model + datasets
models = ["FSOL"]
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# how many most recent vectors are we taking a moving average of?
Ks = [64]

# list of settings to build up
settings = []

# create a separate job for each combination of model, dataset, weight_scheme, K value
for model in models: # 1x
    for dataset in ordered_datasets: # 16x
        for K in Ks:
            for seed in range(5): # 5x
                settings.append((model, dataset, K, seed))
                
# command-line argument to get which setting we're running (80 settings total!)
model, dataset, K, seed = settings[int(sys.argv[1])]

# immediately load in the best hyperparameters for this dataset + model
log2eta, log10lmbda = pd.read_csv("base_variants/FSOL_hparams.csv")\
.query(f"dataset == '{dataset}'")[["log2eta", "log10lmbda"]].values[0]

# what's the implied file name?
fheader = f"model={model}_K={K}_seed={seed}"

# metrics to collect
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"] # not recording L1 norm of the weight vector anymore!
SW_metrics = [f"SW_{metric}" for metric in metrics]
metric_vars = ["timestep"] + SW_metrics + ["time_elapsed"] # no longer recording x_t's l1 norm!

# create a .csv that we want to load + repeatedly save to, and save our columns
with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "w") as file:
    file.write(','.join([str(entry) for entry in metric_vars]))
    file.write('\n')

# specify default maximum sizes of our train and test sets
N_train_max, N_test_max = np.inf, np.inf
    
# get our eta and lmbda
eta = 2.0 ** log2eta
lmbda = 10.0 ** log10lmbda

# compute our lmbda_t
lmbda_t = eta * lmbda

# ufuncs for faster computing
def updateW(w_j, theta_j):
    return np.sign(theta_j) * np.maximum(0.0, np.abs(theta_j) - lmbda_t)
ufunc_updateW = np.frompyfunc(updateW, 2, 1)

# load in our dataset, with automatic shuffling
X, y = load_data(dataset=dataset, remove_zeroes=True, add_bias=False, seed=seed)
N, D = X.shape # no bias yet
D += 1 # account for bias

# get train + test set sizes + sanity-check for leakage: 70/30 split!
N_train = int(N * 0.7) # effective_N_trains[dataset]
N_test = min(N - N_train, N_test_max)
assert N_train + N_test <= N, "Train-test leakage!"

# split data into train + test partitions: make sure y_train, y_test are COLUMNS!
X_train, y_train = X[:N_train], y[:N_train].reshape(-1, 1)
X_test, y_test = X[-N_test:], y[-N_test:].reshape(-1, 1)

# what's our WRS computation resolution? How often are we recording metrics?
resolution_step = int(N_train / 200)

# set a seed for reproducibility
np.random.seed(858)

###########

# for moving average, we'll still have a WRS-type reservoir implemented as a dictionary
reservoir = {} # members of form (w_t, no. of subseq. passive steps)

# "weight" is just the timestep the solutions existed.
w_i_vals, wid_i_vals = np.array([]), np.array([])

# store the weight vectors in the reservoir as a matrix.
W_matrix = np.array([])

# create a counter for the number of unique weights observed (0 vector corresponds to idx 0)
weight_id = 0

###########

# initialize starting wt_dense and theta_t as just zeroes
thetat_dense = np.zeros(shape=(D,))
wt_dense = np.zeros(shape=(D,))

# track our current candidate weights
candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1)}

# iterate thru our online stream data
for t_step in range(N_train):

    # instantiate our row for insertion into metric_logs
    row = [t_step]

    ######## OBSERVE OUR DATA

    # get our x_t as a SPARSE vector (WITH NO APPENDED 1 FOR BIAS) and our y_t
    x_t = X_train[t_step]
    y_t = y_train[t_step, 0]
    
    ######## COMPUTING MOVING-AVERAGE METRICS USING PRE-UPDATE CANDIDATES!

    # for moving average, we only need to look at simple average. No weighting involved.
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):
        
        # simple average. No fancy tricks
        wt_SA = W_matrix.mean(axis=1).reshape(-1, 1)
        row += metric_pack(wt_SA, x_t, y_t, X_train, y_train, X_test, y_test)

    ######## PASSIVE-AGGRESSIVE UPDATES: RESERVOIR FIRST, AND THEN WEIGHT CANDIDATE!

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # go aggressive if solution requires a passive-aggressive update.
    if inst_hinge_loss_dense > 1e-4:
        
        #### RESERVOIR MANAGEMENT
        
        # for sliding window approaches, our (deterministic) weight is just the timestep.
        w_i = t_step # we want to keep the most recent ones + remove the older ones.
        
        # encode the w_i for our candidate vector
        candidate_wt_dense["w_i"] = w_i
        
        # with weights w_i computed, consider whether add to reservoir
        # ALWAYS add, of course, if reservoir is not full.
        if len(w_i_vals) < K:

            # deterministically add the weight-vectors to our reservoir!
            reservoir[weight_id] = copy.deepcopy(candidate_wt_dense)

            # update our {k_i, w_i, PS}_vals and W_matrix. SMALLEST k_i is always LEFT-MOST!
            if len(w_i_vals) == 0:
                w_i_vals = np.array([w_i])
                wid_i_vals = np.array([weight_id])
                W_matrix = np.array(candidate_wt_dense["wt"])

            # SMALLEST w_i is always LEFT-MOST!
            elif w_i < w_i_vals[0]:
                w_i_vals = np.concatenate([[w_i], w_i_vals])
                wid_i_vals = np.concatenate([[weight_id], wid_i_vals])
                W_matrix = np.concatenate([candidate_wt_dense["wt"], W_matrix], axis=1)

            # put our current w_i to the very end, should be fine. Just keep min. w_i in front!
            else:
                w_i_vals = np.concatenate([w_i_vals, [w_i]])
                wid_i_vals = np.concatenate([wid_i_vals, [weight_id]])
                W_matrix = np.concatenate([W_matrix, candidate_wt_dense["wt"]], axis=1)

        # reservoir already full! need to do sampling.
        else:

            # find the smallest w_i, which we will set as our threshold
            min_dense_idx = np.argmin(w_i_vals)
            T = w_i_vals[min_dense_idx]

            # if current k_i > T, replace the members in reservoirs + speed-arrays
            if w_i > T:

                # delete the old reservoir member from the dictionary + add new one
                del reservoir[wid_i_vals[min_dense_idx]]
                reservoir[weight_id] = copy.deepcopy(candidate_wt_dense)

                # update our speed-arrays
                w_i_vals[min_dense_idx] = w_i
                wid_i_vals[min_dense_idx] = weight_id
                W_matrix[:,min_dense_idx] = candidate_wt_dense["wt"].flatten()
        
        #### MAKE OUR WEIGHT-UPDATE FOR FSOL

        # sparse update for thetat_dense. separate out the bias term!
        np.add.at(thetat_dense,  x_t.indices, (eta*y_t)*x_t.data)
        thetat_dense[-1] += eta*y_t

        # get the active indices (i.e., where x_t is nonzero) + make updates inplace!
        active_j = x_t.indices
        ufunc_updateW.at(wt_dense, active_j,  thetat_dense[active_j])

        # finally, account for the bias term too!
        wt_dense[D-1] = updateW(wt_dense[D-1], thetat_dense[D-1])
        
        #### CREATE OUR NEW WEIGHT-CANDIDATE ENTRY
        
        # RECALL: WILL LATER ADD IN "w_i" and "k_i" entries upon aggressive update.
        candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1)}

        # also need to increment our unique encountered weight_id
        weight_id += 1
    
    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    
    # only record the time elapsed if we're logging this timestep
    if t_step != 0:
        row.append(end_time - start_time)

    # finally, add this row to our .csv
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')