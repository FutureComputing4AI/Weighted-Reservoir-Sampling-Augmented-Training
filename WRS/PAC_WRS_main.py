import numpy as np
import pandas as pd
import sys, copy, os, pickle
import time
import scipy
from scipy import sparse

# dataloaders + update methods, metric extractors
from utils_v2 import *
from metric_extractors import * 

# what are our possible settings for this script? start with the model
models = ["PAC"]

# which datasets are we working with?
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# reservoir size K settings are the exact same, also same exponential weights or no exponential weights
Ks = [1, 4, 16, 64]

'''
Explanation of different weighting schemes:
1. exp-dense: w = exp(# of subsequent passive steps of DENSE weight)
2. dense: w = # of subsequent passive steps of DENSE weight
'''
weight_schemes = ["exp-dense", "dense"]

# list of settings to build up
settings = []

# create a separate job for each combination of model, dataset, weight_scheme, K value
for model in models:
    for dataset in ordered_datasets:
        for weight_scheme in weight_schemes:
            for K in Ks:
                for seed in range(5):
                    settings.append((model, dataset, weight_scheme, K, seed))
                
# command-line argument to get which setting we're running
model, dataset, weight_scheme, K, seed = settings[int(sys.argv[1])]

# immediately load in the best hyperparameters for this dataset + model
log10Cerr = pd.read_csv("base_variants/PAC_hparams.csv")\
.query(f"dataset == '{dataset}'")[["log10Cerr"]].values[0,0]

# what's the implied file name? see if we've already run this setting + do some checkpointing
fheader = f"model={model}_ws={weight_scheme}_K={K}_seed={seed}"
if f"{fheader}_reservoir-structures.pickle" in os.listdir(f"results/{model}/{dataset}"):
    sys.exit()

# set of metrics
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"] # not recording L1 norm of the weight vector anymore!

# which variables that we want to keep a record of?
WRS_metrics = [f"WRS_{metric}_{a_type}{v_type}" for a_type in ["WA", "SA"] 
               for v_type in ["", "_VZ"] for metric in metrics]
metric_vars = ["timestep"] + WRS_metrics + ["time_elapsed"]


# create a .csv that we want to load + repeatedly save to, and save our columns
with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "w") as file:
    file.write(','.join([str(entry) for entry in metric_vars]))
    file.write('\n')

# specify default maximum sizes of our train and test sets
N_train_max, N_test_max = np.inf, np.inf
    
# get our log10Cerr for PAC
C_err = 10.0 ** log10Cerr

# load in our dataset, with automatic shuffling
X, y = load_data(dataset=dataset, remove_zeroes=True, add_bias=False, seed=seed)
N, D = X.shape # no bias yet
D += 1 # account for bias

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

# initialize starting wt_dense
wt_dense = np.zeros(shape=(D,))

# track our current candidate weights
# WILL LATER ADD IN "w_i" and "k_i" entries upon aggressive update.
candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1), "PS" : 0}

# iterate thru our online stream data
for t_step in range(N_train):

    # instantiate our row for insertion into metric_logs
    row = [t_step]

    ######## OBSERVE OUR DATA

    # get our x_t as a SPARSE vector (WITH NO APPENDED 1 FOR BIAS) and our y_t
    x_t = X_train[t_step]
    y_t = y_train[t_step, 0]
    
    ######## COMPUTING INSTANTANEOUS METRICS USING PRE-UPDATE CANDIDATES!

    # instantaneous metrics only if recording timestep or final timestep
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):
        
        # extract normalized weights + get weighted-average, no voting-based zeroing. Add metrics.
        normalized_w_i_vals = extract_normalized_reservoir_weights(w_i_vals)
        wt_WRS_WA = (W_matrix @ normalized_w_i_vals).reshape(-1,1)
        row += metric_pack(wt_WRS_WA, x_t, y_t, X_train, y_train, X_test, y_test)
        
        # weighted average, yes voting-based zeroing
        zero_mask = (W_matrix != 0.0).mean(axis=1) < 0.5 # 0 out idxs where most WRS vecs are 0.0
        wt_WRS_WA_VZ = wt_WRS_WA.copy()
        wt_WRS_WA_VZ[zero_mask] = 0.0
        row += metric_pack(wt_WRS_WA_VZ, x_t, y_t, X_train, y_train, X_test, y_test)
        
        # simple average, no voting-based zeroing
        wt_WRS_SA = W_matrix.mean(axis=1).reshape(-1, 1)
        row += metric_pack(wt_WRS_SA, x_t, y_t, X_train, y_train, X_test, y_test)
        
        # simple average, yes voting-based zeroing (don't regen mask!)
        wt_WRS_SA_VZ = wt_WRS_SA.copy()
        wt_WRS_SA_VZ[zero_mask] = 0.0
        row += metric_pack(wt_WRS_SA_VZ, x_t, y_t, X_train, y_train, X_test, y_test)

    ######## PASSIVE-AGGRESSIVE UPDATES: RESERVOIR FIRST, AND THEN WEIGHT CANDIDATE!

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # go aggressive if solution requires a passive-aggressive update.
    if inst_hinge_loss_dense > 1e-4:
        
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
                W_matrix = np.array(candidate_wt_dense["wt"])

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
        
        #### MAKE OUR WEIGHT-UPDATE FOR PAC

        # MAKE OUR WEIGHT-UPDATE FOR PAC
        tau_t = l(wt_dense, x_t, y_t)[0,0] / ( (0.5/C_err) + (sparse.linalg.norm(x_t) ** 2) )
        np.add.at(wt_dense,  x_t.indices, (tau_t*y_t)*x_t.data)
        wt_dense[-1] += tau_t*y_t
        
        #### CREATE OUR NEW WEIGHT-CANDIDATE ENTRY
        
        # RECALL: WILL LATER ADD IN "w_i" and "k_i" entries upon aggressive update.
        candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1), "PS" : 0.0}

        # also need to increment our unique encountered weight_id
        weight_id += 1
    
    # if we stayed passive, just increment the counters. candidate_wts lives another cycle.
    else:
        
        # increment counters on candidate_wt_dense's subseq. PS steps.
        candidate_wt_dense["PS"] += 1.0
    
    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    
    # only record the time elapsed if we're adding it to our dataframe
    if t_step != 0:
        row.append(end_time - start_time)

    # finally, add this row to our .csv
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')
            
# at the very end, save our reservoir structures
reservoir_structures = {"k_i_vals" : k_i_vals, 
                        "w_i_vals" : w_i_vals, 
                        "PS_i_vals" : PS_i_vals, 
                        "wid_i_vals" : wid_i_vals,
                        "W_matrix" : W_matrix}
with open(f"results/{model}/{dataset}/{fheader}_reservoir-structures.pickle", "wb") as file:
    pickle.dump(obj=reservoir_structures, file=file)