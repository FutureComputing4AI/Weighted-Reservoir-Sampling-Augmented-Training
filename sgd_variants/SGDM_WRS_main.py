#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import sys, copy, os, pickle
import time
import scipy

# dataloader methods, metric extractors
from utils_v2 import load_data 
from metric_extractors import *

# what are our possible settings for this script? start with the model + dataset
models = ["sgd+momentum"]
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

# specify default maximum sizes of our train and test sets
N_train_max, N_test_max = np.inf, np.inf

# load in our dataset, with automatic shuffling
X, y = load_data(dataset=dataset, remove_zeroes=True, add_bias=False, seed=seed)
N, D = X.shape # no bias yet
D += 1 # account for bias.

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
fheader = f"model={model}_K={K}_seed={seed}_Edified"

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

# initialize our w
wt_dense = np.zeros(shape=(D, ))

# track our current candidate weights
candidate_wt_dense = {"wt" : wt_dense.copy().reshape(-1,1), "PS" : 0}

# for use in tracking indices
full_indices = np.arange(D-1)

# create vector of size D
stale = np.zeros(shape=(D-1,))
b_t = np.zeros(shape=(D-1,))

# let's pick a momentum parameter of \mu = 0.9. No weight decay, so lmbda = 0.0,
# also no dampening so \tau = 0.0. Let's use learning rate 0.1
mu, gamma = 0.9, 0.1

# iterate through our training datapoints
for t_step in range(N_train):

    # instantiate our row for insertion into metric_logs
    row = [t_step]

    ######## OBSERVE OUR DATA

    x_t = X_train[t_step,:]
    y_t = y_train[t_step, 0]

    ######## COMPUTING INSTANTANEOUS + WRS METRICS USING PRE-UPDATE CANDIDATES!

    # instantaneous metrics only if recording timestep or final timestep
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):

        # just take our instantaneous solution + update
        row += metric_pack(wt_dense.reshape(-1, 1), x_t, y_t, X_train, y_train, X_test, y_test)

        # simple average, no voting-based zeroing
        wt_WRS_SA = W_matrix.mean(axis=1).reshape(-1, 1)
        row += metric_pack(wt_WRS_SA, x_t, y_t, X_train, y_train, X_test, y_test)

    ######## ITERATIVE UPDATES: RESERVOIR FIRST (IF ERROR), AND THEN WEIGHT CANDIDATE!

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()
    
    # weight vector active clean-up
    for idx in x_t.indices:
        wt_dense[idx] += -np.power(gamma, t_step - stale[idx]) * b_t[idx] 
        stale[idx] = t_step

    # compute hinge loss on current candidate -- we will consider adding to reservoir if make error.
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # see if we need to add to the reservoir (based on strict inst. train accuracy)
    if np.sign((x_t @ wt_dense[:-1]) + wt_dense[-1]) == y_t:

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
            
            # total-cleanup before adding into the reservoir
            wt_dense[:-1] += -np.power(gamma, t_step - stale) * b_t
            stale[:] = t_step
            
            # deterministically add the weight-vectors to our reservoir!
            candidate_wt_dense["wt"] = wt_dense.copy().reshape(-1, 1)
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
                
                # total-cleanup before adding into the reservoir
                wt_dense[:-1] += -np.power(gamma, t_step - stale) * b_t
                stale[:] = t_step
                
                # delete the old reservoir member from the dictionary + add new one
                del reservoir[wid_i_vals[min_dense_idx]]
                candidate_wt_dense["wt"] = wt_dense.copy().reshape(-1, 1)
                reservoir[weight_id] = copy.deepcopy(candidate_wt_dense)

                # update our speed-arrays
                k_i_vals[min_dense_idx] = k_i
                w_i_vals[min_dense_idx] = w_i
                PS_i_vals[min_dense_idx] = PS_i
                wid_i_vals[min_dense_idx] = weight_id
                W_matrix[:,min_dense_idx] = candidate_wt_dense["wt"].flatten()
                
        # finally, reset counter
        candidate_wt_dense["PS"] = 0.0

    # if we stayed passive, just increment the counters. candidate_wts lives another cycle.
    else:

        # increment counters on candidate_wt_dense's subseq. PS steps.
        candidate_wt_dense["PS"] += 1.0

        #### MAKE OUR WEIGHT-UPDATE FOR STOCHASTIC GRADIENT DESCENT WITH MOMENTUM
        # reference: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

        # tracking the base algorithm time
        base_start_time = time.time()

        # let's do casework on the gradient + timestep
        if t_step == 0:
            b_t = -y_t * np.array(x_t.todense()).flatten() # dense
            b_scalar = 1.0
        else:

            # we know we're not on the first timestep - regardless, need to account for momentum
            # b_t[b_t != 0.0] *= mu (altar of my sins)
            b_scalar *= mu

            # checking that we're not going into unstable fp range, else clean up
            if np.abs(b_scalar) < 1e-6:
                b_t *= b_scalar
                b_scalar = 1

            # now, if we made a mistake, our subgradient will be nontrivial
            if inst_hinge_loss_dense > 1e-4:
                np.add.at(b_t, x_t.indices, -y_t*x_t.data / b_scalar)

    # finally, do our gradient descent step
    mask = b_t != 0.0
    np.add.at(wt_dense, full_indices[mask], -gamma*b_t[mask])

    # tracking the base algorithm time
    base_end_time = time.time()

    # update our candidate_wt_dense
    candidate_wt_dense["wt"] = wt_dense.copy().reshape(-1, 1)

    # also need to increment our unique encountered weight_id
    weight_id += 1

    #### FINISH LOGGING + WRITE TO FILE.

    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    
    # only record the time elapsed if we're adding it to our dataframe
    if t_step != 0:
        row.append(end_time - start_time)
        row.append(base_end_time - base_start_time)

    # finally, add this row to our .csv
    if (len(w_i_vals) != 0) and (( (t_step % resolution_step == 0) or (t_step == N_train - 1))):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')

