import numpy as np
import pandas as pd
import sys, copy, os, shutil, math, pickle
import time
import scipy
from scipy import sparse 

# dataloaders + update methods, metric extractors
from utils_v2 import *
from metric_extractors import * 

# what are our possible settings for this script? start with the model
models = ["PAC"]

# only looking at sparse datasets, ordered by difficulty / time required.
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# PAC-specific slack_vars
C_errs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# list of settings to build up
settings = []

# create a separate job for each model + dataset + C_err + seed
for model in models:
    for dataset in ordered_datasets:
        for C_err in C_errs:
            for seed in range(5):
                
                # add to our list of settings (total of 560)
                settings.append((model, dataset, C_err, seed))
                
# command-line argument to get which setting we're running
model, dataset, C_err, seed = settings[int(sys.argv[1])]

# get the log-version just for naming
log10Cerr = int(np.log10(C_err))

# what's the implied file name? see if we've already run this setting + do some checkpointing
fheader = f"model={model}_log10Cerr={log10Cerr}_seed={seed}"
if f"{fheader}_metrics.csv" in os.listdir(f"results/{model}/{dataset}"):
    sys.exit()

# let's use full train/test splits!
N_train_max, N_test_max = np.inf, np.inf

# revised set of metrics
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"]

# which variables that we want to keep a record of?
inst_metrics = [f"inst_{metric}" for metric in metrics]
metric_vars = ["timestep"] + inst_metrics + ["time_elapsed"]

# create a .csv that we want to load + repeatedly save to, and save our columns
with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "w") as file:
    file.write(','.join([str(entry) for entry in metric_vars]))
    file.write('\n')

# specify default maximum sizes of our train and test sets
N_train_max, N_test_max = np.inf, np.inf

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

# initialize starting wt_dense
wt_dense = np.zeros(shape=(D,))

# iterate thru our online stream data
for t_step in range(N_train):

    # instantiate our row for insertion into metric_logs
    row = [t_step]

    ######## OBSERVE OUR DATA

    ## divide up wt_dense into wt and wt_bias.

    # get our x_t as a SPARSE vector (WITH NO APPENDED 1 FOR BIAS) and our y_t
    x_t = X_train[t_step]
    y_t = y_train[t_step, 0]
    
    ######## COMPUTING INSTANTANEOUS METRICS USING PRE-UPDATE CANDIDATES!

    # instantaneous metrics only if recording timestep or final timestep
    if ( (t_step % resolution_step == 0) or (t_step == N_train - 1)):
        row += metric_pack(wt_dense.reshape(-1,1), x_t, y_t, X_train, y_train, X_test, y_test)
    else:
        row += [np.nan]*7

    ######## PASSIVE-AGGRESSIVE UPDATES

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss for DENSE candidate and train-acc for SPARSE candidate
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # go aggressive if the DENSE solution requires a passive-aggressive update.
    if inst_hinge_loss_dense > 1e-4:

        # MAKE OUR WEIGHT-UPDATE FOR PAC
        tau_t = l(wt_dense, x_t, y_t)[0,0] / ( (0.5/C_err) + (sparse.linalg.norm(x_t) ** 2) )
        np.add.at(wt_dense,  x_t.indices, (tau_t*y_t)*x_t.data)
        wt_dense[-1] += tau_t*y_t

    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    row.append(end_time - start_time)

    # finally, add this row to our dataframe
    if ( (t_step % resolution_step == 0) or (t_step == N_train - 1)):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')

