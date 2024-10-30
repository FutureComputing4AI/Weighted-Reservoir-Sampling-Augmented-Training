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
models = ["FSOL"]

# only looking at sparse datasets, ordered by difficulty / time required.
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# FSOL-specific parameters
log2etas = np.arange(-3, 9+1.0)
log10lmbdas = np.array([-np.inf] + list(np.arange(-3.0, 3+1.0)))

# list of settings to build up
settings = []

# create a separate job for each model + dataset + log2etas + log10lmbdas setting
for model in models:
    for dataset in ordered_datasets:
        for log2eta in log2etas:
            for log10lmbda in log10lmbdas:
                for seed in range(5):
                
                    # add to our list of settings
                    settings.append((model, dataset, log2eta, log10lmbda, seed))
                
# command-line argument to get which setting we're running
model, dataset, log2eta, log10lmbda, seed = settings[int(sys.argv[1])]

# what's the implied file name? see if we've already run this setting + do some checkpointing
fheader = f"model={model}_log2eta={log2eta}_log10lmbda={log10lmbda}_seed={seed}"
if f"{fheader}_metrics.csv" in os.listdir(f"results/{model}/{dataset}"):
    sys.exit()

# let's use full train/test splits!
N_train_max, N_test_max = np.inf, np.inf

# set of metrics
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"] # not recording L1 norm of the weight vector anymore!

# which variables that we want to keep a record of?
inst_metrics = [f"inst_{metric}" for metric in metrics]
metric_vars = ["timestep"] + inst_metrics + ["time_elapsed"]

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

# ufuncs for faster compute
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

# what's our WRS computation resolution? How often do we store metrics?
resolution_step = int(N_train / 200)

# set a seed for reproducibility
np.random.seed(858)

# initialize starting wt_dense and theta_t as just zeroes
thetat_dense = np.zeros(shape=(D,))
wt_dense = np.zeros(shape=(D,))

# iterate thru our online stream data
for t_step in range(N_train):

    # instantiate our row for insertion into logs
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

    ######## PASSIVE-AGGRESSIVE UPDATE

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss for DENSE candidate and train-acc for SPARSE candidate
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # go aggressive if the DENSE solution requires a passive-aggressive update.
    if inst_hinge_loss_dense > 1e-4:

        # MAKE OUR WEIGHT-UPDATE FOR FSOL

        # sparse update for thetat_dense. separate out the bias term!
        np.add.at(thetat_dense,  x_t.indices, (eta*y_t)*x_t.data)
        thetat_dense[-1] += eta*y_t

        # get the active indices (i.e., where x_t is nonzero) + make updates inplace!
        active_j = x_t.indices
        ufunc_updateW.at(wt_dense, active_j,  thetat_dense[active_j])

        # finally, account for the bias term too!
        wt_dense[D-1] = updateW(wt_dense[D-1], thetat_dense[D-1])

    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    row.append(end_time - start_time)

    # finally, add this row to our dataframe
    if ( (t_step % resolution_step == 0) or (t_step == N_train - 1)):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')

