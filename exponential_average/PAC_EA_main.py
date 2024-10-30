import numpy as np
import pandas as pd
import sys, copy, os, pickle
import time
import scipy
from scipy import sparse

# dataloaders + update methods, metric extractors
from utils_v2 import *
from metric_extractors import * 

# what are our possible settings for this script? start with the model + dataset
models = ["PAC"]
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# list of settings to build up
settings = []
for model in models: # 1x
    for dataset in ordered_datasets: # 16x
        for seed in range(5): # 5x
            settings.append((model, dataset, seed))
                
# command-line argument to get which setting we're running (80 settings total!)
model, dataset, seed = settings[int(sys.argv[1])]

# immediately load in the best hyperparameters for this dataset + model
log10Cerr = pd.read_csv("base_variants/PAC_hparams.csv")\
.query(f"dataset == '{dataset}'")[["log10Cerr"]].values[0,0]

# what's the implied file name?
fheader = f"model={model}_seed={seed}"

# metrics to collect
metrics = ["obs-acc", "obs-hinge", 
           "train-set-acc", "train-set-hinge", 
           "test-set-acc", "test-set-hinge",
           "sparsity"] # not recording L1 norm of the weight vector anymore!
EW_metrics = [f"EW_{metric}" for metric in metrics]
metric_vars = ["timestep"] + EW_metrics + ["time_elapsed"] # no longer recording x_t's l1 norm!


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

###########

# initialize starting wt_dense
wt_dense = np.zeros(shape=(D,))

# FOR EXPONENTIAL-WINDOW, WE JUST NEED TO DO A RECURSIVE UPDATE
wt_exp = np.zeros(shape=(D,))

# iterate thru our online stream data
for t_step in range(N_train):

    # instantiate our row for insertion into metric_logs
    row = [t_step]

    ######## OBSERVE OUR DATA

    # get our x_t as a SPARSE vector (WITH NO APPENDED 1 FOR BIAS) and our y_t
    x_t = X_train[t_step]
    y_t = y_train[t_step, 0]
    
    ######## COMPUTING EXPONENTIAL-AVERAGE METRICS USING PRE-UPDATE CANDIDATES!

    # just need to look at moving exponential average of weights
    if (t_step % resolution_step == 0) or (t_step == N_train - 1):
        
        # no need to recompute anything :)
        row += metric_pack(wt_exp.reshape(-1,1), x_t, y_t, X_train, y_train, X_test, y_test)
        
    ######## PASSIVE-AGGRESSIVE UPDATES: RESERVOIR FIRST, AND THEN WEIGHT CANDIDATE!

    # start a timer TO RECORD INTO OUR LOGS
    start_time = time.time()

    # compute hinge loss
    inst_hinge_loss_dense = l(wt_dense.reshape(-1,1), x_t, y_t)[0,0]

    # go aggressive if solution requires a passive-aggressive update.
    if inst_hinge_loss_dense > 1e-4:
        
        #### RESERVOIR MANAGEMENT
        
        # just directly update our wt_exp: \bar{w}_t = 0.1 \bar{w}_{t-1} + 0.9 w_t
        wt_exp = (wt_exp * 0.1) + (wt_dense * 0.9)
        
        #### MAKE OUR WEIGHT-UPDATE FOR PAC

        # MAKE OUR WEIGHT-UPDATE FOR PAC
        tau_t = inst_hinge_loss_dense / ( (0.5/C_err) + (sparse.linalg.norm(x_t) ** 2) )
        np.add.at(wt_dense,  x_t.indices, (tau_t*y_t)*x_t.data)
        wt_dense[-1] += tau_t*y_t
    
    # end our timer + compute time elapsed (in seconds)
    end_time = time.time()
    
    # only record the time elapsed if we're adding it to our dataframe
    if t_step != 0:
        row.append(end_time - start_time)

    # finally, add this row to our .csv
    if (t_step % resolution_step == 0) or (t_step == N_train - 1):
        with open(f"results/{model}/{dataset}/{fheader}_metrics.csv", "a") as file:
            file.write(','.join([str(entry) for entry in row]))
            file.write('\n')