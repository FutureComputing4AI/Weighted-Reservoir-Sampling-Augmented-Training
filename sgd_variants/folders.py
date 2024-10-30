# the only import that we need
import os

# what are our models and datasets of interest? NOTE THAT SIMPLE AVERAGE ON AVAZU-APP WAS WAYYYY TOO SLOW!!!
models = ["tgd"] # "sgd+momentum", "adagrad", 

# which datasets are we working with?
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# create our directories
if "results" not in os.listdir():
    os.mkdir("results")
if "outputs" not in os.listdir():
    os.mkdir("outputs")
if "errors" not in os.listdir():
    os.mkdir("errors")
    
# create our subdirectories
for model in models:
    
    # create a subdirectory for this model
    if model not in os.listdir("results"): 
        os.mkdir(f"results/{model}")
        
    # create a subdirectory for each dataset
    for dataset in ordered_datasets:
        
        # check if we have directory, else make it
        if dataset not in os.listdir(f"results/{model}"):
            os.mkdir(f"results/{model}/{dataset}")