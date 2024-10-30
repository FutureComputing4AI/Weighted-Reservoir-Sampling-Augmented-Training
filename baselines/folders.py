# the only import that we need
import os

# what are our models and datasets of interest?
models = ["TOPK"]
bases = ["FSOL", "PAC"]
ordered_datasets = ["pcmac_binary_sparse", "dexter_binary_sparse", "w8a_binary_sparse", 
                    "webspam_binary_sparse", "dorothea_binary_sparse", "sst2_binary_sparse", 
                    "mnist8-4+9_binary_sparse", "real-sim_binary_sparse", "newsgroups_binary_sparse",
                    "news20_binary_sparse", "rcv1_binary_sparse", "url_binary_sparse",
                    "avazu-app_binary_sparse", "avazu-site_binary_sparse", 
                    "kdd2010-a_binary_sparse", "criteo_binary_sparse"]

# create our directories
if "results" not in os.listdir():
    os.mkdir("results")
    
# create our subdirectories
for model in models:
    
    # create a subdirectory for this model
    if model not in os.listdir("results"): 
        os.mkdir(f"results/{model}")
    
    # subdirectories for our baseline models PAC and FSOL
    for base in bases:
        if base not in os.listdir(f"results/{model}"): 
            os.mkdir(f"results/{model}/{base}")
        
        # create a subdirectory for each dataset
        for dataset in ordered_datasets:

            # check if we have directory, else make it
            if dataset not in os.listdir(f"results/{model}/{base}"):
                os.mkdir(f"results/{model}/{base}/{dataset}")