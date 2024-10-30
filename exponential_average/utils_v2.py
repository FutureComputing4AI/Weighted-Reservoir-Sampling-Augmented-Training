import numpy as np
from scipy.sparse import csr_matrix, load_npz

# encode our instantaneous loss function directly too
def l(w, x, y):
    
    # make sure we have properly encoded binary classification.
    assert y in [-1, 1, -1.0, 1.0], "Invalid y value."
    
    # make sure everything is the right dimensions (column vector)!
    if "todense" in dir(x):
        x = np.array(x.todense()).reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
        
    # repeat for w
    if "todense" in dir(w):
        w = np.array(w.todense()).reshape(-1, 1)
    else:
        w = w.reshape(-1, 1)

    return np.maximum( 0.0, 1.0 - ( y * (w.T @ x) ) )


# data-loading function
def load_data(dataset, remove_zeroes=False, add_bias=True, seed=858):

    # create a list of supported datasets (aside from "QUERY")
    datasets = ["sst2_binary_sparse", "real-sim_binary_sparse", "rcv1_binary_sparse", "news20_binary_sparse", 
                "webspam_binary_sparse", "pcmac_binary_sparse", "dorothea_binary_sparse", "dexter_binary_sparse",
                "url_binary_sparse", "mnist8-4+9_binary_sparse", "w8a_binary_sparse", "newsgroups_binary_sparse", 
                "avazu-app_binary_sparse", "avazu-site_binary_sparse", "criteo_binary_sparse", "kdd2010-a_binary_sparse"]
    
    # set a seed for shuffling reproducibility
    np.random.seed(seed)
    
    # this is just to return the list of possible datasets
    if dataset == "QUERY":
        
        # just return the list of possible datasets
        return sorted(datasets)
    
    # load our binary datasets - let's use both newsgroups_binary_sparse + news20.binary (more public + accepted)
    elif dataset == "newsgroups_binary_sparse":
        
        # directly load from disk. already encoded ys for (computers + science) as +1, !(computers + science) as -1.
        X = load_npz(file="../data/newsgroups_binary_X.npz")
        y = np.loadtxt(fname="../data/newsgroups_binary_y.txt")
        
    elif dataset == "sst2_binary_sparse":
        
        # directly load from disk.
        X = load_npz(file="../data/sst2_binary_X.npz")
        y = np.loadtxt("../data/sst2_binary_y.txt")
        
    elif dataset == "real-sim_binary_sparse":
        
        # directly load from disk.
        X = load_npz(file="../data/real-sim_binary_X.npz")
        y = np.loadtxt("../data/real-sim_binary_y.txt")

    elif dataset == "rcv1_binary_sparse":

        # just load in our data as is according to our files ("old" folder has the train data only)
        X = load_npz("../data/rcv1_binary_X.npz")
        y = np.loadtxt("../data/rcv1_binary_y.txt")
        
    elif dataset == "news20_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/news20_binary_X.npz")
        y = np.loadtxt("../data/news20_binary_y.txt")
        
    elif dataset == "webspam_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/webspam_binary_X.npz")
        y = np.loadtxt("../data/webspam_binary_y.txt")
        
    elif dataset == "pcmac_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/pcmac_binary_X.npz")
        y = np.loadtxt("../data/pcmac_binary_y.txt")
        
    elif dataset == "dorothea_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/dorothea_binary_X.npz")
        y = np.loadtxt("../data/dorothea_binary_y.txt")
        
    elif dataset == "dexter_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/dexter_binary_X.npz")
        y = np.loadtxt("../data/dexter_binary_y.txt")
        
    elif dataset == "url_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/url_binary_X.npz")
        y = np.loadtxt("../data/url_binary_y.txt")
        
    elif dataset == "mnist8-4+9_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/mnist8-4+9_binary_X.npz")
        y = np.loadtxt("../data/mnist8-4+9_binary_y.txt")
        
    elif dataset == "w8a_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/w8a_binary_X.npz")
        y = np.loadtxt("../data/w8a_binary_y.txt")
        
    # added April 2024
    elif dataset == "avazu-app_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/avazu-app_binary_X.npz")
        y = np.loadtxt("../data/avazu-app_binary_y.txt")
        
    # added April 2024
    elif dataset == "avazu-site_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/avazu-site_binary_X.npz")
        y = np.loadtxt("../data/avazu-site_binary_y.txt")
        
    elif dataset == "criteo_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/criteo_binary_X.npz")
        y = np.loadtxt("../data/criteo_binary_y.txt")
        
    elif dataset == "kdd2010-a_binary_sparse":
        
        # just load in our data as is according to our files
        X = load_npz("../data/kdd2010-a_binary_X.npz")
        y = np.loadtxt("../data/kdd2010-a_binary_y.txt")
        
        
    # note that there may be rows of all zeros. Let's remove these rows (breaks the optim. of max margin if no bias term)
    if remove_zeroes:
        
        # find the indices of all zero rows.
        nonzero_rows = np.array((np.abs(X).sum(axis=1) != 0.0)).flatten()
        X = X[nonzero_rows]
        y = y[nonzero_rows]

    # shuffle all of our data, put into X and y
    shuffle_idxs = np.random.choice(a=np.arange(X.shape[0]), 
                                    size=X.shape[0], replace=False)

    # get our full X and y, BUT SHUFFLED
    X = X[shuffle_idxs]
    y = y[shuffle_idxs]
       
    # append a column of 1's to our X to have a bias term, if called for
    if add_bias:
        X = csr_matrix(np.hstack([np.ones(shape=(X.shape[0], 1)), X.todense()])) if "todense" in dir(X) else np.hstack([np.ones(shape=(X.shape[0], 1)), X])
        
    # at the very end, just return the X, y
    return X, y