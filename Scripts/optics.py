# Author: Matt Williams
# Version: 07/15/2022



from sklearn.cluster import OPTICS
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, WordVectorModels
import numpy as np

def run_optics(word_vec_model, min_samples = 5, max_eps = np.inf, metric = 'minkowski', \
            p = 2, cluster_method = 'xi', leaf_size = 30, min_cluster_size = None, \
            xi = 0.05): 
    
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, leaf_size=leaf_size, \
                    metric=metric, p = p, cluster_method=cluster_method, xi = xi, \
                    min_cluster_size=min_cluster_size)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.OPTICS.value, 
        "min_samples" : min_samples, 
        "max_eps" : max_eps, 
        "leaf_size" : leaf_size, 
        "metric" : metric, 
        "p" : p, 
        "cluster_method" : cluster_method, 
        "xi" : xi, 
        "min_cluster_size" : min_cluster_size

    }

    run_clustering(word_vec_model, optics, model_details)

# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_optics(word_vec_model)