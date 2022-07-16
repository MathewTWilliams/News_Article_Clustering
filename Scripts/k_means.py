#Author: Matt Williams
#Version: 07/11/2022

from sklearn.cluster import KMeans
from utils import Categories, ClusteringAlgorithms, \
    RESULT_WORD_VEC_MOD_KEY, RESULT_CLUSTER_ALGO_KEY, WordVectorModels
from run_clustering import run_clustering


def run_k_means(vec_model_name, n_init = 10, tol = 1e-4): 


    k_means = KMeans(n_clusters = len(Categories.get_values_as_list()), \
        n_init = n_init, tol = tol)


    model_details = {
        RESULT_WORD_VEC_MOD_KEY : vec_model_name, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.KMEANS.value, 
        "n_init" : n_init, 
        "tol" : tol, 
    }

    run_clustering(vec_model_name, k_means, model_details)

# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_k_means(word_vec_model)