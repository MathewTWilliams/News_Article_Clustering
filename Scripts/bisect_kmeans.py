# Author: Matt Williams
# Version: 07/15/2022


from sklearn.cluster import BisectingKMeans
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, Categories, WordVectorModels


def run_bisect_kmeans(word_vec_model, n_init = 1, max_iter = 300, tol = 1e-4, \
    bisecting_strategy = "biggest_inertia"): 
    
    bisect_kmeans = BisectingKMeans(n_clusters=len(Categories.get_values_as_list()), \
                                    init = "k-means++", n_init = n_init, max_iter=max_iter, \
                                    tol = tol, bisecting_strategy=bisecting_strategy)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.BI_KMEANS.value,
        "init" : "k-means++",
        "n_init" : n_init, 
        "max_iter" : max_iter, 
        "tol" : tol, 
        "bisecting_strategy" : bisecting_strategy 
    }

    run_clustering(word_vec_model, bisect_kmeans, model_details)


# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_bisect_kmeans(word_vec_model)