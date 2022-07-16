# Author: Matt Williams
# Version: 07/15/2022


from sklearn.cluster import DBSCAN
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, WordVectorModels


def run_dbscan(word_vec_model, eps = 0.5, min_samples = 5, metric = 'euclidean', leaf_size = 30, p = None): 
    
    dbscan = DBSCAN(eps = eps, \
                    min_samples = min_samples, \
                    leaf_size=leaf_size, \
                    metric = metric, \
                    p = p)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.DBSCAN.value,
        "eps" : eps, 
        "min_samples" : min_samples, 
        "metric" : metric, 
        "leaf_size" : leaf_size, 
        "p" : p, 
    }

    run_clustering(word_vec_model, dbscan, model_details)

# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_dbscan(word_vec_model)