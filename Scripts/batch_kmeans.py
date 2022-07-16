# Author: Matt Williams
# Version: 07/15/2022


from mimetypes import init
from sklearn.cluster import MiniBatchKMeans
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, Categories, WordVectorModels


def run_batch_kmeans(word_vec_model, max_iter = 100, batch_size = 1024, \
                    max_no_imporvement = 10, init_size = None, n_init = 3, \
                    reassignment_ratio = 0.01): 
    
    batch_kmeans = MiniBatchKMeans(n_clusters = len(Categories.get_values_as_list()), \
                                    max_iter=max_iter, batch_size=batch_size, \
                                    max_no_improvement=max_no_imporvement, \
                                    init_size=init_size, n_init=n_init, \
                                    reassignment_ratio=reassignment_ratio)          

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.BATCH_KMEANS.value,
        "max_iter" : max_iter, 
        "batch_size" : batch_size, 
        "max_no_improvement" : max_no_imporvement, 
        "init_size" : init_size, 
        "n_init" : n_init, 
        "reassignment_ ratio" : reassignment_ratio
    }

    run_clustering(word_vec_model, batch_kmeans, model_details)


# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_batch_kmeans(word_vec_model)