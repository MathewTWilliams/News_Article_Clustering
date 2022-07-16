# Author: Matt Williams
# Version: 07/15/2022


from sklearn.cluster import SpectralCoclustering
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, Categories, WordVectorModels


def run_co_spectral(word_vec_model, n_init = 10, n_svd_vecs = None): 
    
    co_spectral = SpectralCoclustering(n_clusters = len(Categories.get_values_as_list()), \
                    svd_method = 'arpack', n_svd_vecs = n_svd_vecs, n_init = n_init)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.SPECT_CO.value,
        'svd_method' : 'arpack',
        'n_init' : n_init,
        'n_svd_vecs' : n_svd_vecs, 
    }

    run_clustering(word_vec_model, co_spectral, model_details)


# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_co_spectral(word_vec_model)