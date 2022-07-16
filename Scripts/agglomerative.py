# Author: Matt Williams
# Version: 07/15/2022



from os import link
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    Categories, ClusteringAlgorithms, WordVectorModels
from sklearn.cluster import AgglomerativeClustering
from run_clustering import run_clustering

def run_agglomerative(word_vec_model, affinity = 'euclidean', linkage = "ward"):

    agglo = AgglomerativeClustering(n_clusters=len(Categories.get_values_as_list()), \
                                    affinity=affinity, \
                                    linkage = linkage )

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY: ClusteringAlgorithms.AGGLO.value,
        "affinity" : affinity, 
        "linkage" : linkage 
    }

    run_clustering(word_vec_model, agglo, model_details)


# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_agglomerative(word_vec_model)