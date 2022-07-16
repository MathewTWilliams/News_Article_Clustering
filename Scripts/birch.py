# Author: Matt Williams
# Version: 07/15/2022


from sklearn.cluster import Birch
from run_clustering import run_clustering
from utils import RESULT_CLUSTER_ALGO_KEY, RESULT_WORD_VEC_MOD_KEY, \
    ClusteringAlgorithms, Categories, WordVectorModels


def run_birch(word_vec_model, threshold = 0.5, braching_factor = 50): 
    
    birch = Birch(n_clusters=len(Categories.get_values_as_list()), \
                threshold=threshold, branching_factor=braching_factor)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.BIRCH.value,
        "threshold" : threshold, 
        "branching_factor" : braching_factor
    }

    run_clustering(word_vec_model, birch, model_details)



# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_birch(word_vec_model)