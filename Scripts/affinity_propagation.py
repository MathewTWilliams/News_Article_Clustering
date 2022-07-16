# Author: Matt Williams
# Version: 07/15/2022

from sklearn.cluster import AffinityPropagation
from utils import ClusteringAlgorithms, WordVectorModels, \
    RESULT_WORD_VEC_MOD_KEY, RESULT_CLUSTER_ALGO_KEY
from run_clustering import run_clustering



def run_affinity_propagation(vec_model_name, damping = 0.5, max_iter = 200, convergence_iter = 15): 
    aff_prop = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : vec_model_name, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.AFF_PROP.value, 
        "damping" : damping, 
        "max_iter" : max_iter, 
        "convergence_iter" : convergence_iter, 
    }

    run_clustering(vec_model_name, aff_prop, model_details)

# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_affinity_propagation(word_vec_model)