#Author: Matt Williams
#Version: 07/13/2022

from sklearn.cluster import MeanShift
from utils import ClusteringAlgorithms, WordVectorModels, \
    RESULT_WORD_VEC_MOD_KEY, RESULT_CLUSTER_ALGO_KEY
from run_clustering import run_clustering

def run_mean_shift(vec_model_name, max_iter = 300 ): 

    mean_shift = MeanShift(bin_seeding=True, max_iter=max_iter)     

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : vec_model_name, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.M_SHIFT.value,  
        "max_iter" : max_iter, 
        "bin_seeding" : True,
    }

    
    run_clustering(vec_model_name, mean_shift, model_details)

# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_mean_shift(word_vec_model)