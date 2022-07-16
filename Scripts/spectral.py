# Author: Matt Williams
# Version: 07/13/2022


from sklearn.cluster import SpectralClustering
from utils import ClusteringAlgorithms, Categories, \
    RESULT_WORD_VEC_MOD_KEY, RESULT_CLUSTER_ALGO_KEY, WordVectorModels
from run_clustering import run_clustering

def run_spectral(word_vec_model, eigen_solver = "arpack", gamma = 1.0, \
    affinity = 'rbf',  assign_labels = 'kmeans', n_init = 10,  \
    n_neighbors = 10, eigen_tol = 0.0, degree = 3, coef0 = 1):
    
  
    spectral = SpectralClustering(n_clusters=len(Categories.get_values_as_list()), 
                                eigen_solver=eigen_solver, 
                                affinity=affinity, 
                                assign_labels=assign_labels, 
                                gamma = gamma, 
                                n_init = n_init, 
                                n_neighbors=n_neighbors, 
                                eigen_tol=eigen_tol, 
                                degree=degree, 
                                coef0 = coef0)


    model_details = {
        RESULT_WORD_VEC_MOD_KEY : word_vec_model, 
        RESULT_CLUSTER_ALGO_KEY : ClusteringAlgorithms.SPECT.value, 
        "eigen_solver" : eigen_solver, 
        "gamma" : gamma, 
        "affinity" : affinity, 
        "assign_labels" : assign_labels, 
        "n_init" : n_init, 
        "n_neighbors" : n_neighbors, 
        "eigen_tol" : eigen_tol, 
        "degree" : degree,
        "coef0" : coef0
    }

    run_clustering(word_vec_model, spectral, model_details)


# used for testing hyper-parameters
if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_spectral(word_vec_model)