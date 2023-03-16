# Author: Matt Williams
# Version: 07/16/2022

from os import link
from affinity_propagation import run_affinity_propagation
from agglomerative import run_agglomerative
from batch_kmeans import run_batch_kmeans
from birch import run_birch
from dbscan import run_dbscan
from k_means import run_k_means
from mean_shift import run_mean_shift
from optics import run_optics
from spectral_bi import run_bi_spectral
from spectral_co import run_co_spectral
from spectral import run_spectral
from bisect_kmeans import run_bisect_kmeans

from utils import WordVectorModels


if __name__ == "__main__":
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        #run_affinity_propagation(word_vec_model)
        run_agglomerative(word_vec_model, linkage = "ward")
        run_batch_kmeans(word_vec_model)
        #run_birch(word_vec_model)
        #run_dbscan(word_vec_model)
        run_k_means(word_vec_model)
        #run_mean_shift(word_vec_model)
        #run_optics(word_vec_model)
        #run_bi_spectral(word_vec_model)
        #run_co_spectral(word_vec_model)
        run_spectral(word_vec_model)
        run_bisect_kmeans(word_vec_model)