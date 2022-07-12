#Author: Matt Williams
#Version: 07/11/2022

from sklearn.cluster import KMeans
from get_article_vectors import get_combined_train_test_info
from utils import Categories, convert_categories_to_numbers, ClusteringAlgorithms, WordVectorModels
from visualize_article_vecs import visualize_article_vecs
from clustering_metrics import calculate_clustering_metrics
import pandas as pd


def run_k_means(vec_model_name, n_init = 10, tol = 1e-4): 
    data, labels = get_combined_train_test_info(vec_model_name)
    labels = convert_categories_to_numbers(labels)

    k_means = KMeans(n_clusters= len(Categories.get_values_as_list()), \
        algorithm = "full", n_init = n_init, tol = tol)

    predictions = k_means.fit_predict(data)
    model_details = {
        "Vector_Model" : vec_model_name, 
        "Clustering" : ClusteringAlgorithms.KMEANS.value, 
        "n_init" : n_init, 
        "tol" : tol, 
    }

    calculate_clustering_metrics(labels, predictions, data, model_details)


if __name__ == "__main__": 
    for word_vec_model in WordVectorModels.get_values_as_list(): 
        run_k_means(word_vec_model)
