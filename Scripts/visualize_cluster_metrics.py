#Author: Matt Williams
#Version: 7/20/2022


import matplotlib.pyplot as plt
from save_load_json import load_test_result
from utils import ClusteringAlgorithms, WordVectorModels




def visualize_clustering_metric(clustering, word_vec_model, metric, value_tuples): 
    pass



if __name__ == "__main__": 
    metrics = ["Silhouette Coef", "Rand_Score", "Homogeneity", "Completeness", "V-Measures", \
        "Mutual Infomration", "Normalized M.I.", "Adjusted M.I."]

    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = {} 
        for word_vec_model in WordVectorModels.get_values_as_list():
            metric_dict[metric][word_vec_model] = []
            for clustering in ClusteringAlgorithms.get_values_as_list():
                results = load_test_result(clustering, word_vec_model)
                metric_dict[metric][word_vec_model].append((clustering, results[metric])) 
            
    for metric in metrics: 
        for word_vec_model in WordVectorModels.get_values_as_list(): 
            for clustering in ClusteringAlgorithms.get_values_as_list(): 
                visualize_clustering_metric(clustering, word_vec_model, metric, metric_dict[metric][word_vec_model])