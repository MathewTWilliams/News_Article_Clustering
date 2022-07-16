#Author: Matt Williams
#Version: 7/11/2022

from sklearn.metrics import silhouette_score, rand_score, \
    homogeneity_completeness_v_measure, normalized_mutual_info_score, \
    adjusted_mutual_info_score, mutual_info_score

from utils import make_result_path
from save_load_json import save_json
import matplotlib.pyplot as plt
from visualize_article_vecs import visualize_article_vecs
import pandas as pd

def calculate_clustering_metrics(labels, predictions, features, model_details): 
    
    model_details["Silhouette Coef"] = silhouette_score(features, labels)
    model_details["Rand_Score"] = rand_score(labels, predictions)
    h,c,v = homogeneity_completeness_v_measure(labels, predictions)
    model_details["Homogeneity"] = h
    model_details["Completeness"] = c
    model_details["V-Measures"] = v
    model_details["Mutual Information"] = mutual_info_score(labels, predictions)
    model_details["Normalized M.I."] = normalized_mutual_info_score(labels, predictions)
    model_details["Adjusted M.I."] = adjusted_mutual_info_score(labels, predictions)

    vec_model_name = model_details['Vector_Model']
    clustering = model_details["Clustering"]


    file_path = make_result_path(clustering, vec_model_name)
    save_json(model_details, file_path)

    visualize_article_vecs(vec_model_name, 2, clustering = clustering, labels = pd.Series(predictions))
    visualize_article_vecs(vec_model_name, 3, clustering = clustering, labels = pd.Series(predictions))





