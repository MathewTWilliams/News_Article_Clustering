# Author: Matt Williams
# Version: 7/14/2022


from get_article_vectors import get_combined_train_test_info
from clustering_metrics import calculate_clustering_metrics
from utils import convert_categories_to_numbers

def run_clustering(word_vec_model, clustering, clustering_details): 
    
    data, labels = get_combined_train_test_info(word_vec_model)
    labels = convert_categories_to_numbers(labels)

    predictions = clustering.fit_predict(data)

    calculate_clustering_metrics(labels, predictions, data, clustering_details)