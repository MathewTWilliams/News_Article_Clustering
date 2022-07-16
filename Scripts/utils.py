#Author: Matt Williams
#Version: 07/11/2022

from enum import Enum
import os
import pandas as pd
import numpy as np
import re

# A Simple Python File that contains constant values and methods important to the project. 
# Most of these values are file path related. 


RESULT_WORD_VEC_MOD_KEY = "Vector_Model"
RESULT_CLUSTER_ALGO_KEY = "Clustering"

class Datasets(Enum): 
    TRAIN = "Train"
    TEST = "Test"

class WordVectorModels(Enum): 
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    WORD2VEC = "word2vec"

    @classmethod
    def get_values_as_list(cls): 
        return [model.value for model in WordVectorModels]

class ClusteringAlgorithms(Enum): 
    KMEANS = "K-Means"
    AFF_PROP = "Affinity Propagation"
    M_SHIFT = "Mean-shift"
    SPECT = "Spectral"
    AGGLO = "Agglomerative"
    DBSCAN = "DBSCAN"
    OPTICS = "OPTICS"
    BIRCH = "BIRCH"
    BI_KMEANS = "Bisecting K-Means"
    BATCH_KMEANS = "Mini Batch K-Means"
    SPECT_BI = "Spectral Bi-Clustering"
    SPECT_CO = "Spectral Co-Clustering"


    @classmethod
    def get_values_as_list(cls): 
        return [model.value for model in ClusteringAlgorithms if model not in \
                [ClusteringAlgorithms.SPECT_BI, ClusteringAlgorithms.SPECT_BI, \
                ClusteringAlgorithms.BIRCH, ClusteringAlgorithms.DBSCAN]]

class Categories(Enum): 
    MEDIA = "MEDIA"
    WEIRD = "WEIRD NEWS"
    GREEN =  "GREEN"
    POST =  "WORLDPOST"
    RELIGION = "RELIGION"
    STYLE =  "STYLE"
    SCIENCE =  "SCIENCE"
    NEWS =  "WORLD NEWS"
    TASTE = "TASTE"
    TECH =  "TECH"

    @classmethod
    def get_values_as_list(cls): 
        return [cat.value for cat in Categories]


#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 
CWD_PATH = os.path.abspath(os.getcwd())


def _create_file_path(folder, file_name, subfolder = None):
    """Given the file path to specified folder, a file_name, and an optional subfolder name, make the folders
    if the folders don't exist and return the full filepath."""
    if not os.path.exists(folder): 
        os.mkdir(folder)

    if subfolder != None:
        res_dir = os.path.join(folder, subfolder)
        if not os.path.exists(res_dir): 
            os.mkdir(res_dir)
        
        return os.path.join(res_dir, file_name)

    return os.path.join(folder, file_name)
        
#File Path to the directory where the article vectors are to be stored.
ARTICLE_VECS_DIR_PATH = os.path.join(CWD_PATH, "Article_Vectors")

def get_article_vecs_path(subfolder, name):
    '''Given subfolder(Train or Test) and the name of a file, return the full file path for that file to be stored at'''

    return _create_file_path(ARTICLE_VECS_DIR_PATH, name, subfolder=subfolder)

#File Path to the directory where classification or clustering results are to be stored.
RESULTS_DIR_PATH = os.path.join(CWD_PATH, "Results")

def _find_json_file_name_number(folder):
    '''Given a file path to a folder containing .json files, find the highest number associated with a filename in that
    folder, then increment that number by 1 and return it '''

    if not os.path.exists(folder): 
        return 1

    highest_num = 0
    num_pattern = re.compile(r"[0-9]+\.json")
    for file in os.listdir(folder):
        match = num_pattern.search(file).group()
        num = int(match.split(".")[0])
        if num > highest_num: 
            highest_num = num
    
    return highest_num + 1

def make_result_path(subfolder, vec_model): 
    '''Given the name of a subfolder(Name of the Classification/Clustering algorithm) and the name
        of the vector model, generate a name for the file and return the full file path for the file to be stored at.'''

    updated_path = os.path.join(RESULTS_DIR_PATH, subfolder)
    num = _find_json_file_name_number(updated_path)
    file_name = vec_model + "_results_{}.json".format(num)
    return _create_file_path(RESULTS_DIR_PATH, file_name, subfolder=subfolder)

def get_result_path(subfolder, file_name): 
    '''Get the full file path for the result with the given filename and the given subfolder.'''

    return _create_file_path(RESULTS_DIR_PATH, file_name, subfolder=subfolder)

def convert_categories_to_numbers(labels):
    '''given a pandas series that contains the different categories, convert those 
    categories into integers. Integers represent location in CATEGORIES list.'''

    if isinstance(labels, pd.Series):
        for i,category in enumerate(Categories.get_values_as_list()):
            labels = labels.replace(category, i)
    
    elif isinstance(labels, np.ndarray):
        for i,category in enumerate(Categories.get_values_as_list()): 
            labels[labels == category] = i

    return labels

RESULT_VISUALS_DIR_PATH = os.path.join(CWD_PATH, "Visuals")
CLUSTERING_VISUALS_DIR_PATH = os.path.join(RESULT_VISUALS_DIR_PATH, "Clusterings")



def get_clustering_visual_file_path(file_name): 
    return _create_file_path(RESULT_VISUALS_DIR_PATH, file_name, subfolder=CLUSTERING_VISUALS_DIR_PATH)