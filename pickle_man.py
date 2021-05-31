import pickle
import os

def get_data(pkl_path):
    """
    read registered feature that store in pickle file

    :param pkl_path: path to pickle file
    :return: list of person name and list of feature with corresponding index
    """
    with open(pkl_path,'rb') as f:
        data_dict = pickle.load(f)
    
    person_list = list(data_dict.keys())
    person_list.sort()
    features = []
    for person in person_list:
        features.append(data_dict[person])
    
    return person_list, features

def delete(person_name):
    pass

def add(person_name):
    pass

def get_person_num():
    pass