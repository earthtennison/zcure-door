import pickle

class PickleMan():
    def __init__(self, pkl_path=None):
        if pkl_path is not None:
            f = open(pkl_path,'rb')
            self.data_dict = pickle.load(f)
        else:
            self.data_dict = {}

    def get_data(self):
        """
        read registered feature that store in pickle file

        :param pkl_path: path to pickle file
        :return: list of person name and list of feature with corresponding index
        """

        person_list = list(self.data_dict.keys())
        person_list.sort()
        features = []
        for person in person_list:
            features.append(self.data_dict[person])
        
        return person_list, features

    def delete(self, person):
        person_list = list(self.data_dict.keys())
        person_list.sort()

        if person in person_list:
            self.data_dict.pop(person)
            return True
        else:
            return False

    def add(self, person, feature):
        person_list = list(self.data_dict.keys())
        person_list.sort()
        if not person in person_list:
            self.data_dict[person] = feature
            return True
        else:
            return False

    def get_person_num(self):
        return len(self.data_dict.keys())

    def combine(self, other_pkl_path):
        f = open(other_pkl_path,'rb')
        other_data_dict = pickle.load(f)
        person_list = list(other_data_dict.keys())
        person_list.sort()
        for person in person_list:
            if person in list(self.data_dict.keys()):
                continue
            self.data_dict[person] = other_data_dict[person]
            
    def save_data(self, save_path):
        with open(save_path,"wb") as f:
            pickle.dump(self.data_dict, f)
        print(f"pickle saved to {save_path}")
