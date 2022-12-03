import os
import pickle as pkl
import os.path as osp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from feature import *

class Users():
    def __init__(self, feature_fp: str, featurelist_fp: str, cache_fp=".cache/users.pkl"):
        if osp.exists(cache_fp):
            self.load_users(cache_fp)
            return
        assert osp.exists(feature_fp), f"invalid feature file path: {feature_fp}"
        assert osp.exists(featurelist_fp), f"invalid feature-list file path: {featurelist_fp}"
        self.featurelist = FeatureList(featurelist_fp)
        Feature.register_featurelist(self.featurelist)
        self.users = defaultdict(self._doesnot_exists)
        with open(feature_fp, 'r') as fr:
            lines = fr.readlines()
            for line in tqdm(lines, total=len(lines)):
                contents = line.strip('\r\n ').split(' ')
                user_id, user_feature = int(contents[0]), Feature(contents[1:])
                new_user = User(user_id, user_feature)
                self.users[user_id] = new_user
    
    @classmethod
    def _doesnot_exists(self):
        return None
    
    def load_users(self, cache_fp):
        with open(cache_fp, 'rb') as c_frb:
            self.users = pkl.load(c_frb)
    
    def dump_users(self, cache_fp=".cache/users.pkl"):
        try:
            with open(cache_fp, 'wb') as c_fwb:
                pkl.dump(self.users, c_fwb)
        except Exception as e:
            print(e)
            os.remove(cache_fp)
    
    def __getitem__(self, id: int):
        return self.users[id]
    
    def __len__(self):
        return len(users)

class User():

    @classmethod
    def _doesnot_exists(cls):
        return False
    
    def __init__(self, user_id: int, user_feature: Feature):
        self.id = user_id
        self.feature = user_feature
        self.neighbors_dict = defaultdict(self._doesnot_exists)
        self.neighbors = set()
        self.stabilized = False
        self.local_density = 0.0
    
    def __repr__(self):
        repr_str = "" 
        repr_str += f"User_id: {self.id}, features_count: {(self.feature.feature_npy!=-1).sum()}"
        repr_str += f", neighbors_count: {len(self.neighbors)}, stablilized={self.stabilized}"
        return repr_str
    
    def add_neighbor_to_dict(self, neighbor):
        self.neighbors_dict[neighbor.id] = True

    def stabilize(self): # once an egonetwork has settled down, call this function to stabilize the user's neighbors
        neighbor_list = [nb_id for nb_id, _ in self.neighbors_dict.items()]
        self.neighbors = set(neighbor_list)
        self.stabilized = True
    
    def __len__(self):
        assert self.stabilized, f"User id={self.id} are not stabilized yet."
        return len(self.neighbors)
    
    def __and__(self, other):
        assert self.stabilized == True, f"User id={self.id} are not stabilized yet."
        assert other.stabilized == True, f"User id={other.id} are not stabilized yet."
        return self.neighbors & other.neighbors

    

if __name__ == "__main__":
    featurelist_fp = '/home/andybi7676/Desktop/ds2022_finalProj/featureList.txt'
    feature_fp = '/home/andybi7676/Desktop/ds2022_finalProj/features.txt'
    cache_fp = '.cache/users.pkl'
    
    users = Users(feature_fp, featurelist_fp)
    if not osp.exists(cache_fp):
        users.dump_users()
    
    a = users[0]
    b = users[1]
    print(a)
    print(b)
    print(a.feature * b.feature)
    a.add_neighbor_to_dict(b)
    a.stabilize()

    print(a.neighbors)
    print(a.neighbors_dict)
    