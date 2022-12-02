import os
import pickle as pkl
import os.path as osp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from feature import *

class Users():
    def __init__(self, cache_fp="./cache/users.pkl"):
        assert osp.exists(cache_fp)
        with open(cache_fp, 'rb') as c_frb:
            self.users = pkl.load(c_frb)
    
    def __init__(self, feature_fp: str, featurelist_fp: str):
        assert osp.exists(feature_fp), f"invalid feature file path: {feature_fp}"
        assert osp.exists(featurelist_fp), f"invalid feature-list file path: {featurelist_fp}"
        self.featurelist = FeatureList(featurelist_fp)
        Feature.register_featurelist(self.featurelist)
        self.users = defaultdict(lambda: None)
        with open(feature_fp, 'r') as fr:
            lines = fr.readlines()
            for line in tqdm(lines[:10]):
                contents = line.strip('\r\n ').split(' ')
                user_id, user_feature = int(contents[0]), Feature(contents[1:])
                new_user = User(user_id, user_feature)
                self.users[user_id] = new_user
    
    def __getitem__(self, id: int):
        return self.users[id]
    
    def __len__(self):
        return len(users)

class User():
    def __init__(self, user_id: int, user_feature: Feature):
        self.id = user_id
        self.feature = user_feature
    
    def __repr__(self):
        return f"User_id: {self.id}, features: {self.feature}"
    

if __name__ == "__main__":
    featurelist_fp = '/home/andybi7676/Desktop/ds/finalProj/featureList.txt'
    feature_fp = '/home/andybi7676/Desktop/ds/finalProj/features.txt'
    # featurelist = FeatureList('/home/andybi7676/Desktop/ds/finalProj/featureList.txt')
    # for i in range(len(featurelist)):
    #     assert i == featurelist(featurelist(i))
    #     print(f"{featurelist(featurelist(i))}: {featurelist(i)}" )
    # Feature.register_featurelist(featurelist)
    
    users = Users(feature_fp, featurelist_fp)
    print(users[0])
    print(users[1])
    print(users[0].feature * users[1].feature)