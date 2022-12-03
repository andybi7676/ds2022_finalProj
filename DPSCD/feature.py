import os
import pickle as pkl
import os.path as osp
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class FeatureList():
    def __init__(self, featurelist_fp: str):
        self.feature_list = []
        self.feature_to_idx_dict = defaultdict(self._doesnot_exists) # assign -1 as unknown idx 
        with open(featurelist_fp, 'r') as fr:
            for idx, line in enumerate(fr):
                feature_repr = line.strip('\r\n ')
                self.feature_list.append(feature_repr)
                self.feature_to_idx_dict[feature_repr] = idx
    
    @classmethod
    def _doesnot_exists(self):
        return -1
    
    def __len__(self):
        return len(self.feature_list)
    
    def __repr__(self):
        repr_str = ""
        for idx, feature_repr in enumerate(self.feature_list):
            repr_str += f"{idx}: {feature_repr}\n"
        return repr_str
    
    def __call__(self, input):
        if type(input) == int:
            return self.feature_list[input]
        if type(input) == str:
            return self.feature_to_idx_dict[input]
        else:
            raise NotImplementedError

class Feature():
    featurelist = None

    @classmethod
    def register_featurelist(cls, featurelist: FeatureList):
        cls.featurelist = featurelist
    
    def __init__(self, user_features: str):
        assert self.featurelist != None, f"invalid featurelist, register it first!"
        self.size = len(self.featurelist)
        self.feature_npy = np.full(self.size, -1) 
        for feature in user_features:
            contents = feature.split(';')
            feature_repr = ';'.join(contents[:-1])
            feature_value = int(contents[-1])
            feature_idx = self.featurelist(feature_repr)
            self.feature_npy[feature_idx] = feature_value
    
    def __mul__(self, other):
        res_npy = (self.feature_npy == other.feature_npy) * (self.feature_npy != -1)
        return sum(res_npy) # maybe we can divide with self.size?
    
    def __repr__(self):
        return f"{self.feature_npy}"