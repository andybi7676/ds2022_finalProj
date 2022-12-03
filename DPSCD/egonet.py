import os
import math
import pickle as pkl
import os.path as osp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from feature import *
from user import *

class Egonet():

    users = None
    @classmethod
    def register_users(cls, users):
        cls.users = users
    
    @classmethod
    def _init_value(cls):
        return 0.0

    def __init__(self, egonet_fpath: str, alpha=0.3, sigma=0.25, d_c=0.1):
        assert self.users != None, f"please register users first!"
        self.alpha = 0.3
        self.sigma = 0.25
        self.d_c = 0.1
        self.egonet_user_ids = []
        with open(egonet_fpath, 'r') as er:
            lines = er.readlines()
            for line in tqdm(lines):
                contents = line.strip('\r\n ').split(':')
                user_id = int(contents[0])
                friends_ids = [int(fr) for fr in contents[1].strip('\r\n ').split()]
                user = self.users[user_id]
                for fr_id in friends_ids:
                    user.add_neighbor_to_dict(self.users[fr_id])
                user.stabilize()
                self.egonet_user_ids.append(user_id)
        self.egonet_user_ids.sort()
        self.size = len(self.egonet_user_ids)
        self.create_iv_pairs()
        self.calculate_distance()
        self.calculate_local_density()
        self.calculate_delta_distance()
    
    def create_iv_pairs(self):
        self.valid_iv_pairs = []
        for id_i in self.egonet_user_ids:
            for id_j in self.egonet_user_ids:
                if id_i != id_j:
                    self.valid_iv_pairs.append((id_i, id_j))
        print((1, 2) in self.valid_iv_pairs)
    
    def log(self, log_str: str):
        print(log_str)

    def simN(self, user_i, user_v):
        salton_score = len(user_i & user_v) / max(1, math.sqrt(len(user_i) * len(user_v)))
        return salton_score
    
    def simP(self, user_i, user_v):
        score = user_i.feature * user_v.feature
        return score
    
    def get_distance(self, id_i: int, id_v: int):
        return self.D[(id_i, id_v)]
    
    def calculate_distance(self):
        self.D = {}
        for id_i, id_v in self.valid_iv_pairs:
            user_i = self.users[id_i]
            user_v = self.users[id_v]
            simP = self.simP(user_i, user_v)
            simN = self.simN(user_i, user_v)
            dis_i_v = 1 / max((self.alpha*simP + (1-self.alpha)*simN), 1e-5)
            self.D[(id_i, id_v)] = dis_i_v
    
    def calculate_local_density(self):
        # store local density in 'user' also
        self.Rho = defaultdict(self._init_value)
        denominator = 2 * self.sigma**2
        def normalize(dis_i_v):
            reg_dis = abs(dis_i_v - self.d_c)
            return - reg_dis / denominator
        
        for id_i, id_v in self.valid_iv_pairs:
            add_val = math.exp(normalize(self.get_distance(id_i, id_v)))
            self.Rho[id_i] += add_val
        for id_i in self.egonet_user_ids:
            self.users[id_i].local_density = self.Rho[id_i] # Might be redundant
    
    def calculate_delta_distance(self):
        self.delta = defaultdict(self._init_value)
        user_ids_npy = np.array(self.egonet_user_ids)
        user_rhos_npy = np.array([self.Rho[user_id] for user_id in self.egonet_user_ids])
        max_user_id = user_ids_npy[np.argmax(user_rhos_npy)]
        for id_i, rho_i in zip(user_ids_npy, user_rhos_npy):
            if id_i == max_user_id:
                v_filter = user_ids_npy != id_i
                metric = "max"
            else:
                v_filter = user_rhos_npy > rho_i # implicitly filter out id_i
                metric = "min"
            v_candidates = user_ids_npy[v_filter]
            self.delta[id_i] = eval(metric)([self.get_distance(id_i, id_v) for id_v in v_candidates])
    

if __name__ == "__main__":
    featurelist_fp = '/home/andybi7676/Desktop/ds2022_finalProj/featureList.txt'
    feature_fp = '/home/andybi7676/Desktop/ds2022_finalProj/features.txt'
    cache_fp = '.cache/users.pkl'
    
    users = Users(feature_fp, featurelist_fp) 
    Egonet.register_users(users)
    ego_0_fpath = '/home/andybi7676/Desktop/ds2022_finalProj/egonets/0.egonet'
    ego_0 = Egonet(ego_0_fpath)  
    print(ego_0.Rho)
    print(ego_0.delta) 

    
