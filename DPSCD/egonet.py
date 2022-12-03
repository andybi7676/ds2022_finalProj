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
        with open(egonets_fp, 'r') as er:
            lines = er.readlines()
            for line in tqdm(lines):
                contents = line.strip('\r\n ').split(':')
                user_id = int(contents[0])
                friends_ids = [int(fr) for fr in contents[1].strip('\r\n ').split()]
                user = self.users[user_id]
                for fr_id in friends_ids:
                    user.add_neighbor_to_dict(fr_id)
                user.stabilized()
                self.egonet_user_ids.append(user)
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
                    self.iv_pairs.append((id_i, id_j))
    
    def log(self, log_str: str):
        print(log_str)

    def simN(self, user_i, user_v):
        salton_score = len(user_i & user_v) / math.sqrt(len(user_i) * len(user_v))
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
        dis_i_v = 1 / (self.alpha*simP + (1-self.alpha)*simN)
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
            v_filter = users_rhos_npy > rho_i # implicitly filter out id_i
            v_candidates = user_ids_npy[v_filter]
            if id_i == max_user_id:
                self.delta[id_i] = max(self.get_distance[(id_i, id_v)] for id_v in v_candidates)
            else:
                self.delta[id_i] = min(self.get_distance[(id_i, id_v)] for id_v in v_candidates)
    
        

    
