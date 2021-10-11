# -*- coding: utf-8 -*-
"""
@author:xxxxxxx
"""
import torch
import torch.utils
import torch.utils.data
import numpy as np
import copy
import os

import pickle
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.utils import to_undirected,remove_self_loops,remove_isolated_nodes
def onehot_encoder(data):
    data = torch.unsqueeze(data, 1)
    if torch.sum(data==-99)>0:
        tmp = torch.ones_like(data)
        index = (data == -99).nonzero()
        tmp[index] = 0.0
        data[index] = 0.0
        data = torch.cat([tmp, data], dim=1)
        #print("--- %s ---" % str(data.size()))
        return data.to(torch.float)
        
    else:
        #print("--- %s ---" % str(data.size()))
        return data.to(torch.float)
def zodiac_encoder(zodiac):
    zodiac = zodiac.to(torch.int64)
    index = (zodiac == -99).nonzero()
    zodiac[index]= 0
    zodiac = F.one_hot(zodiac)
    return zodiac.to(torch.float)

def gender_encoder(gender):
    gender = gender.to(torch.int64)
    index = (gender == -99).nonzero()
    gender[index]= 0
    gender = F.one_hot(gender)
    return gender.to(torch.float)

def age_style_encoder(age_style):
    age_style = age_style.to(torch.int64)
    index = (age_style == -99).nonzero()
    age_style[index]= 6
    age_style = F.one_hot(age_style)
    return age_style.to(torch.float)

def area_encoder(area):
    area = area.to(torch.int64)
    index = (area == -99).nonzero()
    area[index]= 0
    area = F.one_hot(area)
    return area.to(torch.float)

def city_encoder(city):
    city = city.to(torch.int64)
    index = (city == -99).nonzero()
    city[index]= 0
    city = F.one_hot(city)
    return city.to(torch.float)

def province_encoder(province):
    province = province.to(torch.int64)
    index = (province == -99).nonzero()
    province[index]= 0
    province = F.one_hot(province)
    return province.to(torch.float)
def cont_norm(data):
    print("--- data: %s ---" % str(data.size()))
    mean, var = torch.var_mean(data, dim = 0, keepdim=True)
    print("--- mean: %s ---" % str(mean))
    print("--- var: %s ---" % str(var))
    data = (data - mean)/torch.sqrt(var)
    print("--- data: %s ---" % str(data.size()))
    return data

class read_motif(object):
    def __init__(self, motif):
        self.motif = motif
    def __iter__(self):
        self.cnt = 0
        if self.motif == 'triangle':
            self.path = './data/triangle'
            self.num = 13
        elif self.motif == 'square':
            self.path = './data/square'
            self.num = 6
        return self
    def __next__(self):
        if self.cnt < self.num :
            motif_path = self.path+'/qglobal_motif_sparse_'+str(self.cnt)+'.npy'
            x = np.load(motif_path, allow_pickle=True)
            self.cnt = self.cnt+1
            return x
        else:
            raise StopIteration

class CVPDataset(Dataset):
    def __init__(self, season,directed,args):
        super(CVPDataset, self).__init__()
        self.batch_size = args.batch_size
        self.directed = directed
        self.season = season
        self.is_weighted = args.edge_weight
        self._load_data()
        

    def _load_data(self):
        relation = np.load(os.path.join('./data', 'relation.npy'), allow_pickle=True)
        motif_tri = read_motif('triangle')
        motif_squ = read_motif('square')
        if self.season == 'q1':
            data = np.load(os.path.join('./data', 'q1_rf.npy'), allow_pickle=True)
        elif self.season == 'q2':
            data = np.load(os.path.join('./data', 'q2_rf.npy'), allow_pickle=True)
        elif self.season == 'q3':
            data = np.load(os.path.join('./data', 'q3_rf.npy'), allow_pickle=True)
        elif self.season == 'q4':
            data = np.load(os.path.join('./data', 'q4_rf.npy'), allow_pickle=True)

        self._construct_graph(data, relation, motif_tri, motif_squ)
    
    def _make_batch(self):
        batch_ids = []
        y_index = self.y_index
        y_index = y_index.reshape(-1)
        num = self.num
        batch_size = self.batch_size
        batch_num = int(num/batch_size)+1
        np.random.shuffle(y_index)
        for i in range(batch_num):
            batch_ids.append(y_index[i*batch_size : min((i+1)*batch_size, num) ])
        self.batch_num = batch_num
        self.batch_ids = batch_ids
        

    def _construct_graph(self, data, relation, motif_tri, motif_squ):
        
        uid = torch.arange(data.shape[0],dtype=torch.float).view(-1, 1)
        
        x = torch.tensor(data[:, 0:-1], dtype=torch.float)
        x = torch.cat([uid, x], dim=1)
        x = self._feature_preprocess(x)
        
        y = torch.tensor(data[:, -1], dtype=torch.float)
        
        edge_index = torch.tensor(relation, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        if self.directed == False:
            edge_index= to_undirected(edge_index)

        graph = Data(x=x, edge_index=edge_index, y=y)
        

        
        y = data[:, -1]
        y_thres = np.mean(y)+1*np.std(y)
        y_index = np.where(y<y_thres)[0]

        self.y_index = y_index
        self.num = y_index.shape[0]
        
        y_index = torch.tensor(y_index, dtype=torch.long)
        graph.y_index=y_index

        for index, motif_adj in enumerate(motif_tri):
            if index in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
                motif_edge = motif_adj[:, 0:2].reshape((-1 ,2))
                edge_index = torch.tensor(motif_edge, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                graph['motif_triangle_'+str(index)]=edge_index
                
                if self.is_weighted == 'weighted':
                    motif_weight = motif_adj[:,2].astype('int').reshape(-1)
                    edge_weight = torch.tensor(motif_weight, dtype=torch.float)
                    edge_weight = edge_weight.t().contiguous()
                    graph['motif_triangle_weight_'+str(index)]=edge_weight
        
        for index, motif_adj in enumerate(motif_squ):
            if index in [0,2,3,4,5]:
                motif_edge = motif_adj[:, 0:2].reshape((-1 ,2))
                edge_index = torch.tensor(motif_edge, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                graph['motif_square_'+str(index)]=edge_index
                
                if self.is_weighted == 'weighted':
                    motif_weight = motif_adj[:,2].astype('int').reshape(-1)
                    edge_weight = torch.tensor(motif_weight, dtype=torch.float)
                    edge_weight = edge_weight.t().contiguous()
                    graph['motif_square_weight_'+str(index)]=edge_weight


        self._make_batch()
        self.trainset=graph
    
    def _feature_preprocess(self, x):

        user_type = F.one_hot(x[:, 2].to(torch.int64)).to(torch.float)
        gender = gender_encoder(x[:, 3])
        zodiac = zodiac_encoder(x[:, 5])
        age_style = age_style_encoder(x[:, 6])
        common_address_area_1y = area_encoder(x[:, 9])
        common_address_city_level_1y = city_encoder(x[:, 10])
        common_address_province_1y = province_encoder(x[:, 11])
        x = torch.cat([
            onehot_encoder(x[:, 1]), user_type, gender, zodiac, age_style,\
            onehot_encoder(x[:, 8]), common_address_area_1y, common_address_city_level_1y, common_address_province_1y, \
            onehot_encoder(x[:,12]), onehot_encoder(x[:, 13]),onehot_encoder(x[:, 14]),\
            #cont_norm(x[:, 15:])], dim = 1)
            x[:, 15:31],x[:,31:]], dim = 1)
        return x

    def __len__(self):
        return self.num
    def __getitem__(self, index):
        graph = self.trainset
        if index == 'all':
            graph.batch_ids = graph.y_index
        else:    
            graph.batch_ids = torch.tensor(self.batch_ids[index],dtype=torch.long)
        return graph




        
