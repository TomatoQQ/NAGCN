import csv
import torch
import random
import numpy as np
import pandas as pd
import torch
from visdom import Visdom
import time

def train(model, train_data, optimizer):
    model.train()
    for epoch in range(0, 100):
        model.zero_grad()
        score,x,y = model(train_data)
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss(score, train_data['c_d'])
        loss.backward()
        optimizer.step()
        print(loss.item())
    score = score.detach().cpu().numpy()
    return model

#读取矩阵里的边权重不为零的边
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

def dataset():
    dataset = dict()
    with open('datasets/circRNADisease/circRNADisease_associations.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
    dataset['c_d'] = torch.Tensor(cd_data)
    zero_index = []
    one_index = []
    cd_pairs = []
    for i in range(dataset['c_d'].size(0)):
        for j in range(dataset['c_d'].size(1)):
            if dataset['c_d'][i][j] < 1:
                zero_index.append([i, j, 0])
            if dataset['c_d'][i][j] >= 1:
                one_index.append([i, j, 1])
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index
    with open('similarity/circRNADisease/disSimilarity.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        dd_data = []
        dd_data += [[float(i) for i in row] for row in reader]
    dd_matrix = torch.Tensor(dd_data)
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}
    with open('similarity/circRNADisease/circSimilarity.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cc_data = []
        cc_data += [[float(i) for i in row] for row in reader]
    cc_matrix = torch.Tensor(cc_data)
    cc_edge_index = get_edge_index(cc_matrix)
    dataset['cc'] = {'data_matrix': cc_matrix,
                     'edges': cc_edge_index}
    return dataset, cd_pairs

# 特征学习
def feature_representation(model, dataset):
    model(dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model = train(model, dataset, optimizer)
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(dataset)
    cir_fea = cir_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, cir_fea, dis_fea

def new_dataset(cir_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []
    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])
    nega_list = []
    for i in range(len(unknown_pairs)):
        a = unknown_pairs[i][0]
        b = unknown_pairs[i][1]
        nega = cir_fea[a, :].tolist() + dis_fea[b, :].tolist() + [0, 1]
        nega_list.append(nega)
    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0], :].tolist() + dis_fea[known_pairs[j][1], :].tolist() + [1, 0]
        posi_list.append(posi)
    samples = posi_list + nega_list
    random.shuffle(samples)
    samples = np.array(samples)
    return samples

#取训练集和测试集
def C_Dmatix(cd_pairs, trainindex, testindex):
    c_dmatix = np.zeros((1405, 270))
    print(trainindex)
    for i in trainindex:
        if cd_pairs[i][2] == 1:
            c_dmatix[cd_pairs[i][0]][cd_pairs[i][1]] = 1
    dataset = dict()
    cd_data = []
    cd_data += [[float(i) for i in row] for row in c_dmatix]
    cd_data = torch.Tensor(cd_data)
    dataset['c_d'] = cd_data
    train_cd_pairs = []
    test_cd_pairs = []
    for m in trainindex:
        train_cd_pairs.append(cd_pairs[m])
    for n in testindex:
        test_cd_pairs.append(cd_pairs[n])
    return dataset['c_d'], train_cd_pairs, test_cd_pairs