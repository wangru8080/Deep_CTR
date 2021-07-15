# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

class WideDeep(nn.Module):
    def __init__(self, args):
        super(WideDeep, self).__init__()

        self.args = args

        self.sparse_embedd = nn.ModuleList([
            nn.Embedding(cate_size, args.cate_embedding_size) for cate_size in args.cate_fea_uniques
        ])

        wide_size = args.dense_features_size
        self.wide = nn.Linear(wide_size, 1)

        deep_size = args.dense_features_size + args.sparse_features_size * args.cate_embedding_size
        self.deep = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(deep_size, 64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(64, 32)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(32, 1))
        ]))

        self.weight = nn.Parameter(torch.ones(2), requires_grad=True)

        self.apply(self.init_weights)

    def forward(self, x_sparse, x_dense=None):
        sparse_embed = [embedd(x_sparse[:, i].unsqueeze(1)) for i, embedd in enumerate(self.sparse_embedd)] # list: [batch, 1, embedd_size]
        sparse_embed = torch.cat(sparse_embed, dim=1) # [batch, sparse_features_size, embedd_size]
        sparse_embed = sparse_embed.view(-1, self.args.sparse_features_size * self.args.cate_embedding_size)

        wide_out = self.wide(x_dense)

        deep_in = torch.cat([sparse_embed, x_dense], dim=1)
        deep_out = self.deep(deep_in)

        weight = F.softmax(self.weight, dim=0) # 自动学习权重，并保证权重和为1
        logit = weight[0] * wide_out + weight[1] * deep_out
        prob = torch.sigmoid(logit)
        return logit, prob

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

class Config(object):
    def __init__(self):
        self.cate_embedding_size = 8
        self.dense_features_size = 6
        self.sparse_features_size = 8
        self.cate_fea_uniques = []

        self.num_train_epochs = 5
        self.per_gpu_batch_size = 128
        self.learning_rate = 1e-3
        self.gpuid = '-1'
        self.n_gpu = len(self.gpuid.split(',')) if self.gpuid != '-1' else 0
        self.seed = 2021
        self.num_workers = 0

args = Config()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
torch.manual_seed(args.seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':
    print('read dataset...')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_val = pd.read_csv('data/y_val.csv')
    data = pd.concat([train, test], axis=0)

    dense_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    sparse_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    label_name = 'income_label'

    # 交叉特征
    cross_feature = [['education', 'occupation'], ['native_country', 'occupation']]
    cross_name = ['_'.join(col) for col in cross_feature]
    crossed_columns = {}
    for name, cross in zip(cross_name, cross_feature):
        crossed_columns[name] = cross

    df_cross = pd.DataFrame()
    for k, v in crossed_columns.items():
        data[k] = data[v].astype(str).apply(lambda x: '-'.join(x), axis=1)

    lbc = LabelEncoder()
    print('start label encoder')
    for col in cross_name:
        print('this feature is', col)
        try:
            data[col] = lbc.fit_transform(data[col].apply(int))
        except:
            data[col] = lbc.fit_transform(data[col].astype(str))
    train[cross_name] = data[cross_name][0: len(train)]
    test[cross_name] = data[cross_name][len(train):]

    # sparse_features = sparse_features + cross_name # 加入交叉特征
    cate_fea_uniques = [data[f].nunique() for f in sparse_features] # [9, 16, 7, 15, 6, 5, 2, 42] [225, 481]
    args.cate_fea_uniques = cate_fea_uniques

    args.sparse_features_size = len(sparse_features)
    args.dense_features_size = len(dense_features)

    train_dataset = TensorDataset(
        torch.tensor(train[sparse_features].values, dtype=torch.long),
        torch.tensor(train[dense_features].values, dtype=torch.float),
        torch.tensor(y_train[label_name].values, dtype=torch.float),
    )
    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=None)

    eval_dataset = TensorDataset(
        torch.tensor(test[sparse_features].values, dtype=torch.long),
        torch.tensor(test[dense_features].values, dtype=torch.float),
        torch.tensor(y_val[label_name].values, dtype=torch.float),
    )
    eval_dataloader = DataLoader(eval_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=None)

    model = WideDeep(args)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    elif args.n_gpu == 1:
        model = model.cuda()

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        with tqdm(iterable=train_dataloader, ascii=True) as t:
            t.set_description('Epoch %d' % (epoch + 1))
            loss_list = []
            acc_list = []
            for step, data in enumerate(train_dataloader):
                sparse, dense, label = data
                size = len(sparse)
                if args.n_gpu > 0:
                    sparse = sparse.cuda()
                    dense = dense.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                logits, probs = model(sparse, dense)

                pred = (probs.view(-1) >= 0.5).float()
                loss = loss_function(logits.view(-1), label)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()

                # 实时显示
                loss_list.append(loss.item())
                avg_loss = sum(loss_list) / len(loss_list)

                acc = torch.eq(pred, label).sum().float().item() / size
                acc_list.append(acc)
                avg_acc = sum(acc_list) / len(acc_list)

                t.set_postfix(loss='%.6f' % avg_loss, acc='%.6f' % avg_acc)
                t.update()


    model.eval()
    y_pred = []
    y_logits = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for step, eval_data in enumerate(tqdm(eval_dataloader)):
            eval_sparse, eval_dense, eval_label = eval_data
            size = len(sparse)
            if args.n_gpu > 0:
                sparse = sparse.cuda()
                dense = dense.cuda()
                label = label.cuda()
            logits, probs = model(eval_sparse, eval_dense)
            pred = (probs.view(-1) >= 0.5).float()
            y_pred.extend(pred.cpu().numpy())
            y_logits.extend(probs.view(-1).cpu().numpy())
            y_true.extend(eval_label.cpu().numpy())

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_logits)

    print('precision=%.5f' % precision)
    print('recall=%.5f' % recall)
    print('f1=%.5f' % f1)
    print('acc=%.5f' % acc)
    print('auc=%.5f' % auc)

