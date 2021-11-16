import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


class GNN(nn.Module):
    def __init__(self, dims_item, dims_user, L, T, step, device):
        super(GNN, self).__init__()
        self.L = L
        self.T = T
        self.step = step
        self.hidden_size = dims_item
        self.input_size = dims_item * 2
        self.MM = 20

        self.W1 = Parameter(torch.Tensor(
            self.input_size, self.hidden_size)).to(device)
        self.W1 = torch.nn.init.xavier_uniform_(self.W1)

        # no b attention
        self.W2 = Parameter(torch.Tensor(
            self.hidden_size, self.hidden_size)).to(device)
        self.W2 = torch.nn.init.xavier_uniform_(self.W2)

        self.AB1 = Parameter(torch.Tensor(
            self.L-self.T, self.L-self.T)).to(device)
        self.AB1 = torch.nn.init.xavier_uniform_(self.AB1)
        # no b attention
        self.AB2 = Parameter(torch.Tensor(
            (self.L-self.T)//2, (self.L-self.T)//2)).to(device)
        self.AB2 = torch.nn.init.xavier_uniform_(self.AB2)

        # no b attention
        self.Ada1 = Parameter(torch.Tensor(
            self.input_size, self.hidden_size)).to(device)
        self.Ada1 = torch.nn.init.xavier_uniform_(self.Ada1)
        # no b attention
        self.Ada2 = Parameter(torch.Tensor(
            self.hidden_size, self.hidden_size)).to(device)
        self.Ada2 = torch.nn.init.xavier_uniform_(self.Ada2)

    def GNNCell(self, A, hidden, for_pred=False):
        input_in1 = torch.matmul(A, hidden)
        input_in_item1 = input_in1
        # input_in_item1 = torch.cat((input_in1, hidden), dim=2)

        # no b have item
        item_hidden1 = torch.matmul(input_in_item1, self.W2)
        item_embs1 = item_hidden1

        if for_pred:
            B = torch.tanh(self.AB2)
        else:
            B = torch.tanh(self.AB1)

        input_in3 = torch.matmul(B, hidden)
        # print(B)
        input_in_item3 = input_in3
        # input_in_item3 = torch.cat((input_in3, hidden), dim=2)
        item_hidden3 = torch.matmul(input_in_item3, self.Ada2)
        item_embs3 = item_hidden3

        item_embs = torch.tanh(item_embs1) * torch.tanh(item_embs3)

        return item_embs

    def forward(self, A, hidden, for_pred):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, for_pred)
        return hidden


class MAGNN(nn.Module):
    def __init__(self, num_users, num_items, model_args, device):
        super(MAGNN, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        T = self.args.T
        dims_item = self.args.d1
        dims_user = self.args.d2
        step = self.args.step
        # dims = self.args.d
        heads = self.args.h
        units = self.args.m

        # add gnn
        self.gnn = GNN(dims_item, dims_user, L, T, step, device).to(device)

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims_user).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims_item).to(device)

        self.W2 = nn.Embedding(num_items, dims_item, padding_idx=0).to(device)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0).to(device)

        # weight initialization
        self.user_embeddings.weight.data.normal_(
            0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(
            0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.attention_item1 = nn.Linear(dims_item, dims_item).to(device)
        # self.attention_item1 = nn.Linear(dims_item, dims_item*2).to(device)
        self.attention_item2 = nn.Linear(dims_item, heads).to(device)
        self.attention_item3 = nn.Linear(heads, 1).to(device)

        self.user_com = Parameter(torch.Tensor(
            dims_item + dims_user, dims_item)).to(device)
        self.user_com = torch.nn.init.xavier_uniform_(self.user_com)

    def forward(self, item_seq, user_ids, items_to_predict, A, for_pred=False):
        item_embs = self.item_embeddings(item_seq)  # [4096,5,128]
        # item_embs = self.gnn(A, item_embs)
        user_emb = self.user_embeddings(user_ids)  # [4096,128]

        # short_embs = item_embs
        # attention_embs = torch.mean(short_embs, dim=1)

        if for_pred:
            short_embs = self.gnn(A, item_embs, True)  # [4096,5,128]
        else:
            short_embs = self.gnn(A, item_embs, False)  # [4096,5,128]
        # attention_embs = torch.mean(short_embs, dim=1)

        attention_matrix1 = torch.tanh(
            self.attention_item1(short_embs))  # [4095,5,128]
        # attention_matrix1 = self.attention_item1(short_embs)  # [4095,5,128]
        attention_matrix2 = self.attention_item2(
            attention_matrix1)  # [4096,5,20]
        attention_matrix = torch.softmax(
            attention_matrix2, dim=2)  # [4096,4,20]

        # # no adaptivate attention
        matrix_z = torch.bmm(short_embs.permute(0, 2, 1),
                             attention_matrix)  # [4096,128,20]
        attention_embs = torch.mean(torch.tanh(matrix_z), dim=2)  # [4096,128]

        # # no user embedding
        # fusion_embs = attention_embs

        # add user embedding
        fusion_embs = torch.cat((attention_embs, user_emb), dim=1)
        fusion_embs = torch.matmul(fusion_embs, self.user_com)

        # train [4096, 6, 128] item_to_predict [4096, 6] 6 = 2 * T
        # evaluation [35119, 128] item_to_pre [35119]
        w2 = self.W2(items_to_predict)
        # train [4096, 6, 1] evaluation [35119, 1]
        b2 = self.b2(items_to_predict)

        if for_pred:
            w2 = w2.squeeze()  # [35119, 128]
            b2 = b2.squeeze()  # [35119]

            # union-level
            res = fusion_embs.mm(w2.t()) + b2  # [4096, 35119]

            # item-item product
            rel_score = torch.matmul(
                item_embs, w2.t().unsqueeze(0))  # [4096, 5, 35119]
            # [4096, 35119] # sum More better
            rel_score = torch.sum(rel_score, dim=1)

            res += rel_score
        else:
            # union-level [4096, 6]
            res = torch.baddbmm(b2, w2, fusion_embs.unsqueeze(2)).squeeze()

            # item-item product
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))  # [4096, 5, 6]
            rel_score = torch.sum(rel_score, dim=1)  # [4096, 6]

            res += rel_score

        return res
