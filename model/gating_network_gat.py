import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device

        self.W = Parameter(torch.Tensor(in_features, out_features)).to(device)
        self.W = torch.nn.init.xavier_uniform_(self.W)
        self.a = Parameter(torch.Tensor(2*out_features, 1)).to(device)
        self.a = torch.nn.init.xavier_uniform_(self.a)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print(input.shape())
        h = torch.matmul(input, self.W)
        # H = h.size()[1]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(
            -1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        e = torch.tanh(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
            # return torch.tanh(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device, step=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.step = step
        self.device = device

        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, device=device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, device=device, alpha=alpha, concat=False)

    def forward(self, x, adj):
        for i in range(self.step):
            # x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
            # x = F.dropout(x, self.dropout, training=self.training)
            # x = torch.tanh(self.out_att(x, adj))
            x = F.elu(self.out_att(x, adj))
            # x = F.log_softmax(x, dim=2)
        return x


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
        heads = self.args.h
        units = self.args.m
        alpha = 0.2
        dropout = 0.3
        nb_heads = 1

        # add gat
        # self.gat = GAT(dims_item, dims_user, L, T, step, device).to(device)
        self.gat = GAT(dims_item, dims_item, dims_item, dropout=dropout,
                       alpha=alpha, nheads=nb_heads, device=device, step=step).to(device)

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

        short_embs = self.gat(item_embs, A)
        # attention_embs = torch.mean(short_embs, dim=1)

        attention_matrix1 = torch.tanh(
            self.attention_item1(short_embs))  # [4095,5,128]
        # attention_matrix1 = self.attention_item1(short_embs)  # [4095,5,128]
        attention_matrix2 = self.attention_item2(
            attention_matrix1)  # [4096,5,20]
        attention_matrix = torch.softmax(
            attention_matrix2, dim=2)  # [4096,4,20]

        # no adaptivate attention
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
