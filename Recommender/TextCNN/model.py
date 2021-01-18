# coding: UTF-8
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json
from os.path import join
import time 
import pandas as pd 
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import math

class Model(nn.Module):
    def __init__(self, args, movie_num):
        super(Model, self).__init__()
        self.args = args
        # init embedding layer
        if args.embedding_pretrained is not None:
            embed_tensor = torch.tensor(np.load(args.embedding)["embeddings"].astype('float32'))
            self.embedding = nn.Embedding.from_pretrained(embed_tensor, freeze=False)
        else:
            # embedding:[vocab_size, embed_size]
            self.embedding = nn.Embedding(args.n_vocab, args.embed, padding_idx=args.n_vocab - 1)
        # define forward layer
        # kernel_size:[filter_height, filter_width=embed_size]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=args.num_filters, kernel_size=(k, args.embed)) for k in args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.full_connection = nn.Linear(in_features=args.num_filters * len(args.filter_sizes), out_features=movie_num)

    def conv_and_pool(self, x, conv_kernel):
        # x: [batch, 1, max_seq_len, embed_size],
        # conv_kernel:[in_channels=1, out_channels=num_filters, filter_height, filter_width=embed_size]
        # x=> [batch, num_filters, max_seq_len-filter_height+1, 1]
        #     [batch, num_filters, max_seq_len-filter_height+1]
        x = F.relu(conv_kernel(x)).squeeze(3)
        # x => [batch, num_filters, max_seq_len-filter_height+1]
        #  =>  [batch, num_filters, 1]
        #  =>  [batch, num_filters]
        x = F.max_pool1d(input=x, kernel_size=x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # check input and output
        # out:[batch, max_seq_len=50, embed_size]
        out = self.embedding(x[0])
        # out:[batch, 1, max_seq_len=50, embed_size]
        out = out.unsqueeze(1)
        # out => [batch, num_filters]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], dim=1)
        out = self.dropout(out)
        # out => [batch, num_filters]
        #     => [batch, movie_num]
        out = self.full_connection(out)
        # print(out.shape)  #torch.Size([64, 33834])
        return out

    def compute_loss(self, y_pred, y, subset='test'):
        # y_pred:[batch, movie_num]
        # y:[batch, 1]
        loss = F.cross_entropy(y_pred, y.squeeze())

        return loss

    def save_model(self, save_path, optimizer, epoch):
        # 存储时, 传入一个存储位置
        state = {'model':self.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, save_path)
        # torch.save(self.state_dict(), save_path)

    def load_model(self, save_path):
        checkpoint = torch.load(save_path, map_location=self.args.device)
        # print(checkpoint.keys())
        self.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        base_epoch = checkpoint['epoch'] + 1
        return base_epoch
    
