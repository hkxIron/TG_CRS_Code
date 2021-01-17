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
import transformers
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from tqdm import tqdm
from torch.utils.data import *
import ipdb
import math
from copy import deepcopy
from models.module import *

# SASRec模型用来融合用户item交互历史
class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        # embeddings: [batch, seq_len, hidden_size]
        self.embeddings = Embeddings(args)
        self.transfomer_encoder = Encoder(args) # transfomer transfomer_encoder
        self.args = deepcopy(args)

        self.act = nn.Tanh() # 激活函数
        self.dropout = nn.Dropout(p=args.hidden_dropout_prob)

        self.apply(self.init_sas_weights)

    def forward(self,
                input_ids, # [batch_size, seq_len]
                attention_mask=None, # [batch_size, seq_len]
                use_cuda=False,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (batch_size, seq_len)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64, (batch_size, 1, 1, seq_len), 在1,2处插入维度

        # 添加mask 只关注前几个物品进行推荐
        max_len = attention_mask.size(-1) # 输入的item长度
        attn_shape = (1, max_len, max_len)
        # 上三角, [1, max_len, max_len]
        subsequent_mask = torch.triu(torch.ones(attn_shape),
                                     diagonal=1)  # torch.uint8
        # 下三角,subsequent_mask: [1, 1, max_len, max_len]
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long() # 转成int64类型

        if use_cuda:
            subsequent_mask = subsequent_mask.to(self.args.device)
        # 下三角 extended_attention_mask: (batch_size, 1, 1, seq_len=max_len)
        # subsequent_mask:[1, 1, max_len, max_len]
        # extended_attention_mask:[batch_size, head_num=1, seq_len, seq_len]
        extended_attention_mask = extended_attention_mask * subsequent_mask

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # extended_attention_mask:[batch_size, 1, seq_len, seq_len]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        # extended_attention_mask:下三角全为0,上三角为-10000
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 如果是1的地方（不被mask），就是0, 如果是0的地方(将被mask)，就是-10000
        # input_ids:[batch_size, seq_len]
        # embedding:[batch_size, seq_len, embed_size]
        embedding = self.embeddings(input_ids, use_cuda)

        # extended_attention_mask 是一个上三角矩阵，上三角的元素为-10000.0， dtype与parameter相同
        # encoder见models.module.py
        # embedding:[batch_size, seq_len, embed_size]
        # extended_attention_mask:[batch_size, head_num=1, seq_len, seq_len]
        encoded_layers = self.transfomer_encoder(
            embedding,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)  # 是否输出所有layer的output
        # encoded_layers,有L层输出，每层的shape:[batch_size, seq_len, hidden_size]
        sequence_output = encoded_layers[-1] # 只取最后一层的输出
        # [batch_size, seq_len, hidden_size]
        return sequence_output

    def init_sas_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 线性以及embedding是用正态分布初始化
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            # layernorm用1与0初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # nn.Linear bias用0初始化
            module.bias.data.zero_()

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.args.device)

    def load_model(self, path):
        load_states = torch.load(path, map_location=self.args.device)
        load_states_keys = set(load_states.keys())
        this_states_keys = set(self.state_dict().keys())
        assert this_states_keys.issubset(this_states_keys) # 这里应该是this_states_keys.issubset(load_states_keys)？
        key_not_used = load_states_keys - this_states_keys
        for key in key_not_used:
            del load_states[key]

        self.load_state_dict(load_states)

    def compute_loss(self, y_pred, y, subset='test'):
        pass

    # seq_out:[batch, seq_len, hidden_size]
    # pos_ids:[batch, seq_len]
    # neg_ids:[batch, seq_len]
    # loss:[batch*seq_len,1]
    def cross_entropy(self, seq_out, pos_ids, neg_ids, use_cuda=True):
        # pos_ids:[batch, seq_len]
        # neg_ids:[batch, seq_len]
        # pos_emb:[batch, seq_len, hidden_size]
        # neg_emb:[batch, seq_len, hidden_size]
        pos_emb = self.embeddings.item_embeddings(pos_ids)
        neg_emb = self.embeddings.item_embeddings(neg_ids)

        # pos_emb:[batch, seq_len, hidden_size]
        # pos:[batch*seq_len, hidden_size], view相当于tf.reshape
        # neg:[batch*seq_len, hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # seq_out:[batch, seq_len, hidden_size]
        # seq_emb:[batch*seq_len, hidden_size]
        seq_emb = seq_out.view(-1, self.args.hidden_size)

        # pos:[batch*seq_len, hidden_size]
        # seq_emb:[batch*seq_len, hidden_size]
        # pos_logits:[batch*seq_len, 1]
        pos_logits = torch.sum(pos * seq_emb, -1) # 计算正样本embed与seq_emb的内积
        neg_logits = torch.sum(neg * seq_emb, -1) # 计算负样本embed与seq_emb的内积

        # pos_ids:[batch, seq_len]
        # is_target:[batch*seq_len]
        # 正样本的id>0,负样本的id<=0
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float()
        # -ylog(y_pred)- (1-y)log(1-y_pred)
        # 背后的物理意义是:seq_out_emb与正样本的embeding内积更大,即更相似,与负样本更不相似
        # pos_logits:[batch*seq_len, 1]
        # is_target:[batch*seq_len]
        # loss:[batch*seq_len,1]
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget \
                         -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget) \
               / torch.sum(istarget)
        return loss

# bert模型用来融合用户回复历史
class BERTModel(nn.Module):
    def __init__(self, args, num_class, bert_embed_size=768):
        super(BERTModel, self).__init__()
        bert_path = args.bert_path
        init_add = args.init_add
        self.args = args
        # 加载预训练的bert模型
        self.bert = BertModel.from_pretrained(bert_path)

        for param in self.bert.parameters():
            param.requires_grad = True # 需要对bert进行微调

        self.full_connection = nn.Linear(in_features=bert_embed_size, out_features=num_class)
        self.add_name = 'addition_model.pth'
        if init_add: # 从现有参数中初始化全连接层
            self.load_addition_params(join(bert_path, self.add_name))

    def forward(self, x, raw_return=True):
        # x:
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1] #  word_id/position_id/segment_id
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        # pooled: [batch_size,  bert_embed_size]
        _, pooled = self.bert(context,
                              token_type_ids=types,
                              attention_mask=mask)

        if raw_return: # 不需要经过全连接
            # [batch_size, hidden_size]
            return pooled
        else: # 需要经过全连接层
            # [batch_size, num_class]
            return self.full_connection(pooled)

    # logit: [batch_size, num_class]
    # y:[batch_size]
    # loss:[batch_size]
    def compute_loss(self, y_pred, y, subset='test'):
        # ipdb.set_trace()
        loss = F.cross_entropy(y_pred, y)
        return loss

    def save_model(self, save_path):
        self.bert.save_pretrained(save_path)
        torch.save(self.full_connection.state_dict(), join(save_path, self.add_name))

    def load_addition_params(self, path):
        self.full_connection.load_state_dict(torch.load(path,
                                                        map_location=self.args.device))

# 将bert以及SASRec的结果融合起来,是SASBERT模型的子模型
class Fusion(nn.Module):
    def __init__(self, args, num_class, bert_embed_size=768):
        super(Fusion, self).__init__()
        self.args = args
        concat_embed_size = bert_embed_size + args.hidden_size
        self.full_connection = nn.Linear(in_features=concat_embed_size, out_features=num_class)

    def forward(self, SASRec_out, BERT_out):
        # SASRec_out:[batch_size, hidden_size]
        # BERT_out:[batch_size, bert_embed_size]
        # represetation:[batch_size,  hidden_size+bert_embed_size]
        representation = torch.cat((SASRec_out, BERT_out), dim=1)
        # represetation:[batch_size, num_class]
        representation = self.full_connection(representation)
        return representation

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.args.device)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.args.device))

class SASBERT(nn.Module):
    # 就是把两个子模型定义挪到下面了
    def __init__(self, opt, args, num_class, bert_embed_size=768):
        super(SASBERT, self).__init__()
        self.args = args
        self.opt = opt

        # bert
        self.BERT = BERTModel(self.args, num_class)
        # SASRec
        self.SASREC = SASRecModel(self.args)
        self.fusion = Fusion(args, num_class)

        if args.load_model:
            self.SASREC.load_model(self.args.sasrec_load_path)
            self.fusion.load_model(args.fusion_load_path)

    def forward(self, x):
        x_bert = x[:3] # context, type, mask
        # pooled: [batch_size, hidden_size1]
        pooled = self.BERT(x_bert, raw_return=True)

        input_ids, target_pos, input_mask, sample_negs = x[-4:]
        # input_ids: [batch_size, seq_len]
        # input_mask:[batch_size, seq_len]
        # sequence_output:[batch_size, seq_len, hidden_size2]
        sequence_output = self.SASREC(
            input_ids,
            input_mask,
            self.args.use_cuda
        )
        # sequence_output:[batch_size, seq_len, hidden_size2]
        #              => [batch_size, hidden_size2]
        sequence_output = sequence_output[:, -1, :]

        # sequence_output:[batch_size, hidden_size2]
        # pooled:         [batch_size, hidden_size1]
        # represetation:  [batch_size, num_class]
        representation = self.fusion(sequence_output, pooled)
        return representation

    def save_model(self, module_name):
        if 'BERT' in module_name:
            self.BERT.save_model(self.opt['model_save_path'])
        if 'SASRec' in module_name:
            self.SASREC.save_model(self.opt['sasrec_save_path'])
        if 'Fusion' in module_name:
            self.fusion.save_model(self.opt['fusion_save_path'])

    def load_model(self, module_name, path):
        pass

    def compute_loss(self, y_pred, y, subset='test'):
        loss = F.cross_entropy(y_pred, y)
        return loss

    def get_optimizer(self):
        bert_param_optimizer = list(self.BERT.named_parameters())  # 模型参数名字列表
        # bert_param_optimizer = [p for n, p in bert_param_optimizer]
        other_param_optimizer = list(self.SASREC.named_parameters()) + list(self.fusion.named_parameters())
        other_param_optimizer = [p for n, p in other_param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.opt['lr_bert'])
        optimizer = transformers.AdamW(
            [
                # {'params': optimizer_grouped_parameters, 'lr': self.opt['lr_bert']},
                {
                    'params': [
                        p for n, p in bert_param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    'lr':
                    self.opt['lr_bert']
                },
                {
                    'params': [
                        p for n, p in bert_param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    'lr':
                    self.opt['lr_bert']
                },
                {
                    'params': other_param_optimizer
                }
            ],
            lr=self.opt['lr_sasrec']
        )

        return optimizer
