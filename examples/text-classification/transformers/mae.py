
import torch
import numpy as np
import pdb
import IPython
import torch.nn as nn
from itertools import combinations
from .modeling_albert import *
from torch.nn.functional import gelu
from torch.distributions import Categorical

class MixtureOfExpert(nn.Module):
    def __init__(self, hidden_size, num_groups, per_head_num, total_head_num):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.per_head_num = per_head_num
        self.total_head_num = total_head_num
        self.scaleup_factor = self.total_head_num / self.per_head_num
        self.index_groups = torch.tensor(list(map(list, combinations(range(num_groups), per_head_num))))
        self.dense1 = nn.Linear(hidden_size, num_groups)
        self.actFN = gelu
        self.dense2 = nn.Linear(num_groups, num_groups)
        self.softmax = nn.Softmax(-1)

    def forward(self, input_data_seq, batch_head_matrix, evaluate=False, every_five_steps=False):

        input_data_mean = torch.mean(input_data_seq, dim=1)
        hidden1 = self.dense1(input_data_mean)
        hiddenact1 = self.actFN(hidden1)
        hidden2 = self.dense2(hiddenact1)
        prob = self.softmax(hidden2)

        scaleup = self.scaleup_factor * batch_head_matrix

        if evaluate:
            prob_multiply_head = [
            torch.sum(
                prob[:, i,None,None].expand(-1, input_data_seq.shape[1], self.total_head_num - 1).unsqueeze(-1).cuda() *
                torch.index_select(scaleup, 2, self.index_groups[i].cuda()),
                dim=-2)
            for i in range(12)
        ]
            prob_matrix = torch.sum(torch.stack(prob_multiply_head), dim=0)
        else:
            if every_five_steps:
                prob_multiply_head = [
                torch.sum(
                prob[:, i,None,None].expand(-1, input_data_seq.shape[1], self.total_head_num - 1).unsqueeze(-1).cuda() *
                torch.index_select(scaleup, 2, self.index_groups[i].cuda()),
                dim=-2)
                for i in range(12)
                ]
                prob_matrix = torch.sum(torch.stack(prob_multiply_head), dim=0)
            else:

                batch_indexes = Categorical(prob).sample()
                # prob_matrix = torch.sum(batch_head_seq_hidden,dim=2)
                sampled_ind = batch_indexes + torch.range(0, len(batch_indexes)-1).cuda().to(torch.long) * scaleup.shape[2]
                sampled = scaleup.permute(0, 2, 1, 3).reshape(-1, 128, 768)[sampled_ind]
                prob_matrix = torch.sum(scaleup, dim=2) - sampled

                # index_selection = self.index_groups[batch_indexes].cuda()
                
                # batch_head_seq_hidden = torch.stack([torch.index_select(scaleup[i,:,:,:], 1, index_selection[i]) for i in range(input_data_seq.shape[0])],dim=0)
                # prob_matrix = torch.sum(batch_head_seq_hidden,dim=2)
                # import pdb; pdb.set_trace()

        return prob, prob_matrix, batch_head_matrix


class MixtureAttentionWeightExpert(nn.Module):
    def __init__(self, hidden_size, num_groups, per_head_num, total_head_num):
        super().__init__()

        self.num_groups = num_groups
        self.per_head_num = per_head_num
        self.short_hidden_size = int(hidden_size / self.per_head_num)
        self.total_head_num = total_head_num
        self.scaleup_factor = 1 / self.per_head_num
        self.dense1 = nn.Linear(self.short_hidden_size, num_groups)
        self.actFN = gelu
        self.dense2 = nn.Linear(num_groups, num_groups)
        self.softmax = nn.Softmax(-1)

    def forward(self, input_data_seq, attention_probs,  value_layer, evaluate=False, every_five_steps=False):

        input_data_mean = torch.mean(input_data_seq, dim=1)
        input_data_12_split= input_data_mean.reshape(input_data_mean.shape[0], 12, self.short_hidden_size)
        hidden1 = self.dense1(input_data_12_split)
        hiddenact1 = self.actFN(hidden1)
        hidden2 = self.dense2(hiddenact1)
        prob = self.softmax(hidden2)
        

        scaleup = self.scaleup_factor * attention_probs

        w = torch.matmul(scaleup, value_layer)
        context = w.permute(0, 2, 1, 3).contiguous()

        if evaluate:
            prob_multiply_head = [
            torch.sum(prob[:, i].unsqueeze(1).expand(input_data_seq.shape[0], input_data_seq.shape[1], 12).unsqueeze(-1).cuda() *
                context.cuda(),
                dim=-2)
            for i in range(12)
            ]

            prob_matrix = torch.stack(prob_multiply_head).permute(2,0,1,3).reshape(input_data_seq.shape[0],input_data_seq.shape[1],input_data_seq.shape[2])

        else:
            if every_five_steps:
                prob_multiply_head = [
                torch.sum(prob[:, i].unsqueeze(1).expand(input_data_seq.shape[0], input_data_seq.shape[1], 12).unsqueeze(-1).cuda() *
                context.cuda(),
                dim=-2)
                for i in range(12)
                ]

                prob_matrix = torch.stack(prob_multiply_head).permute(2,0,1,3).reshape(input_data_seq.shape[0],input_data_seq.shape[1],input_data_seq.shape[2])
            else:

                scaleup = scaleup / self.scaleup_factor  
                orig_shape = scaleup.shape

                batch_indexes = Categorical(prob).sample()
                # sampled_ind = batch_indexes + (torch.range(0, len(batch_indexes)-1).cuda().to(torch.long) * scaleup.shape[1]).view(-1, 1)
                # scaleup = scaleup.reshape(-1, scaleup.shape[2], scaleup.shape[3])
                
                sampled_ind = torch.range(0, len(batch_indexes)-1).cuda().to(torch.long).view(-1, 1).expand(-1, 12).reshape(-1)
                # import pdb;pdb.set_trace()
                scaleup = scaleup[(sampled_ind, batch_indexes.view(-1))].view(orig_shape)

                # scaleup = torch.stack([torch.index_select(scaleup[i],0,batch_indexes[i]) for i in range(input_data_seq.shape[0])], dim=0)

                
                w = torch.matmul(scaleup, value_layer)
            context = w.permute(0, 2, 1, 3).contiguous()

        return prob, context, value_layer