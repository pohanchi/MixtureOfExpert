
import torch
import numpy as np
import tensorflow as tf
import ipdb
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

                batch_indexes = Categorical(prob).sample()

                scaleup = torch.stack([torch.index_select(scaleup[i],0,batch_indexes[i]) for i in range(input_data_seq.shape[0])], dim=0)

                w = torch.matmul(scaleup, value_layer)
                context = w.permute(0, 2, 1, 3).contiguous()

        return prob, context, value_layer


class AlbertAttention_MAE(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        mixed_value_layer = self.value(input_ids)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.synthesizer:
            if self.mix:
                if self.full_att:
                    mixed_query_layer = self.query(input_ids)
                    mixed_key_layer = self.key(input_ids)
                    query_layer = self.transpose_for_scores(mixed_query_layer)[:, self.hand_crafted:, :, :]
                    key_layer = self.transpose_for_scores(mixed_key_layer)[:, self.hand_crafted:, :, :]
                    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
                else:
                    hidden_transposed = self.transpose_for_scores(input_ids)
                    hidden_transposed = hidden_transposed[:, self.hand_crafted:, :, :]
                    query_layer = self.query_(hidden_transposed)
                    attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

                attention_scores = torch.cat((self.R.R + attention_mask, attention_scores + attention_mask), axis=1)
            else:
                attention_scores = self.R.R + attention_mask
        else:
            if self.full_att:
                mixed_query_layer = self.query(input_ids)
                mixed_key_layer = self.key(input_ids)
                query_layer = self.transpose_for_scores(mixed_query_layer)
                key_layer = self.transpose_for_scores(mixed_key_layer)
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            else:
                query_layer = self.transpose_for_scores(input_ids)
                query_layer = self.query_(query_layer)

                # Take the dot product between "query" and "key" to get the raw attention scores.
                attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size) + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        # attention head: each 64 rows represent a head embeddings.
        projected_context_layer = torch.einsum("bfnd,ndh->bfnh", context_layer, w)


        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)
