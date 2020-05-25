class Dense_projection(nn.Module):
    def __init__(self, attention_head_size, seq_len):
        super().__init__()
        self.dense1 = nn.Linear(attention_head_size, 16)
        self.dense2 = nn.Linear(16, seq_len)
        self.act = ACT2FN['relu']

    def forward(
        self,
        hidden_states
    ):
        res = self.act(self.dense1(hidden_states))
        res = self.dense2(res)
        return res



class Conv_1(nn.Module):
    def __init__(self, attention_head_size, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(attention_head_size, seq_len, 5, padding=2, padding_mode='replicate')

    def forward(
        self,
        hidden_states
    ):
        res = self.conv1(hidden_states.permute(0, 2, 1))
        return res


class Rand_par(nn.Module):  # attention map
    def __init__(self, seq_len, num_attention_heads, noatt, requires_grad=True):
        super(Rand_par, self).__init__()

        rand_att = list()


        rand_att.append(torch.eye(seq_len))
        rand_att.append(torch.roll(torch.eye(seq_len), 1, -1))
        rand_att.append(torch.roll(torch.eye(seq_len), -1, -1))


        tmp = torch.pow(torch.range(0, seq_len-1)+1, 3).unsqueeze(0).expand(seq_len, -1)
        mask = torch.tril(torch.roll(torch.tril(1.0-torch.eye(seq_len)), -1, -1))
        mask[-1][-1] = 0.0
        tm = tmp*mask
        
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        
        mask = torch.triu(torch.roll(torch.triu(1.0-torch.eye(seq_len)), 1, -1))
        mask[1][1] = 0.0
        tm = tmp*mask
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        tm = tmp
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        tmp = torch.pow(torch.flip(torch.range(0, seq_len-1)+1, (0,)), 3).unsqueeze(0).expand(seq_len, -1)

        tm = tmp
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        R = torch.stack(rand_att)

        R[torch.where(R == 0)] = -10000.0

        hand_crafted = 7

        if num_attention_heads > hand_crafted:
            self.R = torch.cat((R, torch.randn((num_attention_heads-hand_crafted, seq_len, seq_len)).normal_(mean=0, std=1)), dim=0)
        else:
            self.R = R
        self.R = nn.Parameter(self.R, requires_grad=requires_grad)



class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        


        self.value = nn.Linear(config.hidden_size, self.all_head_size)


        self.synthesizer = config.synthesizer # bool: 是否使用synthesizer架構
        self.mix = config.mix # 確定使用synthesizer時，是否使用mix的作法（7個head用synthesizer，剩下的用本來的full attention或是變體）
        self.full_att = config.full_att # bool: 是否使用本來的full attention


        if self.synthesizer:
            self.seq_len = 128 # synthesizer系列的module (random initialized/had-crafted 的attention score 或是 直接把hiddenstate餵進dense/conv 產生attention score) 會用到
            
            # synthesizer系列的module
            # self.dense_attn = Dense_projection(config.hidden_size, self.seq_len) # 兩層的dense，把hidden state直接做成attention score
            # self.conv1 = Conv_1(self.attention_head_size, self.seq_len) # 一層的conv_1d，把hidden state直接做成attention score
            if self.mix:
                self.R = Rand_par(self.seq_len, 7, config.rand_nonatt)
                if self.full_att:
                    self.query = nn.Linear(config.hidden_size, self.all_head_size)
                    self.key = nn.Linear(config.hidden_size, self.all_head_size)
                else:
                    # 本來的 full attention的變體，先把hidden state弄成12head的形狀（batch, 12, seq_len, 768/12）再投影
                    self.query_ = nn.Linear(self.attention_head_size, self.attention_head_size)
            else:
                self.R = Rand_par(self.seq_len, 12, config.rand_nonatt) # nn.Parameter, （head數, seq_len, seq_len）的矩陣，直接拿出來就是attention score
        else:
            if self.full_att:
                self.query = nn.Linear(config.hidden_size, self.all_head_size)
                self.key = nn.Linear(config.hidden_size, self.all_head_size)
            else:
                # 本來的 full attention的變體，先把hidden state弄成12head的形狀（batch, 12, seq_len, 768/12）再投影
                self.query_ = nn.Linear(self.attention_head_size, self.attention_head_size)


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):

        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.synthesizer:
            if self.mix:
                if self.full_att:
                    mixed_query_layer = self.query(hidden_states)
                    mixed_key_layer = self.key(hidden_states)
                    query_layer = self.transpose_for_scores(mixed_query_layer)[:, 7:, :, :]
                    key_layer = self.transpose_for_scores(mixed_key_layer)[:, 7:, :, :]
                    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
                else:
                    hidden_transposed = self.transpose_for_scores(hidden_states)
                    hidden_transposed = hidden_transposed[:, 7:, :, :]
                    query_layer = self.query_(hidden_transposed)
                    attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

                attention_scores = torch.cat((self.R.R + attention_mask, attention_scores + attention_mask), axis=1)
            else:
                attention_scores = self.R.R
        else:
            if self.full_att:
                mixed_query_layer = self.query(hidden_states)
                mixed_key_layer = self.key(hidden_states)
                query_layer = self.transpose_for_scores(mixed_query_layer)
                key_layer = self.transpose_for_scores(mixed_key_layer)
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            else:
                query_layer = self.transpose_for_scores(hidden_states)
                query_layer = self.query_(query_layer)

                # Take the dot product between "query" and "key" to get the raw attention scores.
                attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)


        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

