import torch
import torch.nn as nn
import torch.nn.functional as F
class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SGNS, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size 
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        init_scale = 0.5 / embed_size
        self.in_embed.weight.data.uniform_(-init_scale, init_scale)
        self.out_embed.weight.data.uniform_(-init_scale, init_scale)

    def forward(self, center_word, target_word, negative_word):
        ''' input_labels: center words, [batch_size, 1]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels:negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]  
        '''
        emb_v = self.in_embed(center_word) # [batch_size, 1, embed_size]
        emb_u = self.out_embed(target_word) # [batch_size, (window * 2), embed_size]
        emb_neg = self.out_embed(negative_word)  # [batch_size, (window * 2 * K), embed_size]      
        emb_v=emb_v.unsqueeze(2)
        pos_score =torch.bmm(emb_u, emb_v) #中心词和上下文词是一对一的因此逐个乘
        pos_score =pos_score.squeeze(2)
        neg_score = torch.bmm(emb_neg,emb_v)
        #一个[中心词,上下文词]词对，对应K个负样本，因此要批量相乘
        log_pos =  F.logsigmoid(pos_score).sum(1) # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg =  F.logsigmoid(-1 * neg_score).sum(1)
        loss = log_pos + log_neg
        return -loss
    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()