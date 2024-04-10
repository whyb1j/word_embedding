from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from collections import Counter
import numpy as np
import random
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import copy
class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, word_freqs,window,K):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__() # #通过父类初始化模型，然后重写两个方法
        # self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text] # 把文本数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = [word2idx.get(word,word2idx['<UNK>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        self.window=window
        self.K=K
        
    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数
    
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_word = torch.LongTensor([self.text_encoded[idx]]) # 取得中心词
        pos_indices = list(range(idx - self.window, idx)) + list(range(idx +1, idx + self.window + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices]
        # x=[-2,-1,0,1,2]
        # pos_indices = [i % 10000 for i in x] <<[9998, 9999, 0, 1, 2]
        select_weight = copy.deepcopy(self.word_freqs)
        select_weight[pos_words] = 0
        select_weight[center_word] = 0
        neg_words = torch.multinomial(select_weight, self.K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出
        # 的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        # while 循环是为了保证 neg_words中不能包含背景词
        return center_word, pos_words, neg_words
    
class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding_v = nn.Embedding(self.vocab_size, self.embed_size, sparse=True)
        self.embedding_u = nn.Embedding(self.vocab_size, self.embed_size, sparse=True)
        
    def forward(self, center_word, target_word, negative_word):
        ''' input_labels: center words, [batch_size, 1]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels:negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]  
        '''
        emb_v = self.embedding_v(center_word) # [batch_size, 1, embed_size]
        emb_u = self.embedding_u(target_word) # [batch_size, (window * 2), embed_size]
        emb_neg = self.embedding_u(negative_word)  # [batch_size, (window * 2 * K), embed_size]       
        pos_score = torch.sum(torch.mul(emb_v, emb_u), dim=2) #中心词和上下文词是一对一的因此逐个乘
        neg_score = torch.sum(torch.mul( emb_neg, emb_v),dim=2)
        #一个[中心词,上下文词]词对，对应K个负样本，因此要批量相乘
        log_pos =  F.logsigmoid(pos_score).squeeze() # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg =  F.logsigmoid(-1 * neg_score).squeeze()
        loss = log_pos + log_neg
        return -loss
    def input_embedding(self):
        return self.embedding_u.weight.data.cpu().numpy()
def build_vocab():
    with open('lmtraining.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    text=text.lower().split()
    # print(type(text)) >>list
    # print(len(text)) >>17005207
    # vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1)) # 得到单词字典表，key是单词，value是次数
    # vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values())) # 把不常用的单词都编码为"<UNK>"
    vocab_dict=dict(Counter(text)) #total=253854
    word2idx = {word:i for i, word in enumerate(vocab_dict.keys())}
    idx2word = {i:word for i, word in enumerate(vocab_dict.keys())}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts ** (3./4.)
    word_freqs = word_freqs / np.sum(word_freqs)
    return text,vocab_dict,word2idx,idx2word,word_freqs
if __name__ == '__main__':
    EMBEDDING_SIZE = 100
    MAX_VOCAB_SIZE=10000 #total=253854
    text,vocab_dict,word2idx,idx2word,word_freqs=build_vocab()
    dataset = WordEmbeddingDataset(text, word2idx, word_freqs,2,15)
    dataloader = DataLoader(dataset, batch_size=128,shuffle=True,num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    vocab_size=len(vocab_dict)
    model = SGNS(vocab_size,EMBEDDING_SIZE )
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for e in range(1):
        print("---start---")
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long().to(device)
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)
            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print('epoch', e, 'iteration', i, loss.item())
    # embedding_weights = model.input_embedding()
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))