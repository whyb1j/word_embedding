from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
import torch
import copy
import numpy as np
class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, word_freqs,window_size,K):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super().__init__() # #通过父类初始化模型，然后重写两个方法
        self.text=text
        self.text_encoded = [word2idx.get(word) for word in self.text] # 把文本转换成数字编码
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.tensor(word_freqs,dtype=torch.float32)
        self.window_size=window_size
        self.K=K
        
    def __len__(self):
        return len(self.text_encoded) # 返回所有单词的总数，即item的总数
    
    def __getitem__(self, idx)  :
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_word = self.text_encoded[idx] # 取得中心词
        left = list(range(idx - self.window_size, idx))
        right = list(range(idx + 1, idx + self.window_size + 1))
        pos_indices = [i % len(self.text) for i in left + right]  
        # pos_indices=list(range(max(0,idx-self.window),min(idx+1+self.window,len(self.text))))
        # pos_indices.remove(idx)      
        pos_words = self.text_encoded[pos_indices]
        select_weight = copy.deepcopy(self.word_freqs)
        select_weight[pos_words] = 0
        select_weight[center_word] = 0
        neg_words = torch.multinomial(select_weight, self.K * pos_words.shape[0], True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        # while 循环是为了保证 neg_words中不能包含背景词
        return center_word, pos_words, neg_words
def read_data(data_path,train_path):
    """读取指定路径的txt文件,经过处理后,保存为npy文件
    """
    print("---读取文本数据---")
    english_stopwords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    with open(data_path, 'r', encoding='utf-8') as file:
        text = file.read().lower().split()
    text = [word for word in text if word not in english_stopwords]
    text = [stemmer.stem(word) for word in text]#>>10890638
    np.save(train_path,text)
def create_map(data_path):
    """读取指定路径的npy文件,建立词到索引的映射,统计词频
    """
    print('---建立词到索引的映射---')
    text=np.load(data_path)
    vocab_dict = dict(Counter(text))# 得到单词字典表，key是单词，value是次数
    word2idx = {word:i for i, word in enumerate(vocab_dict.keys())}
    idx2word = {i:word for i, word in enumerate(vocab_dict.keys())}
    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs =word_counts ** (3./4.)
    word_freqs = word_freqs / np.sum(word_freqs)
    return text,word2idx,word_freqs
