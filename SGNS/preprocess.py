from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
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
