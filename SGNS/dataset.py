from torch.utils.data import Dataset
import torch
import copy
class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, word_freqs,window_size,K):
        ''' 初始化
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
        ''' 
        每一个背景词对应self.K个负采样词
        每个返回值的元素都是张量

        Returns:
            center_word: list 中心词 [1]
            pos_words: list 背景词 [2*self.windows]
            neg_words: list 负采样词  [2*self.windows*self.K]
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