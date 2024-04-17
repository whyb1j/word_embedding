from scipy.sparse import coo_matrix,save_npz,load_npz
from scipy.sparse.linalg import svds
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import os
np.random.seed(42)
class svd_embedding():
    """实现了svd降维构建词向量

    输入的训练集文本要求以空白字符为间隔

    Attributes:
        train_path: str 训练集路径
        test_path: str 测试集路径
        save_dir: str 保存目录
        text: list 处理好的文本内容
        vocab: set 处理好的词汇表
        word2idx: dict 单词对应的索引
        idx2word: dict 索引对应的单词
        co_matrix: coo 稀疏矩阵
        english_stopwords list 停用词列表
        stemmer PorterStemmer 词干提取器

    """
    def __init__(self, train_path: str, test_path: str, K: int, dim: int, save_dir: str = 'svd_data'):   
        """初始化

        Args:
            train_path: 训练集路径
            test_path: 测试数据路径
            K: 滑动窗口大小
            dim: 词向量维度
            save_dir: 数据保存目录
        """
        self.train_path=train_path
        self.test_path=test_path
        self.K=K
        self.dim=dim
        self.english_stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        if not os.path.isdir(save_dir):   
            os.mkdir(save_dir)
        self.save_dir=save_dir
    def read_data(self):
        """读取训练集文本,对文本做如下处理:小写化,词干提取,去除停用词 
        
        处理前的vocab_size=253854,text_size=17005207
        """
        print('---开始读数据---')
        path=os.path.join(self.save_dir,'text.npy')
        if os.path.exists(path):
            print('---直接加载已有数据---')
            self.text=np.load(path)
        else:
            with open(self.train_path, 'r', encoding='utf-8') as file:
                text = file.read().lower().split()
            text = [word for word in text if word not in self.english_stopwords]
            self.text = [self.stemmer.stem(word) for word in text]#>>10890638
            np.save(path,self.text)
        #不用set是因为集合的无序性，对集合迭代单词和索引的映射会出问题
        #误会了。
        #当你创建一个set对象时，它的元素会根据它们被添加到集合中的顺序来迭代，
        #即使set本身并不保证元素的顺序。这种行为是为了保证迭代的一致性，特别是当你在遍历集合时使用enumerate函数时。
        self.vocab=set(self.text)  #>>201626
        self.word2idx={word:idx for idx,word in enumerate(self.vocab)}
        self.idx2word={idx:word for idx,word in enumerate(self.vocab)}
    def create_co_matrix(self):
        """创建共现矩阵
        """
        print('---开始创建共现矩阵---')
        path=os.path.join(self.save_dir,'co_matrix.npz')
        if os.path.exists(path):
            print('---直接加载已有的共现矩阵---')
            co_matrix=load_npz(path)
        else:
            row=[]
            col=[]
            count=[]
            for i in range(len(self.text)) :
                center_word=self.word2idx[self.text[i]]
                #在句子中的位置
                window=list(range(max(0,i-self.K),min(i+1+self.K,len(self.text))))
                window.remove(i)
                for j in window :
                    #要转换为index
                    if center_word!=self.word2idx[self.text[j]] :
                        row.append(center_word)
                        col.append(self.word2idx[self.text[j]])
                        count.append(1)
            co_matrix = coo_matrix((count, (row, col)), shape=(len(self.vocab),len(self.vocab)),dtype=np.float32)
            save_npz(path, co_matrix)
        self.co_matrix=co_matrix
    def svd(self):
        """奇异值分解
        """
        print('---开始进行奇异值分解---')
        path=os.path.join(self.save_dir,f'U_{self.dim}.npy')
        if os.path.exists(path):
            print('---直接加载降维后的矩阵---')
            U=np.load(path)
            S=np.load(os.path.join(self.save_dir,f'S_{self.dim}.npy'))
        else:
            U, S, Vt = svds(self.co_matrix, k=self.dim)
            np.save(path, U)
            np.save(os.path.join(self.save_dir,f'S_{self.dim}.npy'),S)
        self.vec_svd = U
        self.S = S
    def evaluate(self):
        """计算文本中两个词之间的余弦相似度
        """
        print('---开始评测---')
        result=[]
        with open(self.test_path, 'r') as file:
            for line in file:
                parts = line.lower().split()
                word1,word2=parts[1],parts[2]
                word1,word2=self.stemmer.stem(word1),self.stemmer.stem(word2)
                if word1 in self.vocab and word2 in self.vocab:
                    index1 = self.word2idx[word1]
                    index2 = self.word2idx[word2]  
                    # 计算余弦相似度
                    vec1 = self.vec_svd[index1]
                    vec2 = self.vec_svd[index2]
                    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                else:
                    # 若有词不在词汇表中，将相似度设为0
                    cosine_sim = 0.0
                temp=line.split()
                temp.append(cosine_sim)
                result.append(temp)
        self.result=result
        path=os.path.join(self.save_dir,f'output_svd_{self.dim}.txt')
        print("---保存结果---")
        with open(path, 'w') as file:
            # 遍历结果列表
            for item in result: 
                # 将每个元素转换为字符串，并以制表符分隔
                line = '\t'.join(map(str, item))
                # 写入文件
                file.write(line + '\n')
    def algorithm(self):
        self.read_data()
        self.create_co_matrix()
        self.svd()
        self.evaluate()
if __name__=='__main__':
    word2vec_svd=svd_embedding('../original_data/lmtraining.txt','../original_data/wordsim353_agreed.txt',5,100)
    word2vec_svd.algorithm()