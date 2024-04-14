from nltk.stem import PorterStemmer
import numpy as np
def evaluate(test_path,save_path,vec,word2idx):     
    print('---开始评测---')
    result=[]
    vocab=set([word for word in word2idx.keys()])
    stemmer = PorterStemmer()   
    with open(test_path, 'r') as file:
        for line in file:
            parts = line.lower().split()
            word1,word2=parts[1],parts[2]
            word1,word2=stemmer.stem(word1),stemmer.stem(word2)
            if word1 in vocab and word2 in vocab:
                index1 = word2idx[word1]
                index2 = word2idx[word2]  
                # 计算余弦相似度
                vec1 = vec[index1]
                vec2 = vec[index2]
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            else:
                # 若有词不在词汇表中，将相似度设为0
                cosine_sim = 0.0
            temp=line.split()
            temp.append(cosine_sim)
            result.append(temp)
    print("---保存结果---")
    with open(save_path, 'w') as file:
        # 遍历结果列表
        for item in result: 
            # 将每个元素转换为字符串，并以制表符分隔
            line = '\t'.join(map(str, item))
            # 写入文件
            file.write(line + '\n')
