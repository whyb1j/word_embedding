from model import SGNS
import torch
from utils import evaluate
from preprocess import read_data,create_map
from dataset import WordEmbeddingDataset
text,word2idx,word_freqs=create_map('../text.npy')
model = SGNS(len(word2idx), 100)
model.load_state_dict(torch.load('./output/2024-04-14 05_40_21/model_weights_1.pth'))
vec=model.input_embedding()
evaluate('../SVD/svd_data/output_svd_100.txt','../result.txt',vec,word2idx)