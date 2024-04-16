from model import SGNS
from preprocess import WordEmbeddingDataset,read_data,create_map
from utils import evaluate
from argparse import Namespace
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import datetime
import os
torch.manual_seed(1)
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
os.mkdir(nowtime)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# wandb.login()   
config=Namespace(
    project_name="sgns_try",
    epochs=20,
    embed_size=100,
    batch_size=512,
    num_workers=16,
    lr=0.001,
    optim_type='Adam',
    window_size=2,
    K=15,
    data_path='../original_data/lmtraining.txt',
    train_path='../text.npy',
    test_path='../original_data/wordsim353_agreed.txt',
    log_ct=2000,    #记录频率
    checkpoint=False,
    checkpoint_path=""
)
save_path=f'{nowtime}/output.txt'
def model_pipeline(config=config):
    # tell wandb to get started 
    with wandb.init(project=config.project_name, config=config.__dict__,name=nowtime):
        # make the model, data, and optimization problem
        model, loader, optimizer, word2idx = make(config)
        # and use them to train the model
        train(model, loader, optimizer, config)
        # and test its final performance
        embedding_weights = model.input_embedding()
        vec=embedding_weights
        evaluate(config.test_path,save_path, vec, word2idx)
        return model
def make(config):
    # Make the data
    if not os.path.exists(config.train_path):
        read_data(config.data_path,config.train_path)
    text,word2idx,word_freqs=create_map(config.train_path)
    dataset=WordEmbeddingDataset(text,word2idx,word_freqs,config.window_size,config.K)
    loader = DataLoader(dataset, batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True,persistent_workers=True,pin_memory=True)
    # Make the model
    model = SGNS(len(word2idx), config.embed_size)
    model=model.to(device)
    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint_path))
    # Make the loss and optimizer
    optimizer = torch.optim.__dict__[config.optim_type](params=model.parameters(), lr=config.lr) 
    return model, loader, optimizer, word2idx
def train(model, loader, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="gradients")
    # Run training and track with wandb
    # total_batches = len(loader) * config.epochs
    # example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in range(1,config.epochs+1):
        for _, (input_labels, pos_labels, neg_labels) in tqdm(enumerate(loader),total=len(loader)):
            loss = train_batch(input_labels, pos_labels, neg_labels,  model, optimizer)
            # example_ct +=  len(input_labels)
            batch_ct += 1
            # Report metrics every count batch
            if ((batch_ct + 1) % config.log_ct) == 0:
                wandb.log({"loss": loss})
        torch.save(model.state_dict(), f'{nowtime}/model_weights_{epoch}.pth')
def train_batch(input_labels, pos_labels, neg_labels,  model, optimizer):
    input_labels, pos_labels, neg_labels = input_labels.to(device), pos_labels.to(device), neg_labels.to(device)
    # Forward pass ➡
    loss = model(input_labels, pos_labels, neg_labels).mean()
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    # Step with optimizer
    optimizer.step()
    return loss


if __name__ == "__main__":
    model_pipeline(config)