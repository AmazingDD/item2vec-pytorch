import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from item2vec import Item2Vec

def choose_with_prob(p_discard):
    p = random.uniform()
    return False if p < p_discard else True

def sgns_sample_generator(train_seqs, vocabulary_size, context_window, discard=False):
    sgns_samples = []
    for seq in train_seqs:
        if discard:
            seq = [w for w in seq if choose_with_prob(prob_discard[w])]
        for i in range(len(seq)):
            target = seq[i]
            # generate positive sample
            context_list = []
            j = i - context_window
            while j <= i + context_window and j < len(seq):
                if j >= 0 and j != i:
                    context_list.append(seq[j])
                    sgns_samples.append([(target, seq[j]), 1])
                j += 1
            # generate negative sample
            for _ in range(len(context_list)):
                neg_idx = random.randrange(0, vocabulary_size)
                while neg_idx in context_list:
                    neg_idx = random.randrange(0, vocabulary_size)
                sgns_samples.append([(target, neg_idx), 0])
    return sgns_samples

class Item2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        label = self.data[index][1]
        xi = self.data[index][0][0]
        xj = self.data[index][0][1]
        label = torch.tensor(label, dtype=torch.float32)
        xi = torch.tensor(xi, dtype=torch.long)
        xj = torch.tensor(xj, dtype=torch.long)

        return xi, xj, label

    def __len__(self):
        return len(self.data)

df = pd.read_csv('./ml-100k/u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'], engine='python')
# for ml-100k, user and item id start from 1, so we need to turn to 0 to make it work
df['user'] -= 1
df['item'] -= 1

args = {
    'context_window': 2,
    'vocabulary_size': df['item'].nunique(),
    'rho': 1e-5,  # threshold to discard word in a sequence
    'batch_size': 256,
    'embedding_dim': 100,
    'epochs': 20,
    'learning_rate': 0.001,
}

train, test = train_test_split(df, test_size=0.2)
train_seqs = train.groupby('user')['item'].agg(list)

word_frequecy = train['item'].value_counts()
prob_discard = 1 - np.sqrt(args['rho'] / word_frequecy)
sgns_samples = sgns_sample_generator(train_seqs, args['vocabulary_size'], args['context_window'])

item2vec_dataset = Item2VecDataset(sgns_samples)
train_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=True)

model = Item2Vec(args)
model.fit(train_loader)
