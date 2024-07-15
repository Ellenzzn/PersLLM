import torch
from transformers import BertModel, AutoModel, BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from tqdm import tqdm, trange
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class passage_dataset(Dataset):
    def __init__(self, lines, model):
        self.max = 512
        self.dats = []
        self.tokenizer = BertTokenizer.from_pretrained(model)
        for line in lines:
            line = line.strip('\n').split(']')[1]
            self.dats.append(line)

    def __len__(self):
        return len(self.dats)

    def __getitem__(self, index):
        line = self.dats[index]
        tok = self.tokenizer.encode(line)
        inp = torch.zeros(self.max).long()
        inp[:min(self.max, len(tok))] = torch.tensor(tok[:min(self.max, len(tok))]).long()
        att = torch.zeros(self.max).long()
        att[:min(self.max, len(tok))] = 1
        return inp, att

def get_embedding(file_path, emb_path, model_path="IR-model/gtr-base"):
    if os.path.isfile(emb_path):
        embeddings = np.load(emb_path)
        return embeddings
    model = SentenceTransformer(model_path) 
    sentences = []
    lines = open(file_path).readlines()
    for idx in range(int(len(lines)/6)):
        sentences.append(lines[idx*6+1].strip('[question]'))
    embeddings = model.encode(sentences)
    np.save(emb_path, embeddings)
    return embeddings

name = 'petunia'

embed_early = get_embedding('data/'+name+'/early_thought_QA_with_COT.txt', 'data/'+name+'/ques_early_thought.npy')
embed_late = get_embedding('data/'+name+'/late_thought_QA_with_COT.txt', 'data/'+name+'/ques_late_thought.npy')

lines0 = open('data/'+name+'/early_thought_QA_with_COT.txt').readlines()
lines1 = open('data/'+name+'/late_thought_QA_with_COT.txt').readlines()

early_ids = []
late_ids = []
cnt = 0
embed_picked = None

for idx in range(len(embed_early)):
    if embed_picked is None:
        embed_picked = np.expand_dims(embed_early[idx], axis=0)
        early_ids.append(idx)
        continue
    scores = np.sum(embed_early[idx]*embed_picked, axis=-1)
    if max(scores)>=0.95:
        cnt += 1
    if max(scores)<0.95 and lines0[idx*6+2].lower().find('[response]')>-1:
        embed_picked = np.concatenate((embed_picked, np.expand_dims(embed_early[idx], axis=0)), axis=0)
        early_ids.append(idx)


for idx in range(len(embed_late)):
    if embed_picked is None:
        embed_picked = np.expand_dims(embed_late[idx], axis=0)
        late_ids.append(idx)
        continue
    scores = np.sum(embed_late[idx]*embed_picked, axis=-1)
    if max(scores)>=0.95:
        cnt += 1
    if max(scores)<0.95 and lines1[idx*6+2].lower().find('[response]')>-1:
        embed_picked = np.concatenate((embed_picked, np.expand_dims(embed_late[idx], axis=0)), axis=0)
        late_ids.append(idx)

print(len(early_ids), len(late_ids), cnt)

fw0 = open('data/'+name+'/'+name+'_dedup/early_thought_train.txt', 'w')
fw1 = open('data/'+name+'/'+name+'_dedup/early_thought_test.txt', 'w')
seq = np.arange(len(early_ids))
np.random.shuffle(seq)
lt = int(len(seq)/5)

for sq in seq[:lt]:
    for jdx in range(6):
        fw1.write(lines0[sq*6+jdx])
fw1.close()

for sq in seq[lt:]:
    for jdx in range(6):
        fw0.write(lines0[sq*6+jdx])
fw0.close()


fw0 = open('data/'+name+'/'+name+'_dedup/late_thought_train.txt', 'w')
fw1 = open('data/'+name+'/'+name+'_dedup/late_thought_test.txt', 'w')
seq = np.arange(len(late_ids))
np.random.shuffle(seq)
lt = int(len(seq)/5)

for sq in seq[:lt]:
    for jdx in range(6):
        fw1.write(lines1[sq*6+jdx])
fw1.close()

for sq in seq[lt:]:
    for jdx in range(6):
        fw0.write(lines1[sq*6+jdx])
fw0.close()

