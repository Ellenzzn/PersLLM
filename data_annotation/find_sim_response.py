import torch
from transformers import BertModel, AutoModel, BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from tqdm import tqdm, trange
import numpy as np
from sentence_transformers import SentenceTransformer

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
    emb_file_path_list =[]
    model = SentenceTransformer(model_path) 
    sentences = []
    responses = []
    lines = open(file_path).readlines()
    
    for idx in range(int(len(lines)/6)):
        sentences.append(lines[idx*6+1])
        responses.append(lines[idx*6+2])
    
    #sentences = lines
    embeddings = model.encode(sentences)
    np.save(emb_path, embeddings)
    #embeddings = np.load(emb_path)
    return embeddings, sentences, responses



embed_main, ques_main, ans_main = get_embedding('data/petunia/late_thought_QA_with_COT.txt', 'data/petunia/response_late_thought.npy')



embed_other = np.load('data/all_response.npy')
ans_other = open('all_response.txt').readlines()


fw = open('data/petunia/dpo_late_thought.txt', 'w')
for idx in range(len(embed_main)):
    scores = np.sum(embed_main[idx]*embed_other, axis=-1)
    srts = np.argsort(-scores)
    for st in srts[5:]:#10
        score = scores[st]
        if ans_main[idx]!=ans_other[st]:#score<0.95:
            break
    fw.write(ques_main[idx])
    fw.write(ans_main[idx])
    fw.write(ans_other[st])
    fw.write('\n')

fw.close()

