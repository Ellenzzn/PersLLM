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
            line = line.strip('\n').split('】')[1]
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


def get_embedding(file_list, model_path):
    emb_file_path_list =[]
    if model_path.find('gtr')>-1:
        #model = SentenceTransformer(model_path)
        for file in file_list:
            sentences = []
            lines = open(file).readlines()
            for line in lines:
                sentences.append(line.split('】')[-1])
            #embeddings = model.encode(sentences)
            
            emb_file_path = file.split('.')[0] + '_emb.npy'
            #np.save(emb_file_path, embeddings)
            emb_file_path_list.append(emb_file_path)
        return emb_file_path_list

    #model = AutoModel.from_pretrained(model_path, output_hidden_states=True)#, add_pooling_layer=False)

    for file in file_list:
        lines = open(file).readlines()
        '''
        DataSet = passage_dataset(lines, model_path)
        dataloader = DataLoader(DataSet, shuffle=False, batch_size=8,
                                num_workers=4, pin_memory=True, drop_last=False)
        model.eval()
        with torch.no_grad():
            allout = None
            for batch in tqdm(dataloader):
                (tokdes, attdes) = batch
                logits = model(tokdes, attention_mask=attdes)#.cuda(), attention_mask=attdes.cuda())
                if allout is None:
                    allout = logits[0][:, 0, :]
                else:
                    allout = torch.cat((allout, logits[0][:, 0, :]), dim=0)
        '''
        emb_file_path = file.split('.')[0] + '_emb.npy'
        #np.save(emb_file_path, allout.detach().cpu())
        emb_file_path_list.append(emb_file_path)

    return emb_file_path_list
