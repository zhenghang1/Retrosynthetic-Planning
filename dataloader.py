import pickle as pkl
import numpy as np
import faiss
import torch
import torch.nn as nn
import dgl
from torch.utils.data import Dataset, DataLoader


class schneider_set(Dataset):
    def __init__(self, datapath):
        self.data = pkl.load(open(datapath, 'rb'))

    def __getitem__(self, index):
        return torch.tensor(self.data[index, 0], dtype=torch.float32),\
            torch.tensor(self.data[index, 1], dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    

def schneider_loader(args):
    train_set = schneider_set("../data/processed_schneider50k/train.pkl")
    val_set = schneider_set("../data/processed_schneider50k/val.pkl")
    test_set = schneider_set("../data/processed_schneider50k/test.pkl")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    return train_loader, val_loader, test_loader



class molecule_eval_set(Dataset):
    def __init__(self, datapath, k=20):
        ori_data = pkl.load(open(datapath, "rb"))
        fp = np.unpackbits(ori_data['packed_fp'], axis=1).astype('float32')
        self.fp = torch.tensor(fp)
        self.values = ori_data['values']
        self.neighbors = np.load(datapath.replace('pkl', 'npy'))[:,:k]
        self.k = k

        neigh_data = pkl.load(open(datapath.replace("test", "train"), "rb"))
        neigh_fp = np.unpackbits(neigh_data['packed_fp'], axis=1).astype('float32')
        self.neigh_fp = torch.tensor(neigh_fp)
        self.neigh_vals = neigh_data['values']
    
    def __getitem__(self, index):
        g = dgl.graph([(i, j) for i in range(self.k+1) for j in range(i+1, self.k+1)])
        fps = torch.cat((self.fp[index].unsqueeze(0), self.neigh_fp[self.neighbors[index],:]), dim=0)
        g.ndata['fp'] = fps
        costs = torch.cat((self.values[index].unsqueeze(0), self.neigh_vals[self.neighbors[index]]), dim=0)
        return g, costs
    
    def __len__(self):
        return self.fp.shape[0]
    


class Collator(object):
    def __init__(self):
        pass

    def collate(self, batch):
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        return batch_graphs, batch_labels



def molecule_eval_loader(args):
    collator = Collator()
    train_set = molecule_eval_set("../data/MoleculeEvaluationData/train.pkl", args.k)
    test_set = molecule_eval_set("../data/MoleculeEvaluationData/test.pkl", args.k)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True, collate_fn=collator.collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True, collate_fn=collator.collate)
    return train_loader, test_loader