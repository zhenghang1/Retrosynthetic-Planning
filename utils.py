from data.rdchiral.rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import os
import faiss
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from tqdm import trange


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight.data)


def get_template(reaction):
    reactants, products = reaction.split('>>')
    inputRec = {'_id': None, 'reactants': reactants, 'products': products}
    ans = extract_from_reaction(inputRec)
    if 'reaction_smarts' in ans.keys():
        # print(ans['reaction_smarts'])
        return ans['reaction_smarts']
    else:
        # print("NULL")
        return None


def get_product_fingerprint(product):
    mol = Chem.MolFromSmiles(product)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=int)
    arr[onbits] = 1
    return arr


def get_samples(datapath):
    df = pd.read_csv(datapath).iloc[:,-1]
    dataset = []
    for reaction in df.to_numpy():
        _, products = reaction.split('>>')
        template = get_template(reaction)
        if template == None:
            continue
        fingerprint = get_product_fingerprint(products)
        dataset.append([fingerprint, template])
    print(f"Size: {len(dataset)}")
    return np.array(dataset)


def process_schneider():
    train_data = get_samples("data/schneider50k/raw_train.csv")
    val_data = get_samples("data/schneider50k/raw_val.csv")
    test_data = get_samples("data/schneider50k/raw_test.csv")

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([train_data[:,1], val_data[:,1], test_data[:,1]]))

    train_data[:,1] = encoder.transform(train_data[:,1]).astype(int)
    val_data[:,1] = encoder.transform(val_data[:,1]).astype(int)
    test_data[:,1] = encoder.transform(test_data[:,1]).astype(int)

    os.makedirs("data/processed_schneider50k", exist_ok=True)
    pkl.dump(train_data, open("data/processed_schneider50k/train.pkl", 'wb'))
    pkl.dump(val_data, open("data/processed_schneider50k/val.pkl", 'wb'))
    pkl.dump(test_data, open("data/processed_schneider50k/test.pkl", 'wb'))

    print(f"Total classes:{len(encoder.classes_)}")


def buildIndex(datapath):
    ori_data = pkl.load(open(datapath, "rb"))
    fp = np.unpackbits(ori_data['packed_fp'], axis=1).astype('float32')
    print("Fingerprint loaded, construct index.")
    print(fp.shape)

    index = faiss.index_factory(fp.shape[1], "HNSW64", faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(fp)
    print("finish normalization")

    index.add(fp)
    faiss.write_index(index, os.path.abspath(datapath+"/../fp.index"))
    print("index created")


def save_knn(datapath):
    print("Saving knn of samples")

    # load feature
    ori_data = pkl.load(open(datapath, "rb"))
    fp = np.unpackbits(ori_data['packed_fp'], axis=1).astype('float32')
    print("fp loaded")

    index = faiss.read_index(os.path.abspath(datapath+"/../fp.index"))
    print("index loaded")

    # checking rate of neighbours with same label
    batch_size = 512
    batch_num = fp.shape[0]//batch_size
    phase = "train" if datapath.find("train") != -1 else "test"
    k = 100 if phase == "train" else 99
    print(phase)

    all_retrieved = []
    with trange(batch_num) as t:
        for i in t:
            batch = fp[i*batch_size:(i+1)*batch_size]
            faiss.normalize_L2(batch)
            _, indice = index.search(batch, k+1)
            all_retrieved.append(indice)

    # remaining samples
    batch = fp[batch_size*batch_num:]
    faiss.normalize_L2(batch)
    _, indice = index.search(batch, k+1)
    all_retrieved.append(indice)

    # save
    all_retrieved = np.concatenate(all_retrieved)
    np.save(os.path.abspath(datapath+"/../"+phase+".npy"), all_retrieved)
    print("finish saving")


if __name__ == '__main__':
    process_schneider()
    buildIndex("data/MoleculeEvaluationData/train.pkl")
    save_knn("data/MoleculeEvaluationData/train.pkl")
    save_knn("data/MoleculeEvaluationData/test.pkl")
    pass
