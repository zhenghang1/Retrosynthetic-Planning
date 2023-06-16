import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import schneider_loader
from models import MLP
import os, argparse
from sklearn.metrics import accuracy_score
from utils import seed_all, weight_init


def evaluate(model, data_loader, device, k=10):
    model.eval()
    acc = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.detach().cpu().numpy().astype('float64')
            sorted_indices = np.argsort(pred, axis=1)[:,::-1]
            topk_pred = sorted_indices[:,:k]
            y = y.detach().cpu().numpy()

            correct = np.any(topk_pred == y[:, np.newaxis], axis=1)
            acc.extend(correct.tolist())
    acc = np.mean(acc)
    return acc


def main(args, i, save_path):
    seed_all(42)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    # load data
    train_loader, val_loader, test_loader = schneider_loader(args)
    print("Data loaded.")

    # create model
    model = MLP(args.neuron_nums, args.dropout)
    model.apply(weight_init)

    model.to(device)
    print("Model created.")

    # create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


    print("Start training.")

    sp_best = 0.
    kill_cnt = 0
    for epoch in range(args.epochs):
        tr_loss = []
        model.train()

        # train
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            pred = model(data)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss.append(loss.item())

        tr_loss = np.mean(tr_loss)

        # evulate
        acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}, train loss: {tr_loss:.4f}, val acc: {acc:.4f}")
        if acc > sp_best:
            sp_best = acc
            torch.save(model.state_dict(), save_path+"mlp")
            print("saving new best model")
        # else:
        #     kill_cnt += 1
        #     if kill_cnt >= 3:
        #         print(f"early stop, best acc: {sp_best:.4f}")
        #     break


    print("Start testing.")
    model.load_state_dict(torch.load(save_path+"mlp"))
    acc = evaluate(model, test_loader, device)
    print(f"Test acc: {acc:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neuron_nums", type=int, nargs='+', default=[2048, 1024, 13506])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs("../ckpts", exist_ok=True)
    os.makedirs("../ckpts/task1", exist_ok=True)
    main(args, 1, save_path="../ckpts/task1/")
