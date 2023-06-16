import sys
sys.path.append("..")
import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import molecule_eval_loader
from models import GAT
import os, argparse
from sklearn.metrics import mean_squared_error
from utils import seed_all
os.environ["DGLBACKEND"] = "pytorch"


def evaluate(model, data_loader, device, k=10):
    model.eval()
    mse = []
    with torch.no_grad():
        for g, y in data_loader:
            g, y = g.to(device), y.to(device)
            pred = model(g)
            pred = pred.detach().cpu().numpy().astype('float64')
            y = y.detach().cpu().squeeze(-1).numpy()
            err = mean_squared_error(y, pred.reshape(y.shape[0], -1))
            mse.append(err)
    mse = np.mean(mse)
    return mse


def main(args, save_path):
    seed_all(42)
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")

    # load data
    train_loader, test_loader = molecule_eval_loader(args)
    print("Data loaded.")

    # create model

    model = GAT(in_dim=2048, hidden_dim=128, out_dim=1, num_heads=2)

    model.to(device)
    print("Model created.")

    # create loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


    print("Start training.")

    sp_best = 0.
    kill_cnt = 0
    for epoch in range(args.epochs):
        tr_loss = []
        model.train()

        # train
        for g, costs in train_loader:
            g, costs = g.to(device), costs.to(device)

            pred = model(g)
            loss = criterion(pred.view(costs.shape[0], -1), costs.squeeze(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss.append(loss.item())

        tr_loss = np.mean(tr_loss)

        # evulate
        mse = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}, train loss: {tr_loss:.4f}, val mse: {mse:.4f}")
        if mse < sp_best:
            sp_best = mse
            torch.save(model.state_dict(), save_path+"gat")
            print("saving new best model")
        # else:
        #     kill_cnt += 1
        #     if kill_cnt >= 3:
        #         print(f"early stop, best acc: {sp_best:.4f}")
        #     break


    print("Start testing.")
    model.load_state_dict(torch.load(save_path+"gat"))
    mse = evaluate(model, test_loader, device)
    print(f"Test acc: {mse:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("../ckpts", exist_ok=True)
    os.makedirs("../ckpts/task2", exist_ok=True)
    main(args, save_path="../ckpts/task2/")