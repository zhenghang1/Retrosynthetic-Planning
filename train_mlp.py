import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from my_models import MLP
import torch.nn as nn

import pickle

# 加载数据
with open('./data/MoleculeEvaluationData/train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('./data/MoleculeEvaluationData/test.pkl', 'rb') as f:
    test_data = pickle.load(f)

packed_fp = train_data['packed_fp']
train_finger = np.unpackbits(packed_fp, axis=1)
train_values = train_data['values']
packed_fp = test_data['packed_fp']
test_finger = np.unpackbits(packed_fp, axis=1)
test_values = test_data['values']

# 转换为 PyTorch 张量
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(train_finger).float().to(device)
y_train = train_values.to(device)
X_test = torch.from_numpy(test_finger).float().to(device)
y_test = test_values.to(device)

# 将训练数据转换为 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型，并将其转换为 CUDA 张量
model = MLP(input_size=2048, hidden_sizes=[512, 256], output_size=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 在测试集上评估模型
def get_test_loss():
    with torch.no_grad():
    # 将数据转换为 CUDA 张量
        y_pred = model(X_test)
        mse = criterion(y_pred, y_test)
    return mse.item()

# 训练模型
min_loss=1000000000
for epoch in range(100):
    loss_epoch=0.0
    for X_batch, y_batch in train_loader:
        # 将数据转换为 CUDA 张量
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播计算损失
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch+=loss.item()
    # 打印每个 epoch 的损失
    print("Epoch [{}/{}], train_loss: {:.6f}".format(epoch + 1, 100, loss_epoch))
    if epoch%1==0:
        test_loss=get_test_loss()
        print('test_loss:{:.6f}'.format(test_loss))
        if test_loss<min_loss:
            torch.save(model.state_dict(), 'MLP_model.pt')

