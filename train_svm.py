import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from my_models import MLP
import torch.nn as nn
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

with open('./data/processed_schneider50k/train.pkl', 'rb') as f:  #39992
    train_data = pickle.load(f)
with open('./data/processed_schneider50k/test.pkl', 'rb') as f:
    test_data = pickle.load(f)
# print(train_data.shape)  #(39992, 2)
# print(test_data.shape)  #(5006, 2)

x_train=train_data[:,0]
y_train=train_data[:,1]

x_test=test_data[:,0]
y_test=test_data[:,1]


x=[i for i in x_train]
x_train=np.vstack(x)
x=[i for i in x_test]
x_test=np.vstack(x)

x_train = x_train.astype(int)
x_test=x_test.astype(int)
y_train = y_train.astype(int)
y_test=y_test.astype(int)

print("x_train.shape",x_train.shape)
print('x_test.shape',x_test.shape)   
#x_train.shape (39992, 2048)
#x_test.shape (5006, 2048)

# 训练svm
# print('训练开始')
# svm = SVC(C=1.0, kernel='rbf',max_iter=200,probability=True)
# svm.fit(x_train, y_train)
# print('训练完成')
# joblib.dump(svm, 'svm_model.joblib')

x_test=x_test[0:100]
y_test=y_test[0:100]
svm = joblib.load('svm_model.joblib')
print('模型加载完毕')

# y_pred = svm.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print('预测的前10个',y_pred[0:10])
# print('模型的准确度为：', accuracy)

# 对测试集中的每个样本进行预测，获取分类器预测的前10个最可能的类别
pred_probs = svm.predict_proba(x_test)
sorted_indices = np.argsort(-pred_probs, axis=1)   #从大到小排序
print('预测完毕')



# 计算Top-1、Top-4、Top-7和Top-10准确率
num_samples = len(y_test)
top1_acc = 0
top4_acc = 0
top7_acc = 0
top10_acc = 0

top_1_labels = sorted_indices[:, 0]
print(top_1_labels.shape)


top_4_labels = sorted_indices[:, :4]
top_7_labels = sorted_indices[:, :7]
top_10_labels = sorted_indices[:, :10]
labels = svm.classes_
print(labels)
top_1_labels_name=labels[top_1_labels]
print('预测的前10个',top_1_labels_name[0:10])
top_4_labels_name=labels[top_4_labels]
top_7_labels_name=labels[top_7_labels]
top_10_labels_name=labels[top_10_labels]


for i in range(num_samples):
    if y_test[i] in top_1_labels_name:
        top1_acc += 1
    if y_test[i] in top_4_labels_name:
        top4_acc += 1
    if y_test[i] in top_7_labels_name:
        top7_acc += 1
    if y_test[i] in top_10_labels_name:
        top10_acc += 1

top1_acc /= num_samples
top4_acc /= num_samples
top7_acc /= num_samples
top10_acc /= num_samples

print('Top-1 Accuracy: {:.4f}%'.format(top1_acc * 100))
print('Top-4 Accuracy: {:.4f}%'.format(top4_acc * 100))
print('Top-7 Accuracy: {:.4f}%'.format(top7_acc * 100))
print('Top-10 Accuracy: {:.4f}%'.format(top10_acc * 100))


