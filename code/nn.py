import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')

df['age0-6'] = 0
df['age6-14'] = 0
df['age>14'] = 0

df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
df.loc[df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# df['Support_M0_SP_M0'] = 0
# df['Support_M01_SP_M01'] = 0

# df.loc[(df['Support_M01'] > df['Support_M01'].mean()) & (df['Support_M0'] < df['Support_M0'].mean()), 'Support_M0_SP_M0'] = 1
# df.loc[(df['SP_M01'] > df['SP_M01'].mean()) & (df['SP_M0'] < df['SP_M0'].mean()), 'Support_M01_SP_M01'] = 1

# 提取特征和目标变量
X = df.drop(['Churn','ID'], axis=1).values  # 特征
feature_names = df.drop(['Churn','ID'], axis=1).columns

y = df['Churn'].values.reshape(-1,1) # 目标变量
# 将混淆矩阵中阳性转化为流失客户
y = np.where(y == 0, 1, 0)
# 数据标准化
mm = MinMaxScaler()
X = mm.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# from imblearn.over_sampling import BorderlineSMOTE
# blsm = BorderlineSMOTE(random_state=42,kind="borderline-1")
# X_resampled, y_resampled = blsm.fit_resample(X_train, y_train)

# # 使用ADASYN算法处理样本不平衡
# from imblearn.over_sampling import ADASYN
# ada = ADASYN(random_state=42)
# X_resampled, y_resampled = ada.fit_resample(X_train, y_train)

# 定义超参数
input_size = X.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 30
batch_size = 64

# 创建数据加载器
train_dataset = CustomDataset(X_train, y_train.reshape(-1,1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # 初始化模型、损失函数和优化器
model = NeuralNet(input_size, hidden_size, output_size)

# 初始化模型、损失函数和优化器
# model = MLP(input_size, 128, 128 , output_size)

# 损失函数中引入权重
weights = torch.tensor([0.063], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 模型训练
train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(train_dataset)
    train_losses.append(epoch_loss)


    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 模型预测
with torch.no_grad():
    model.eval()
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred_prob = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    predicted = (outputs >= 0.3).squeeze().numpy().astype(int)


# 打印混淆矩阵和召回率
cm = confusion_matrix(y_test, predicted)
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print("Confusion Matrix:")
print(cm)

# 计算召回率（Recall）
recall = cm[0][0] / (cm[0][0] + cm[0][1])
print("Recall:", recall)

# 计算精确率（Precision）
precision = cm[0][0] / (cm[0][0] + cm[1][0])
print("Precision:", precision)

# 计算Fb分数
b = 2
f1_score = (1+b**2) * (precision * recall) / (b**2*precision + recall)
print("Fb Score:", f1_score)


# Calculate fpr, tpr, and thresholds for AUC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Calculate precision, recall, and thresholds for PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

# 绘制AUC曲线
plt.plot(fpr, tpr, label='AUC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# # 计算PR曲线下面积（AUC）
# area = auc(recall, precision)

# # 绘制PR曲线
# plt.plot(recall, precision, label='PR Curve (AUC = %0.2f)' % area)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.show()


# # 获取预测结果的概率
# probabilities = torch.softmax(outputs, dim=1)

# # 获取最有可能 churn 的前 50 人的索引
# top_50_indices = torch.argsort(probabilities[:, 0], descending=True)[:50]

# # 获取最有可能 churn 的前 50 人的特征数据
# top_50_features = X_test[top_50_indices]

# # 创建一个空的 DataFrame
# data = pd.DataFrame()

# # 添加特征数据和预测概率
# for i, feature in enumerate(feature_names):
#     data[feature] = top_50_features[:, i]

# data["Churn Probability"] = probabilities[top_50_indices].numpy()
# data["Churn"] = y_test[top_50_indices]

# # 保存数据到 Excel 表格
# data.to_excel("top_50_churn_customers1.xlsx", index=False)