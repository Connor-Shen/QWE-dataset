import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.float()
        return self.model(x)

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()
        return self.model(x)

# 定义生成对抗网络模型
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        return self.discriminator(self.generator(x))
    


# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')
df1 = pd.read_excel('QWE\QWE.xlsx')
df1 = df1.drop(['ID'], axis=1)
df = df[df['Churn'] == 1]

# df['age0-6'] = 0
# df['age6-14'] = 0
# df['age>14'] = 0

# df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
# df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
# df.loc[df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# 提取特征和目标变量
X = df.drop(['Churn','ID'], axis=1).values  # 特征
feature_names = df.drop(['Churn','ID'], axis=1).columns

y = df['Churn'].values.reshape(-1,1) # 目标变量
# # 将混淆矩阵中阳性转化为流失客户
# y = np.where(y == 0, 1, 0)
# 数据标准化
mm = MinMaxScaler()
X = mm.fit_transform(X)


# 定义损失函数和优化器
criterion = nn.BCELoss()
input_dim = X.shape[1]  # 输入维度
output_dim = 1  # 输出维度（二分类）
generator = Generator(input_dim, output_dim)
generator_fake = Generator(input_dim, input_dim)
discriminator = Discriminator(input_dim)
gan_model = GAN(generator, discriminator)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)



# 将数据转换为TensorDataset格式
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# 定义数据加载器
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    for i, (batch_X, batch_y) in enumerate(data_loader):
        batch_size = batch_X.size(0)
        # 更新判别器
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        # 生成真实样本和生成样本
        real_samples = batch_X
        fake_samples = generator_fake(torch.randn(batch_size, input_dim))
        # 计算判别器的损失函数
        real_loss = criterion(discriminator(real_samples), real_labels)
        fake_loss = criterion(discriminator(fake_samples), fake_labels)
        discriminator_loss = real_loss + fake_loss
        # 更新判别器参数
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # 更新生成器
        generator_optimizer.zero_grad()
        # 生成样本
        fake_samples = generator_fake(torch.randn(batch_size, input_dim))
        # 计算生成器的损失函数
        generator_loss = criterion(discriminator(fake_samples), real_labels)
        # 更新生成器参数
        generator_loss.backward()
        generator_optimizer.step()
        # 打印损失等信息
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {discriminator_loss.item():.4f}, "
                f"Generator Loss: {generator_loss.item():.4f}")


num_samples = 5000
generator.eval()
# 生成虚拟数据
with torch.no_grad():
    generator.eval()
    generated_samples = generator_fake(torch.randn(num_samples, input_dim))

generated_data = generated_samples.numpy()

generated_df = pd.DataFrame(generated_data, columns=feature_names)
generated_df['Churn'] = 1  # 添加标签列，假设生成的数据的标签为1
merged_df = pd.concat([df1, generated_df], ignore_index=True)


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


merged_df['age0-6'] = 0
merged_df['age6-14'] = 0
merged_df['age>14'] = 0

merged_df.loc[merged_df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
merged_df.loc[(merged_df['Customer_Age_inmonths'] > 6) & (merged_df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
merged_df.loc[merged_df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# df['Support_M0_SP_M0'] = 0
# df['Support_M01_SP_M01'] = 0

# df.loc[(df['Support_M01'] > df['Support_M01'].mean()) & (df['Support_M0'] < df['Support_M0'].mean()), 'Support_M0_SP_M0'] = 1
# df.loc[(df['SP_M01'] > df['SP_M01'].mean()) & (df['SP_M0'] < df['SP_M0'].mean()), 'Support_M01_SP_M01'] = 1

# 提取特征和目标变量
X = merged_df.drop(['Churn'], axis=1).values  # 特征
feature_names = merged_df.drop(['Churn'], axis=1).columns

y = merged_df['Churn'].values.reshape(-1,1) # 目标变量
# 将混淆矩阵中阳性转化为流失客户
y = np.where(y == 0, 1, 0)
# 数据标准化
mm = MinMaxScaler()
X = mm.fit_transform(X)



# 定义超参数
input_size = X.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 20
batch_size = 64

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 创建数据加载器
train_dataset = CustomDataset(X_train, y_train.reshape(-1,1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # 初始化模型、损失函数和优化器
model = NeuralNet(input_size, hidden_size, output_size)

# 初始化模型、损失函数和优化器
# model = MLP(input_size, 128, 128 , output_size)

# 损失函数中引入权重
weights = torch.tensor([0.083], dtype=torch.float32)
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

test_data = pd.read_excel('QWE/QWE.xlsx')
test_data['age0-6'] = 0
test_data['age6-14'] = 0
test_data['age>14'] = 0

test_data.loc[test_data['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
test_data.loc[(test_data['Customer_Age_inmonths'] > 6) & (test_data['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
test_data.loc[test_data['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# 提取特征和目标变量
X = test_data.drop(['Churn','ID'], axis=1).values  # 特征
y = test_data['Churn'].values.reshape(-1,1)  # 目标变量
# # 将混淆矩阵中阳性转化为流失客户
# y = np.where(y == 0, 1, 0)

# 数据标准化
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test.values

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

# # 绘制AUC曲线
# plt.plot(fpr, tpr, label='AUC Curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend()
# plt.show()


# 计算PR曲线下面积（AUC）
area = auc(recall, precision)

# 绘制PR曲线
plt.plot(recall, precision, label='PR Curve (AUC = %0.2f)' % area)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()