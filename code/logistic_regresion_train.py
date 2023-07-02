import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')

df['age0-6'] = 0
df['age6-14'] = 0
df['age>14'] = 0

df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
df.loc[df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

df['Support_M0_SP_M0'] = df['Support_M0'] * df['SP_M0']
df['Support_M01_SP_M01'] = df['Support_M01'] * df['SP_M01']

# 提取特征和目标变量
X = df.drop(['Churn','ID'], axis=1)  # 特征
feature_names = X.columns
y = df['Churn']  # 目标变量
# 将混淆矩阵中阳性转化为流失客户
y = np.where(y == 0, 1, 0)

# 数据标准化
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 使用SMOTE算法处理样本不平衡
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # 使用BorderlineSMOTE算法处理样本不平衡
# from imblearn.over_sampling import BorderlineSMOTE
# blsm = BorderlineSMOTE(random_state=42,kind="borderline-1")
# X_resampled, y_resampled = blsm.fit_resample(X_train, y_train)

# 使用ADASYN算法处理样本不平衡
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42)
X_resampled, y_resampled = ada.fit_resample(X_train, y_train)


# # 对训练集进行欠采样
# undersampler = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=20, C = 2.0, class_weight='balanced')

# 训练模型
model.fit(X_resampled, y_resampled)

# 预测
y_pred = model.predict(X_test)

# 评估预测性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
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


y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# 假设y_scores为模型的预测概率得分，y_true为真实标签
y_scores = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

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


# # 打印系数值
# coefficients = model.coef_[0]
# print("Coefficients:")
# for i, coef in enumerate(coefficients):
#     feature_name = feature_names[i]
#     print(f"{feature_name}: {coef}")

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.figure(figsize=(10, 6))
# plt.contourf(xx, yy, Z, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Logistic Regression Decision Boundary')
# plt.colorbar()

# plt.show()
