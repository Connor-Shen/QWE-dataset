import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import export_graphviz
import graphviz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')

df['age0-6'] = 0
df['age6-14'] = 0
df['age>14'] = 0

df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
df.loc[df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# 提取特征和目标变量
X = df.drop(['Churn','ID'], axis=1)  # 特征
feature_names = X.columns
y = df['Churn']  # 目标变量
# 将混淆矩阵中阳性转化为流失客户
# y = np.where(y == 0, 1, 0)


# 数据标准化
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""样本不平衡处理"""
# 使用SMOTE算法处理样本不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# # 对训练集进行欠采样
# undersampler = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# # 对训练集进行过采样
# oversampler = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# 初始化随机森林模型
model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42, class_weight='balanced')

# 训练模型
model.fit(X_resampled, y_resampled)

# 预测概率
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 计算AUC值
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc_score)

y_pred = model.predict(X_test)

# 打印混淆矩阵
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

# print(feature_names)
# base_tree = model.estimators_[2]
# dot_data = export_graphviz(base_tree, out_file=None, feature_names=feature_names,
#                            class_names=['0', '1'], filled=True, rounded=True,
#                            special_characters=True, max_depth=4)  # 指定最大深度为2，可根据需要调整

# graph = graphviz.Source(dot_data)
# graph.render("subtree")  # 保存为PDF文件
# graph.view()  # 在默认的图形查看器中显示图像