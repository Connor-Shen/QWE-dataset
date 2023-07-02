from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')

df['age0-6'] = 0
df['age6-14'] = 0
df['age14-35'] = 0
df['age>35'] = 0

df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
df.loc[df['Customer_Age_inmonths'] > 14, 'age14-35'] = 1
df.loc[df['Customer_Age_inmonths'] > 35, 'age>35'] = 1

# 提取特征和目标变量
X = df.drop(['Churn','ID'], axis=1)  # 特征
y = df['Churn']  # 目标变量
# 将混淆矩阵中阳性转化为流失客户
# y = np.where(y == 0, 1, 0)

# 数据标准化
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))
# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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


# svm_model1 = SVC(C=1, kernel='linear')
# svm_model2 = SVC(C=1, kernel='linear', class_weight={0:1, 1:50})
# svm_model3 = SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={0:1, 1:2})
# svm_model4 = SVC(C=0.8, kernel='rbf', gamma=0.5, class_weight={0:1, 1:10})


# 初始化SVM模型
svm_model = SVC(class_weight = "balanced", probability=True)

# 训练模型
svm_model.fit(X_resampled, y_resampled)

# 预测结果
y_pred = svm_model.predict(X_test)

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

from sklearn.metrics import roc_curve, auc, precision_recall_curve


# Obtain predicted probabilities
y_pred_prob = svm_model.predict_proba(X_test)[:, 1]

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
