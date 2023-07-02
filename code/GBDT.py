import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.tree import export_graphviz
import graphviz


# 读取数据集
df = pd.read_excel('QWE\QWE.xlsx')
df['age0-6'] = 0
df['age6-14'] = 0
df['age>14'] = 0

df.loc[df['Customer_Age_inmonths'] <= 6, 'age0-6'] = 1
df.loc[(df['Customer_Age_inmonths'] > 6) & (df['Customer_Age_inmonths'] <= 14), 'age6-14'] = 1
df.loc[df['Customer_Age_inmonths'] > 14, 'age>14'] = 1

# 提取特征和目标变量
X = df.drop(['Churn','ID', 'Customer_Age_inmonths'], axis=1)  # 特征
y = df['Churn']  # 目标变量
y = np.where(y == 0, 1, 0)

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
model = GradientBoostingClassifier(n_estimators = 50, max_depth=5, min_samples_split = 200, random_state=42)

# 训练模型
model.fit(X_resampled, y_resampled)

# 预测概率
y_pred = model.predict(X_test)

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

# 提取特征重要性
feature_importance = model.feature_importances_

# 创建特征重要性DataFrame
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# 按照重要性排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # 绘制特征重要性条形图
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()



# base_tree = model.estimators_[0]
# dot_data = export_graphviz(base_tree, out_file=None, feature_names=feature_names,
#                            class_names=['0', '1'], filled=True, rounded=True,
#                            special_characters=True, max_depth=4)  # 指定最大深度为2，可根据需要调整

# graph = graphviz.Source(dot_data)
# graph.render("subtree")  # 保存为PDF文件
# graph.view()  # 在默认的图形查看器中显示图像