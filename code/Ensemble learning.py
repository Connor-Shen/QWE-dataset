from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
# 将混淆矩阵中阳性转化为流失客户
y = np.where(y == 0, 1, 0)

# 数据标准化
mm = MinMaxScaler()
X = pd.DataFrame(mm.fit_transform(X))

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""样本不平衡处理"""
# 使用SMOTE算法处理样本不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# 初始化多个分类器
classifier1 = LogisticRegression(max_iter=20, C = 2.0, class_weight='balanced')
classifier2 = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=2, random_state=42, class_weight='balanced')
classifier3 = GradientBoostingClassifier()

# 创建集成学习模型
ensemble_model = VotingClassifier(estimators=[('lr', classifier1), ('rf', classifier2), ('gbdt', classifier3)], voting='soft', weights=[5, 1, 1])

# 训练模型
ensemble_model.fit(X_resampled, y_resampled)


y_pred = ensemble_model.predict(X_test)
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