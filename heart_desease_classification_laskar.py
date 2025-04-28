#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# # Load Data

# In[3]:


df = pd.read_csv('heart.csv')


# # EDA

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe()


# In[29]:


# 2. Plot distribusi dengan bar chart
counts = df['target'].value_counts().sort_index()
labels = counts.index.astype(str)

plt.figure(figsize=(6,4))
plt.bar(labels, counts.values)
plt.title('Distribusi Kelas Target')
plt.xlabel('Target (0 = Sehat, 1 = Sakit)')
plt.ylabel('Jumlah Sampel')
for i, v in enumerate(counts.values):
    plt.text(i, v + max(counts.values)*0.01, str(v), ha='center')
plt.show()


# In[31]:


correlation_matrix = df.corr()

# 2. Plot heatmap korelasi
plt.figure(figsize=(10, 8))  # Ukuran gambar
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()


# # Data Scaling

# In[7]:


le = LabelEncoder()
df['age'] = le.fit_transform(df['age'])
df['trestbps'] = le.fit_transform(df['trestbps'])
df['age'] = le.fit_transform(df['age'])
df['chol'] = le.fit_transform(df['chol'])
df['age'] = le.fit_transform(df['age'])
df['thalach'] = le.fit_transform(df['thalach'])
df['thalach'] = le.fit_transform(df['thalach'])
df.head()


# # Split Data

# In[8]:


X = df.drop(['target'], axis=1)
y = df['target']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # Model dan Evaluasi

# ## LightGBM

# In[19]:


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',  # Karena ini adalah masalah klasifikasi biner
    'metric': 'binary_error',  # Kita menggunakan error sebagai metric
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree (GBDT)
    'num_leaves': 31,  # Jumlah leaves dalam tree
    'learning_rate': 0.1,  # Learning rate
    'feature_fraction': 0.9,  # Persentase fitur yang dipilih untuk tiap iterasi
}

bst = lgb.train(params,
                train_data,
                valid_sets=[test_data],  # Data validasi
                num_boost_round=100,  # Jumlah iterasi
                callbacks=[lgb.early_stopping(stopping_rounds=50)]) # Jika tidak ada peningkatan dalam 50 iterasi, pelatihan dihentikan


# ## Decision tree

# In[20]:


dt_model = DecisionTreeClassifier(random_state=42)

# 2. Training model dengan data training
dt_model.fit(X_train, y_train)


# # Evaluasi

# In[22]:


y_pred_lgb = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary_lgb = (y_pred_lgb >= 0.5).astype(int)
accuracy_tuned = accuracy_score(y_test, y_pred_binary_lgb)
print(f'🟢 Akurasi Lightgbm: {accuracy_tuned:.4f}')
print('🟢 Classification Report Lightgbm:')
print(classification_report(y_test, y_pred_binary_lgb))


y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]  # Untuk ROC
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'🟢 Akurasi Decision Tree: {accuracy_dt:.4f}')
print('🟢 Classification Report Decision Tree:')
print(classification_report(y_test, y_pred_dt))


# In[27]:


# Menghitung ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_lgb)

# Menghitung AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Garis diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[28]:


# Hitung ROC Curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr_dt, tpr_dt, color='blue', label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Garis random
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Decision Tree')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




