import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
data = sns.load_dataset('iris')
sns.lineplot(x='sepal_length', y='sepal_width', data=data)
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
dt.head()
dt.shape
dt.value_counts()
X = dt.iloc[:,:]
y = dt.iloc[:,:]
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=71)
X_train
y_train
X_test
y_test
X_train.shape
y_train.shape
X_test.shape
y_test.shape
model = LogisticRegression()
scaler = StandardScaler()