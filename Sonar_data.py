import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('E:\\machine earning\\sonar.csv', header=None)
print("Read Data: ",sonar_data)
# head is used to show five row from top
print(sonar_data.head())
print(sonar_data.describe())