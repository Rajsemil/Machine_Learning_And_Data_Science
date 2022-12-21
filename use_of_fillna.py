import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
# fillna method is used to fill the value bin NaN data
dt.fillna(0)
# fill the value in rowwise
dt.fillna('Id':32, 'Mobille':0)
# for backbard vallue fill
dt.fillna(method = 'bfill')
