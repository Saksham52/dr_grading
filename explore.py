import pandas as pd
df = pd.read_csv('data/raw/train.csv')
print(df.head(10))
print(df.shape)
print(df['diagnosis'].value_counts())