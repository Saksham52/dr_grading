import pandas as pd
df = pd.read_csv('data/raw/train.csv')
for grade in range(5):
    sample = df[df['diagnosis'] == grade].iloc[0]['id_code']
    print(f"Grade {grade}: {sample}.png")