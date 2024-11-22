import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv("results/zero-shot_cot_True.csv")

s = sum(df["relation"] == df["predict"])
print(s)