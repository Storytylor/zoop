import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Age":[20,25,np.nan,30,35],
    "Score":[80,90,75,np.nan,95],
    "Category":["Image","Text","Image","Text","Image"]
})

print("Original:\n",df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Score"] = df["Score"].fillna(df["Score"].mean())

df["Age"] = (df["Age"]-df["Age"].min())/(df["Age"].max()-df["Age"].min())
df["Score"] = (df["Score"]-df["Score"].min())/(df["Score"].max()-df["Score"].min())

df = pd.get_dummies(df,columns=["Category"],dtype=int)

print("Preprocessed:\n",df)