import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    "Age":[20,22,25,28,30,32,35,40],
    "Score":[55,60,65,70,72,78,85,90],
    "Salary":[25,28,32,35,40,45,50,60]
})

plt.hist(df["Score"],bins=5)
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

plt.scatter(df["Age"],df["Score"])
plt.title("Age vs Score")
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()