
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic_data/train.csv")

for mot in df.head():
    print(mot)

for categorie in df.head():
    if categorie not in ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']:
        sns.countplot(x=categorie, hue='Survived', data=df)
        plt.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
