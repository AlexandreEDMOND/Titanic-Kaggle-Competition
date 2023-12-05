
import pandas as pd

df = pd.read_csv("titanic_data/train.csv")

men = df.loc[df.Sex == 'male']["Survived"]
print(sum(men))