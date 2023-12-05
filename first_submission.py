from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# Chargement des données
df = pd.read_csv("titanic_data/train.csv")
test_df = pd.read_csv("titanic_data/test.csv")

# Préparation des données
y = df["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(test_df[features])

# Division des données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Initialisation et entraînement du modèle
model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de validation
predictions_val = model.predict(X_val)

# Évaluation des performances sur l'ensemble de validation
accuracy = accuracy_score(y_val, predictions_val)
print(f'Accuracy on validation set: {accuracy:.2f}')

# Prédictions sur l'ensemble de test
predictions_test = model.predict(X_test)

# Enregistrement des prédictions dans un fichier CSV
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions_test})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
