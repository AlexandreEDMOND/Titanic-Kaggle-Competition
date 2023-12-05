# Importation des bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Chargement des données
train_data = pd.read_csv("titanic_data/train.csv")
test_data = pd.read_csv("titanic_data/test.csv")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Création et entraînement du modèle Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluation des performances du modèle
accuracy = accuracy_score(y_test, predictions)
print(f"Taux de réussite : {accuracy * 100:.2f}%")
