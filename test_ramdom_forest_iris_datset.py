from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Chargement du jeu de données Iris (un exemple de jeu de données intégré à scikit-learn)
iris = load_iris()
X = iris.data
y = iris.target

# Création d'un diagramme de dispersion pour visualiser les caractéristiques
plt.figure(figsize=(8, 6))

# Visualisation de la longueur du sépale par rapport à la largeur du sépale
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Longueur du Sépale (cm)')
plt.ylabel('Largeur du Sépale (cm)')
plt.title('Diagramme de Dispersion des Sépales')

# Affichage du diagramme
plt.show()

# Division du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=78)

# Création du modèle de forêt aléatoire
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=78)

# Entraînement du modèle sur l'ensemble d'entraînement
random_forest_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = random_forest_model.predict(X_test)

# Évaluation de la précision du modèle
accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle : {accuracy}")
