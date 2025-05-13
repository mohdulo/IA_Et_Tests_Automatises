import joblib
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python predict.py chemin_vers_input.pkl")
    sys.exit(1)

input_path = sys.argv[1]

# Charger le modèle
model = joblib.load('random_forest_churn_model.pkl')

# Charger les données d'entrée au format Pickle
try:
    new_data = pd.read_pickle(input_path)
except Exception as e:
    print(f"Erreur lors du chargement du fichier : {e}")
    sys.exit(1)

# Affichage des données
print("✅ Données reçues :")
print(new_data.head())

# Faire la prédiction
prediction = model.predict(new_data)

# Afficher le résultat
resultat = 'Client va résilier (Churn)' if prediction[0] == 1 else 'Client fidèle'
print(f"\n🎯 Prédiction : {resultat}")
