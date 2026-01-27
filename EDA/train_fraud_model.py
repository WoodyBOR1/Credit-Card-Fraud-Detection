import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Configuration
os.makedirs('models', exist_ok=True)
os.makedirs('EDA/bank/results', exist_ok=True)

print("--- Chargement des données pour le Machine Learning ---")
df = pd.read_csv('BDD/bank_transactions.csv')

# 1. Prétraitement
# On ne garde pas les IDs pour l'entraînement
cols_to_drop = ['TransactionID', 'CustomerID', 'Date']
X = df.drop(columns=cols_to_drop + ['IsFraud'])
y = df['IsFraud']

# Encodage des variables catégorielles
le_dict = {}
cat_cols = ['Category', 'Location', 'Source']

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# 2. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Taille du train : {len(X_train)}")
print(f"Taille du test : {len(X_test)}")

# 3. Entraînement Random Forest
print("\n--- Entraînement du modèle Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 4. Évaluation
y_pred = rf_model.predict(X_test)
report = classification_report(y_test, y_pred)
print("\nRapport de Classification :")
print(report)

# Importance des variables
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nImportance des variables :")
print(feature_importance)

# 5. Sauvegarde
joblib.dump(rf_model, 'models/fraud_rf_model.joblib')
joblib.dump(le_dict, 'models/fraud_encoders.joblib')
feature_importance.to_csv('EDA/bank/results/feature_importance.csv', index=False)

print("\nModèle et encodeurs sauvegardés dans le dossier 'models/'.")
print("Importance des variables sauvegardée dans 'EDA/bank/results/feature_importance.csv'.")
