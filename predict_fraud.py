import pandas as pd
import joblib
import os
import numpy as np

# Chemins
MODEL_PATH = 'models/xgb_fraud_model.pkl'
SCALER_AMOUNT_PATH = 'models/scaler_amount.pkl'
SCALER_TIME_PATH = 'models/scaler_time.pkl'
DATA_LITE = 'creditcard_lite.csv'

def run_prediction():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_AMOUNT_PATH, SCALER_TIME_PATH]):
        print("Erreur : Modèles ou scalers introuvables. Lancez modeling_fraud.py d'abord.")
        return

    print("--- Chargement du système de détection ---")
    model = joblib.load(MODEL_PATH)
    scaler_amount = joblib.load(SCALER_AMOUNT_PATH)
    scaler_time = joblib.load(SCALER_TIME_PATH)

    # Charger des données de test
    df = pd.read_csv(DATA_LITE)
    
    # Prendre un mix de fraudes et transactions normales
    sample_fraud = df[df['Class'] == 1].sample(2)
    sample_normal = df[df['Class'] == 0].sample(3)
    test_samples = pd.concat([sample_fraud, sample_normal]).sample(frac=1)

    print(f"\nTest sur {len(test_samples)} transactions (mélange fraudes/normales) :")
    
    for i, (idx, row) in enumerate(test_samples.iterrows()):
        true_class = row['Class']
        
        # Préparation des features
        features = row.drop('Class').to_frame().T
        
        # Scaling
        features['scaled_amount'] = scaler_amount.transform(features['Amount'].values.reshape(-1, 1))
        features['scaled_time'] = scaler_time.transform(features['Time'].values.reshape(-1, 1))
        
        # Drop colonnes originales
        features_final = features.drop(['Time', 'Amount'], axis=1)
        
        # Prédiction
        pred = model.predict(features_final)[0]
        prob = model.predict_proba(features_final)[0][1]
        
        status = "FRAUDE 🚩" if pred == 1 else "NORMALE ✅"
        result_icon = "✔️ Correct" if pred == true_class else "❌ Erreur"
        
        print(f"Transaction {i+1} : Montant {row['Amount']:.2f}€ | Prédiction: {status} ({prob:.2%} proba) | Réel: {true_class} | {result_icon}")

if __name__ == "__main__":
    run_prediction()
