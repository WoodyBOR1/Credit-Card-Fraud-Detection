import pandas as pd
import os

DATA_PATH = r"C:\Users\woody\Documents\projet Data Science\Credit Card Fraud Detection\BDD\creditcard.csv"
LITE_PATH = "creditcard_lite.csv"

if os.path.exists(DATA_PATH):
    print("Création d'une version allégée du dataset pour le dashboard...")
    df = pd.read_csv(DATA_PATH)
    
    # Garder toutes les fraudes
    frauds = df[df['Class'] == 1]
    # Prendre un échantillon de transactions normales
    normals = df[df['Class'] == 0].sample(n=10000, random_state=42)
    
    df_lite = pd.concat([frauds, normals]).sample(frac=1).reset_index(drop=True)
    df_lite.to_csv(LITE_PATH, index=False)
    print(f"Version allégée sauvegardée : {LITE_PATH} ({len(df_lite)} lignes)")
else:
    print("Dataset introuvable.")
