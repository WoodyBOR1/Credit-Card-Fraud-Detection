import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import xgboost as xgb
import joblib
import os

# Configuration
DATA_PATH = r"C:\Users\woody\Documents\projet Data Science\Credit Card Fraud Detection\BDD\creditcard.csv"
MODEL_DIR = "models"
PLOTS_DIR = "evaluation_plots"

for d in [MODEL_DIR, PLOTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Prétraitement
print("Prétraitement...")
# Scaling Amount et Time (les autres variables V1-V28 sont issues d'une PCA et déjà centrées/réduites)
scaler_amount = StandardScaler()
scaler_time = StandardScaler()

df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Définition de X et y
X = df.drop('Class', axis=1)
y = df['Class']

# Split Train/Test avec stratification pour préserver le ratio de fraude
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# --- 1. Isolation Forest (Apprentissage non supervisé / détection d'anomalies) ---
print("\nEntraînement de Isolation Forest...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
iso_forest.fit(X_train)

# IF prédit -1 pour les anomalies et 1 pour normal
y_pred_if = iso_forest.predict(X_test)
y_pred_if = [1 if x == -1 else 0 for x in y_pred_if]

print("Résultats Isolation Forest :")
print(classification_report(y_test, y_pred_if))

# --- 2. Random Forest Classifier ---
print("\nEntraînement de Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Résultats Random Forest :")
print(classification_report(y_test, y_pred_rf))

# --- 3. XGBoost ---
print("\nEntraînement de XGBoost...")
# scale_pos_weight = count(negative) / count(positive) pour compenser le déséquilibre
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("Résultats XGBoost :")
print(classification_report(y_test, y_pred_xgb))

# --- Évaluation Graphique (AUPRC car les classes sont déséquilibrées) ---
def plot_precision_recall(y_true, y_prob, label, color):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color=color, label=f'{label} (AUC = {pr_auc:.3f})')

plt.figure(figsize=(10, 7))
plot_precision_recall(y_test, y_prob_rf, 'Random Forest', 'blue')
plot_precision_recall(y_test, y_prob_xgb, 'XGBoost', 'green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (AUPRC)')
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, 'pr_curves.png'))

# Matrice de confusion pour le meilleur modèle (souvent XGBoost ou RF sur ce dataset)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('Vrai')
plt.xlabel('Prédit')
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_xgb.png'))

# Sauvegarde des modèles
joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_fraud_model.pkl'))
joblib.dump(scaler_amount, os.path.join(MODEL_DIR, 'scaler_amount.pkl'))
joblib.dump(scaler_time, os.path.join(MODEL_DIR, 'scaler_time.pkl'))

print(f"\nModélisation terminée. Modèles dans '{MODEL_DIR}/' et graphiques dans '{PLOTS_DIR}/'.")
