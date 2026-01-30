# 🛡️ Détection de Fraude sur Carte Bancaire

## 📝 Présentation du Projet
Ce projet est une démonstration complète d'un pipeline de **Data Science** appliqué à la sécurité financière. L'objectif est d'identifier les transactions frauduleuses parmi un dataset de transactions anonymisées, tout en gérant le défi majeur du **déséquilibre des classes** (0.17% de fraudes).

Le projet inclut une analyse exploratoire, l'entraînement de modèles de Machine Learning performants et le déploiement d'un dashboard interactif en mode **serverless** sur GitHub Pages.

---

## 🛠️ Installation et Dépendances

### Prérequis
- Python 3.10 ou supérieur
- Pip (gestionnaire de paquets)

### Installation locale
1. **Cloner le projet** :
   ```bash
   git clone https://github.com/WoodyBOR1/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Installer les dépendances** :
   ```bash
   pip install pandas matplotlib seaborn panel scikit-learn xgboost joblib hvplot holoviews
   ```

---

## 🚀 Utilisation

### 1. Analyse Exploratoire (EDA)
Le script `eda_fraud.py` génère des graphiques statiques dans le dossier `eda_plots/` pour comprendre les corrélations et les distributions des variables.
```bash
python eda_fraud.py
```

### 2. Modélisation Machine Learning
Le script `modeling_fraud.py` entraîne plusieurs modèles (Isolation Forest, Random Forest, et **XGBoost**). Il sauvegarde le meilleur modèle dans le dossier `models/` et génère des courbes de précision-rappel dans `evaluation_plots/`.
```bash
python modeling_fraud.py
```

### 3. Dashboard Interactif
Le dashboard est conçu pour être visionné directement sur GitHub Pages, mais vous pouvez aussi le lancer localement :
```bash
panel serve dashboard.py --show
```

---

## 📊 Analyse des Résultats
- **Métrique principale** : AUPRC (Area Under Precision-Recall Curve), car la précision est plus cruciale que l'accuracy sur des données déséquilibrées.
- **Modèle retenu** : XGBoost, offrant le meilleur compromis entre détection des fraudes et limitation des faux positifs.
- **Variables discriminantes** : Les graphiques d'analyse montrent que les variables issues de la PCA (V17, V14, V12) sont les plus révélatrices des comportements frauduleux.

---

## 📁 Structure du Projet
- `docs/` : Version web du dashboard (HTML/JS + dataset lite).
- `models/` : Modèles entraînés et scalers sérialisés.
- `eda_plots/` : Graphiques d'analyse exploratoire.
- `evaluation_plots/` : Matrices de confusion et courbes de performance.
- `dashboard.py` : Source du tableau de bord interactif.
- `modeling_fraud.py` : Pipeline d'entraînement ML.

---
**Développé par Woody B.** 🚀  
*Ce projet fait partie d'un portfolio de Data Science : https://woodybor1.github.io/Woody-BORGELLA-Portfolio/.*
