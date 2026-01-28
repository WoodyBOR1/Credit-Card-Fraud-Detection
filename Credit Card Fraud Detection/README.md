# Portfolio Data Science : Analyse de Fraude Bancaire 💳

Ce projet présente une Analyse Exploratoire des Données (EDA) sur un dataset de transactions par carte de crédit, avec pour objectif d'identifier des schémas de fraude.

## Objectifs du Projet
- **Compréhension des données** : Analyse de la distribution des transactions et du déséquilibre des classes.
- **Visualisation** : Création de graphiques clairs pour illustrer les différences entre transactions normales et frauduleuses.
- **Dashboard Interactif** : Mise en œuvre d'un tableau de bord via Panel + Pyodide, déployable sur GitHub Pages.

## Structure du Projet
- `BDD/` : Contient le dataset original.
- `eda_fraud.py` : Script Python pour générer les analyses statiques.
- `modeling_fraud.py` : Entraînement des modèles ML (Random Forest, XGBoost, Isolation Forest).
- `predict_fraud.py` : Script de démonstration de prédiction sur de nouvelles données.
- `dashboard.py` : Code source du tableau de bord interactif.
- `prep_lite_data.py` : Script de préparation des données pour le web.
- `docs/` : Dossier contenant la version exportée du dashboard pour GitHub Pages.
- `models/` : Dossier contenant les modèles entraînés (XGBoost) et les scalers.
- `evaluation_plots/` : Courbes PR et matrices de confusion.

## Technologies
- **Analyse** : Pandas, NumPy, Seaborn, Matplotlib.
- **Machine Learning** : Scikit-learn, XGBoost, Imbalanced-learn (SMOTE/Stratification).
- **Dashboard** : Panel, Holoviews, HvPlot, Pyodide/PyScript.

## Modélisation ML
Nous avons testé trois approches :
1. **Isolation Forest** : Détection d'anomalies (non supervisé).
2. **Random Forest** : Classification avec poids de classe équilibrés.
3. **XGBoost** : Classification avec pondération des classes positives (meilleurs résultats sur l'AUPRC).

Les modèles sont évalués avec l'**AUPRC (Area Under Precision-Recall Curve)**, car le dataset est extrêmement déséquilibré.

## Comment exécuter
1. Installer les dépendances : `pip install pandas matplotlib seaborn panel scikit-learn xgboost joblib`
2. Lancer l'EDA : `python eda_fraud.py`
3. Lancer la modélisation : `python modeling_fraud.py`
4. Tester une prédiction : `python predict_fraud.py`
5. Lancer le dashboard : `panel serve dashboard.py`
