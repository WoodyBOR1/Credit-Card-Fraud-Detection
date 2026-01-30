# 🛡️ Détection de Fraude sur Carte Bancaire

[![Dashboard](https://img.shields.io/badge/Live-Dashboard-blue?style=for-the-badge&logo=github)](https://WoodyBOR1.github.io/Credit-Card-Fraud-Detection/)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)

## 🚀 Présentation du Projet
Ce projet est une solution complète de **Data Science** pour la détection de transactions frauduleuses. Il couvre l'ensemble du pipeline, de l'**Analyse Exploratoire des Données (EDA)** à la mise en production via un **Dashboard interactif (Panel + Pyodide)**.

Le dataset utilisé contient des transactions effectuées par des titulaires de cartes européennes, où les fraudes ne représentent que **0.172%** de l'ensemble des données, posant un défi majeur de classification déséquilibrée.

---

## 📊 Fonctionnalités Clés
- **EDA Approfondie** : Analyse de la distribution des montants, du temps et des variables PCA (V1-V28).
- **Modélisation ML Avancée** :
  - **Isolation Forest** (Non supervisé).
  - **Random Forest** (Poids de classes ajustés).
  - **XGBoost** (Optimisé pour l'AUPRC).
- **Dashboard Interactif** : Une interface web tournant entièrement dans le navigateur (serverless) pour explorer les données en temps réel.
- **Prédiction en Temps Réel** : Script prêt à l'emploi pour évaluer de nouvelles transactions.

---

## 🛠️ Installation et Utilisation

### 1. Cloner le projet
```bash
git clone https://github.com/WoodyBOR1/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```
*Note : Si le fichier n'est pas présent, utilisez :*
`pip install pandas matplotlib seaborn panel scikit-learn xgboost joblib hvplot holoviews`

### 3. Exécuter les composants
- **Générer l'analyse statique** : `python eda_fraud.py`
- **Entraîner les modèles** : `python modeling_fraud.py`
- **Lancer le dashboard localement** : `panel serve dashboard.py --show`

---

## 🌐 Déploiement GitHub Pages
Le dashboard est automatiquement déployé via le dossier `docs/`. 
Si vous souhaitez le redéployer sur votre propre compte :
1. Allez dans **Settings** > **Pages**.
2. Source : **Deploy from a branch**.
3. Branch : **main** / Folder : **/docs**.

---

## 📈 Résultats et Évaluation
Étant donné le fort déséquilibre des classes, nous utilisons l'**AUPRC (Area Under Precision-Recall Curve)** comme métrique principale :
- **XGBoost** a montré la meilleure performance pour identifier les fraudes tout en minimisant les faux positifs.
- Les visualisations incluses dans le dashboard permettent d'isoler rapidement les variables les plus discriminantes (comme V17, V14 et V12).

---

## 📁 Structure du Dépôt
- `docs/` : Contient le dashboard web (Index.html + Dataset lite).
- `models/` : Modèles entraînés et scalers sérialisés.
- `eda_plots/` & `evaluation_plots/` : Graphiques d'analyse et de performance.
- `eda_fraud.py`, `modeling_fraud.py`, `dashboard.py` : Scripts sources.

---
**Développé avec ❤️ par Woody B.**
