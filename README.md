# 📊 Projet Data Science & Machine Learning Portfolio

Bienvenue dans ce projet complet d'analyse de données (EDA) et de Machine Learning, conçu pour démontrer des compétences en manipulation de données, visualisation et modélisation prédictive.

## 🚀 Fonctionnalités du Projet

### 1. 🎯 Analyse Marketing & Segmentation
- **Objectif** : Identifier les segments de clients les plus rentables.
- **Données** : Analyse démographique (Éducation, Statut Marital, Revenus) et comportementale (Campagnes, Dépenses).
- **Outils** : Pandas, Seaborn, Matplotlib, Plotly.

### 2. 🛡️ Détection de Fraude Bancaire (IA)
- **Objectif** : Prédire les transactions suspectes en temps réel.
- **Machine Learning** : Modèle **Random Forest Classifier** entraîné sur des schémas de fraude complexes.
- **Interactive** : Simulateur de prédiction intégré pour tester des scénarios.
- **Outils** : Scikit-learn, NumPy, Joblib.

### 3. 🌐 Dashboard Interactif (Streamlit)
- Interface utilisateur fluide et interactive.
- Filtres en temps réel.
- Visualisations dynamiques avec Plotly.

## 🛠️ Installation Locale

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/VOTRE_NOM/projet-data-science.git
   cd projet-data-science
   ```

2. **Créer un environnement virtuel** (Recommandé) :
   ```bash
   py -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Préparer les données et le modèle** :
   ```bash
   py EDA/marketing_eda.py
   py EDA/bank_eda.py
   py EDA/train_fraud_model.py
   ```

5. **Lancer l'application** :
   ```bash
   streamlit run app.py
   ```

## ☁️ Déploiement

Cette application est prête à être déployée sur **Streamlit Cloud** :
1. Poussez le code sur GitHub.
2. Connectez votre dépôt sur [share.streamlit.io](https://share.streamlit.io).
3. Point d'entrée : `app.py`.

## 🧰 Tech Stack
- **Langage** : Python 3.14+
- **Manipulation** : Pandas, NumPy
- **Visualisation** : Seaborn, Matplotlib, Plotly
- **Machine Learning** : Scikit-Learn
- **Interface** : Streamlit

---
*Projet réalisé dans le cadre d'un portfolio Data Science.*
