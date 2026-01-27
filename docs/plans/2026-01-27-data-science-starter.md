# Data Science Starter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Mettre en place un environnement complet pour l'Analyse Exploratoire des Données (EDA) et la création d'un tableau de bord interactif.

**Architecture:** Création d'un dataset synthétique, d'un script d'analyse statistique et de visualisation, et d'une application Streamlit pour la présentation des résultats.

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Streamlit.

---

### Task 1: Préparation de l'environnement

**Files:**
- Create: `requirements.txt`
- Create: `EDA/generate_data.py`

**Step 1: Installer les bibliothèques nécessaires**
Run: `py -m pip install -r requirements.txt`

**Step 2: Générer le jeu de données initial**
Run: `py EDA/generate_data.py`
Expected: Création du fichier `EDA/sales_data.csv`.

---

### Task 2: Analyse Statistique et Visualisation

**Files:**
- Create: `EDA/analysis.py`

**Step 1: Exécuter l'analyse EDA**
Run: `py EDA/analysis.py`
Expected: Sortie console des statistiques et création de fichiers `.png` dans `EDA/plots/`.

---

### Task 3: Tableau de Bord Interactif

**Files:**
- Create: `EDA/dashboard.py`

**Step 1: Lancer le dashboard Streamlit**
Run: `streamlit run EDA/dashboard.py`
Expected: Ouverture d'un onglet navigateur avec le dashboard interactif.
