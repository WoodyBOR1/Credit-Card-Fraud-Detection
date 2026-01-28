import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_PATH = r"C:\Users\woody\Documents\projet Data Science\Credit Card Fraud Detection\BDD\creditcard.csv"
OUTPUT_DIR = "eda_plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Chargement des données...")
df = pd.read_csv(DATA_PATH)

print(f"Dimensions du dataset : {df.shape}")
print("\nAperçu des données :")
print(df.head())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nVérification des valeurs manquantes :")
print(df.isnull().sum().max())

# Distribution de la variable cible
print("\nDistribution des classes (0: Normal, 1: Fraude) :")
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='viridis')
plt.title('Distribution des Classes (0: Normal, 1: Fraude)')
plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))

# Distribution de Time et Amount
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Time'], bins=50, kde=True, color='blue')
plt.title('Distribution du Temps')

plt.subplot(1, 2, 2)
sns.histplot(df['Amount'], bins=50, kde=True, color='red')
plt.yscale('log')
plt.title('Distribution des Montants (log scale)')

plt.savefig(os.path.join(OUTPUT_DIR, 'time_amount_distribution.png'))

# Corrélation
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Matrice de Corrélation')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))

# Boxplots pour les variables les plus corrélées (exemples : V17, V14, V12, V10 sont souvent corrélées à Class)
top_corr_vars = ['V17', 'V14', 'V12', 'V10', 'V16', 'V11', 'V4', 'V2']
plt.figure(figsize=(16, 12))
for i, var in enumerate(top_corr_vars):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Class', y=var, data=df)
    plt.title(f'{var} vs Class')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_correlated_vars.png'))

print("\nEDA Terminée. Les graphiques sont sauvegardés dans 'eda_plots/'.")
